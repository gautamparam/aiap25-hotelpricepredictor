"""Tests for the evaluation metrics.

`loss_captured_at_budget` is the metric the model is actually chosen on, so it
is worth pinning down at its boundaries: a perfect ranking must capture
everything, and a random one must capture roughly the budget.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (classification_metrics, loss_captured_at_budget,
                          choose_threshold, per_group_metrics)


def test_perfect_ranking_captures_all_the_loss():
    """If the score orders bookings exactly by loss, the top 10% holds the top 10%."""
    losses = np.array([100.0] * 10 + [0.0] * 90)
    scores = losses.copy()
    result = loss_captured_at_budget(scores, losses, budgets=(0.10,))
    assert result.loc[0, "share_of_loss"] == 1.0


def test_random_ranking_captures_about_the_budget():
    """A uniform random score has no information, so capture ~= budget."""
    rng = np.random.default_rng(0)
    losses = rng.random(10_000)
    scores = rng.random(10_000)
    share = loss_captured_at_budget(scores, losses, budgets=(0.20,)).loc[0, "share_of_loss"]
    assert 0.17 < share < 0.23


def test_inverted_ranking_captures_almost_nothing():
    losses = np.array([100.0] * 10 + [0.0] * 90)
    result = loss_captured_at_budget(-losses, losses, budgets=(0.10,))
    assert result.loc[0, "share_of_loss"] == 0.0


def test_lift_over_random_is_reported():
    losses = np.array([1.0] * 100)
    scores = np.arange(100, dtype=float)
    row = loss_captured_at_budget(scores, losses, budgets=(0.25,)).loc[0]
    assert row["lift_over_random"] == np.float64(1.0)  # uniform loss -> no lift possible


def test_accuracy_exposes_the_majority_trap():
    """A model predicting 'never cancels' scores well on accuracy and finds nobody."""
    y = pd.Series([0] * 72 + [1] * 28)
    proba = np.full(100, 0.01)
    metrics = classification_metrics(y, proba, threshold=0.5)
    assert metrics["accuracy"] == 0.72
    assert metrics["recall"] == 0.0


def test_threshold_flags_the_requested_budget():
    proba = np.linspace(0, 1, 1000)
    threshold = choose_threshold(pd.Series([0] * 1000), proba, budget=0.10)
    assert (proba >= threshold).mean() == pytest.approx(0.10, abs=0.01)


def test_per_group_skips_single_class_groups():
    """AUC is undefined for a group with one class; it must be skipped, not crash."""
    y = pd.Series([0, 1, 0, 1, 0, 0])
    proba = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.4])
    groups = pd.Series(["A", "A", "A", "A", "B", "B"])
    out = per_group_metrics(y, proba, groups)
    assert list(out["group"]) == ["A"]


def test_explode_carries_the_originating_booking_index():
    """Room-nights must know which booking they came from.

    Regression test. The original implementation rebuilt this positionally with
    `np.repeat` over the unfiltered frame, but `stayable()` drops rows (nights
    outside 1-30, unparseable dates) and the panel merge drops more. Any
    positional rebuild therefore attaches every room-night after the first
    dropped booking to the WRONG booking, and the failure is silent — the
    totals still balance, only the per-booking attribution is scrambled.
    """
    from src.temporal import explode_room_nights

    df = pd.DataFrame({
        "hotel": ["A", "A", "A"],
        "arrival_date": pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-10"]),
        # the middle booking has an implausible stay and is dropped by stayable()
        "total_nights": [2, 99, 3],
        "is_canceled": [1, 1, 1],
        "adr": [100.0, 100.0, 100.0],
    }, index=[10, 20, 30])

    out = explode_room_nights(df, columns=["adr"], honoured_only=False)
    assert "booking_index" in out.columns
    # 2 nights from booking 10 and 3 from booking 30; booking 20 is excluded.
    assert sorted(out["booking_index"].unique().tolist()) == [10, 30]
    assert (out["booking_index"] == 10).sum() == 2
    assert (out["booking_index"] == 30).sum() == 3


def test_realised_loss_never_exceeds_a_booking_s_own_value():
    """A booking cannot lose more than it was worth.

    The invariant the attribution bug violated: 1,719 bookings were credited
    with a loss larger than their own value before the fix.
    """
    from src.temporal import lost_room_nights

    panel = pd.DataFrame({
        "hotel": ["A"] * 3,
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "occupied": [1, 1, 1], "capacity": [10, 10, 10],
        "empty_rooms": [9, 9, 9], "month": [1, 1, 1],
    })
    bookings = pd.DataFrame({
        "hotel": ["A", "A"],
        "arrival_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "total_nights": [2, 1],
        "is_canceled": [1, 1],
        "adr": [100.0, 250.0],
        "status_days_before_arrival": [5.0, 5.0],
    }, index=[1, 2])

    rn = lost_room_nights(bookings, panel)
    per_booking = rn.groupby("booking_index").apply(
        lambda g: (g["p_lost"] * g["adr"]).sum(), include_groups=False)
    values = bookings["adr"] * bookings["total_nights"]
    for idx, loss in per_booking.items():
        assert loss <= values.loc[idx] + 1e-9, f"booking {idx}: loss {loss} > value {values.loc[idx]}"
