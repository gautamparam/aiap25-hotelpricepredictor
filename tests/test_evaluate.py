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
