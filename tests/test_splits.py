"""Tests for temporal splitting.

A random split would leak the future into training and inflate every metric.
These tests assert the property that prevents it: chronological ordering.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.splits import sort_by_date, temporal_split, tuning_cv


@pytest.fixture
def dated() -> pd.DataFrame:
    """200 bookings on consecutive days, deliberately shuffled.

    Shuffled so a split that happened to rely on row order rather than on the
    date column would fail this test.
    """
    dates = pd.date_range("2023-01-01", periods=200, freq="D")
    frame = pd.DataFrame({
        "booking_date": dates,
        "is_canceled": ([0, 1] * 100),
    })
    return frame.sample(frac=1.0, random_state=0)


def test_splits_are_chronological(dated):
    """Every training booking must precede every test booking."""
    split = temporal_split(dated)
    train = dated.loc[split.train_idx, "booking_date"]
    val = dated.loc[split.val_idx, "booking_date"]
    test = dated.loc[split.test_idx, "booking_date"]
    assert train.max() <= val.min()
    assert val.max() <= test.min()


def test_splits_are_disjoint_and_complete(dated):
    split = temporal_split(dated)
    sizes = len(split.train_idx) + len(split.val_idx) + len(split.test_idx)
    assert sizes == len(dated)
    assert not set(split.train_idx) & set(split.val_idx)
    assert not set(split.val_idx) & set(split.test_idx)


def test_fractions_are_respected(dated):
    split = temporal_split(dated, train_frac=0.5, val_frac=0.25)
    assert len(split.train_idx) == pytest.approx(100, abs=2)
    assert len(split.val_idx) == pytest.approx(50, abs=2)


def test_rows_without_a_date_are_excluded(dated):
    """A booking that cannot be placed on the timeline must not silently
    default into training."""
    with_nat = pd.concat([dated, pd.DataFrame({"booking_date": [pd.NaT],
                                               "is_canceled": [1]})])
    split = temporal_split(with_nat)
    total = len(split.train_idx) + len(split.val_idx) + len(split.test_idx)
    assert total == len(dated)


def test_invalid_fractions_raise(dated):
    with pytest.raises(ValueError):
        temporal_split(dated, train_frac=0.9, val_frac=0.2)


def test_summary_reports_base_rates(dated):
    split = temporal_split(dated)
    summary = split.summary(dated, "is_canceled", "booking_date")
    assert list(summary["split"]) == ["train", "validation", "test"]
    assert summary["bookings"].sum() == len(dated)


def test_tuning_cv_is_time_ordered():
    """TimeSeriesSplit, not KFold: tuning must not train on the future either."""
    cv = tuning_cv(3)
    frame = sort_by_date(pd.DataFrame({"booking_date": pd.date_range("2023-01-01", periods=30)}))
    for train_idx, test_idx in cv.split(frame):
        assert max(train_idx) < min(test_idx)
