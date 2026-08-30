"""Tests for feature engineering.

The most important test in this file is `test_excluded_columns_never_reach_the_model`.
Leakage is the failure mode that does not announce itself: the model scores
brilliantly, ships, and then performs at chance in production. A unit test is
the cheapest possible guard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.cleaning import clean_bookings
from src.features import (EXCLUDED_COLUMNS, FeatureBuilder, assert_no_leakage,
                          engineer)


@pytest.fixture
def cleaned(raw_bookings):
    return clean_bookings(raw_bookings)


def test_excluded_columns_never_reach_the_model(cleaned):
    """The leakage guard: six columns are inadmissible (section 8)."""
    builder = FeatureBuilder().fit(cleaned)
    features = builder.transform(cleaned)
    assert not set(features.columns) & set(EXCLUDED_COLUMNS)


def test_leakage_guard_raises_on_a_planted_column(cleaned):
    """The guard must actually fire, not merely exist."""
    frame = pd.DataFrame({"deposit_type": ["NO DEPOSIT"], "lead_time": [10]})
    with pytest.raises(ValueError, match="Leakage guard tripped"):
        assert_no_leakage(frame)


def test_derived_columns_are_correct(cleaned):
    """total_nights, total_guests and booking_value are simple sums/products."""
    out = engineer(cleaned).set_index("booking_id")
    assert out.loc["SG1", "total_nights"] == 3        # 1 weekend + 2 week
    assert out.loc["SG1", "total_guests"] == 2        # 2 adults
    assert out.loc["SG1", "booking_value"] == pytest.approx(150.0 * 3)


def test_season_and_event_flags(cleaned):
    """July is peak, March is shoulder, December is trough (section 5.3)."""
    out = engineer(cleaned).set_index("booking_id")
    assert out.loc["SG1", "season_band"] == "peak"       # July
    assert out.loc["SG3", "season_band"] == "shoulder"   # March
    assert out.loc["SG5", "season_band"] == "trough"     # December
    # SG3 (5 Mar 2024) and SG4 (6 Mar 2024) fall inside the 28 Feb - 12 Mar
    # 2024 demand event; SG1 and SG2 (July) and SG5 (December) do not.
    assert out.loc["SG3", "is_event_period"] == 1
    assert out.loc["SG4", "is_event_period"] == 1
    assert out.loc["SG1", "is_event_period"] == 0
    assert out["is_event_period"].sum() == 2


def test_winsorisation_uses_train_only(cleaned):
    """The cap must come from the fitting frame, not from the frame transformed.

    Fitting it globally would leak the test period's price distribution into
    training — which matters here because the March 2024 event sits at the far
    right of the ADR distribution (section 6.2).
    """
    train = cleaned[cleaned["adr"] < 1000]
    builder = FeatureBuilder(winsorise_adr_quantile=1.0).fit(train)
    cap = builder.adr_cap_
    assert cap == pytest.approx(train["adr"].max())

    transformed = builder.transform(cleaned)
    assert transformed["adr_winsorised"].max() <= cap + 1e-9


def test_booking_value_uses_the_winsorised_rate(cleaned):
    """Clipping only the rate would leave the outlier alive in the exposure."""
    builder = FeatureBuilder(winsorise_adr_quantile=0.5).fit(cleaned)
    out = builder.transform(cleaned)
    implied = out["adr_winsorised"] * out["total_nights"]
    assert np.allclose(out["booking_value"], implied, equal_nan=True)


def test_unseen_categories_do_not_raise(cleaned):
    """Production must score a booking from a brand-new agent without crashing."""
    builder = FeatureBuilder().fit(cleaned)
    novel = cleaned.copy()
    novel["country"] = "ZZZ"
    novel["agent"] = "99999.0"
    out = builder.transform(novel)
    assert out["country_grouped"].isin(["OTH", "NONE"]).all()
    assert out["agent_grouped"].isin(["OTHER_AGENT", "NONE"]).all()


def test_category_levels_are_frozen_at_fit_time(cleaned):
    """Two frames must encode to identical category sets.

    Without freezing, `astype("category")` infers levels per frame, so a level
    missing from one split would shift every category code silently.
    """
    builder = FeatureBuilder().fit(cleaned)
    a = builder.transform(cleaned.head(3))
    b = builder.transform(cleaned.tail(2))
    for col in builder.categorical_features():
        assert list(a[col].cat.categories) == list(b[col].cat.categories)


def test_missing_identifiers_become_a_level_not_a_nan(cleaned):
    """Absence of an agent or company is informative, not a defect (section 1.3)."""
    builder = FeatureBuilder().fit(cleaned)
    out = builder.transform(cleaned)
    assert out[builder.categorical_features()].isna().sum().sum() == 0
    assert "NONE" in set(out["agent_grouped"].astype(str))


def test_waiting_list_can_be_ablated(cleaned):
    """The switch exists so section 8.6's confound can be tested in one flag."""
    with_wl = FeatureBuilder(include_waiting_list=True).fit(cleaned).transform(cleaned)
    without = FeatureBuilder(include_waiting_list=False).fit(cleaned).transform(cleaned)
    assert "days_in_waiting_list" in with_wl.columns
    assert "days_in_waiting_list" not in without.columns
