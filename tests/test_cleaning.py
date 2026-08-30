"""Tests for the cleaning contract.

This is where a silent error is most expensive: every downstream number — the
base rate, the loss attribution, every metric — is computed on the output of
`clean_bookings`. A normalisation rule that quietly stops firing would not
raise anywhere, it would just make the results subtly wrong.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.cleaning import (MISSING_TOKENS, binarise, clean_bookings,
                          cleaning_contract, filter_modelling_rows,
                          parse_status_date)


def test_duplicate_booking_ids_are_dropped(raw_bookings):
    """SG1 appears twice; only the first occurrence should survive."""
    cleaned = clean_bookings(raw_bookings)
    assert len(cleaned) == 5
    assert cleaned["booking_id"].is_unique


def test_index_column_removed(raw_bookings):
    """`Unnamed: 0` is an exported pandas index with no analytical meaning."""
    assert "Unnamed: 0" not in clean_bookings(raw_bookings).columns


@pytest.mark.parametrize("spelling,expected", [
    ("0", 0), ("1", 1), ("No", 0), ("Yes", 1), ("false", 0), ("true", 1),
    ("  TRUE  ", 1), ("nonsense", None),
])
def test_all_is_canceled_spellings_binarise(spelling, expected):
    """The source uses six spellings across three encodings (section 2.1)."""
    result = binarise(pd.Series([spelling])).iloc[0]
    assert (result is pd.NA) if expected is None else (result == expected)


def test_deposit_type_variants_collapse(raw_bookings):
    """26 raw spellings collapse to 3 documented levels (section 8.1).

    The fixture carries 'No Depost' (typo), 'no-deposit' (case and separator)
    and 'NON REFUND', which must land on three distinct canonical values.
    """
    cleaned = clean_bookings(raw_bookings)
    levels = set(cleaned["deposit_type"].dropna())
    assert levels <= {"NO DEPOSIT", "NON REFUND", "REFUNDABLE"}
    assert cleaned.loc[cleaned["booking_id"] == "SG2", "deposit_type"].iloc[0] == "NO DEPOSIT"
    assert cleaned.loc[cleaned["booking_id"] == "SG4", "deposit_type"].iloc[0] == "NO DEPOSIT"


def test_market_segment_typos_are_repaired(raw_bookings):
    """'Onlin TA' and 'Offline TA/T0' (digit zero) are genuine source typos."""
    cleaned = clean_bookings(raw_bookings)
    segments = set(cleaned["market_segment"].dropna())
    assert "ONLIN TA" not in segments and "OFFLINE TA/T0" not in segments
    assert {"ONLINE TA", "OFFLINE TA/TO"} <= segments


def test_customer_type_typo_is_repaired(raw_bookings):
    cleaned = clean_bookings(raw_bookings)
    types = set(cleaned["customer_type"].dropna())
    assert "TRANSIANT" not in types
    assert "TRANSIENT-PARTY" in types  # 'Transient Party' unified with the hyphenated form


def test_country_names_and_iso_codes_unify(raw_bookings):
    """'Singapore' and 'SGP' are the same country and must not split."""
    cleaned = clean_bookings(raw_bookings)
    assert set(cleaned["country"].dropna()) <= {"SGP", "CHN"}


def test_undefined_is_preserved_but_null_is_not(raw_bookings):
    """`UNDEFINED` is a documented level of `meal`; `NULL` means missing.

    Collapsing them together would destroy signal, which is why UNDEFINED is
    deliberately absent from MISSING_TOKENS.
    """
    assert "UNDEFINED" not in MISSING_TOKENS
    cleaned = clean_bookings(raw_bookings)
    meals = cleaned.set_index("booking_id")["meal"]
    assert meals["SG3"] == "UNDEFINED"
    assert pd.isna(meals["SG4"])


@pytest.mark.parametrize("value,expected", [
    ("2024-07-06 00:00:00", "2024-07-06"),   # ISO
    ("03-13-24", "2024-03-13"),              # dash + 2-digit year -> month first
    ("27/11/2023", "2023-11-27"),            # slash + 4-digit year -> day first
    ("08 Mar 2024", "2024-03-08"),           # day month-name year
])
def test_all_four_status_date_formats_parse(value, expected):
    """931 rows use three non-ISO formats; naive parsing drops them silently.

    Each family's day/month convention was established by counting the rows
    where one reading is arithmetically impossible (section 1.8).
    """
    parsed = parse_status_date(pd.Series([value])).iloc[0]
    assert parsed == pd.Timestamp(expected)


def test_month_names_and_abbreviations_both_parse(raw_bookings):
    """The source mixes 'July' and 'Jul' — 23 variants across 12 months."""
    cleaned = clean_bookings(raw_bookings).set_index("booking_id")
    assert cleaned.loc["SG1", "arrival_date"] == pd.Timestamp("2024-07-03")
    assert cleaned.loc["SG2", "arrival_date"] == pd.Timestamp("2024-07-04")


def test_implausible_rows_are_flagged_not_dropped(raw_bookings):
    """Cleaning repairs and flags; only `filter_modelling_rows` removes rows.

    Keeping the two steps separate is what makes the cost of each exclusion
    measurable instead of hidden inside preprocessing.
    """
    cleaned = clean_bookings(raw_bookings)
    assert len(cleaned) == 5
    flags = cleaned.set_index("booking_id")
    assert bool(flags.loc["SG3", "flag_bad_adr"])          # adr == 0
    assert bool(flags.loc["SG4", "flag_bad_lead_time"])    # lead_time == -30
    assert bool(flags.loc["SG5", "flag_bad_lead_time"])    # lead_time == 4989
    assert bool(flags.loc["SG5", "flag_zero_nights"])      # 0 weekend + 0 week


def test_filter_reports_a_reason_for_every_excluded_row(raw_bookings):
    cleaned = clean_bookings(raw_bookings)
    kept, reasons = filter_modelling_rows(cleaned)
    assert len(kept) + len(reasons) == len(cleaned)
    assert reasons.any(axis=1).all()  # no row excluded without a stated reason


def test_target_agrees_with_reservation_status(raw_bookings):
    """Two independently stored encodings of the same event must agree."""
    cleaned = clean_bookings(raw_bookings)
    check_out = cleaned["reservation_status"] == "CHECK-OUT"
    assert (cleaned.loc[check_out, "is_canceled"] == 0).all()
    assert (cleaned.loc[~check_out, "is_canceled"] == 1).all()


def test_contract_is_documented():
    """The rendered contract must stay in step with the code that applies it."""
    contract = cleaning_contract()
    assert not contract.empty
    assert {"step", "columns", "issue", "action"} <= set(contract.columns)
