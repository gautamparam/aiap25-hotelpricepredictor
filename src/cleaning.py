"""The cleaning contract.

Every rule here is justified by a finding in `eda.ipynb` section 1. The rules
are declared in `CLEANING_RULES` so the notebook can render them as a table
and the pipeline can apply them, guaranteeing the documented contract and the
executed contract are the same object.

Design note: cleaning *repairs* values and *flags* implausible ones, but does
not drop rows. Row exclusion is a modelling decision, kept separate in
`filter_modelling_rows` so its cost is visible and configurable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Tokens that mean "no value". `UNDEFINED` is deliberately absent: the data
# dictionary lists "Undefined" as a legitimate level of `meal` (no package)
# and `market_segment`, so collapsing it into missing would destroy signal.
MISSING_TOKENS = {"", "-", "NA", "N/A", "NAN", "NULL", "NONE", "UNKNOWN"}

# Beyond this, a "reservation" is not a real booking horizon for a city hotel.
MAX_PLAUSIBLE_LEAD_DAYS = 730

TRUTHY = {"1", "YES", "TRUE", "Y", "T"}
FALSY = {"0", "NO", "FALSE", "N", "F"}

MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# Genuine typos and aliases found by enumerating every distinct value.
CATEGORICAL_ALIASES: dict[str, dict[str, str]] = {
    "market_segment": {
        "OFFLINE TA/T0": "OFFLINE TA/TO",   # digit zero for letter O
        "ONLIN TA": "ONLINE TA",
    },
    "customer_type": {
        "TRANSIANT": "TRANSIENT",
        "TRANSIENT PARTY": "TRANSIENT-PARTY",
    },
    "deposit_type": {
        "NO DEPOST": "NO DEPOSIT",
    },
    "reservation_status": {
        "CHECK OUT": "CHECK-OUT",
        "NO SHOW": "NO-SHOW",
    },
    "country": {
        "SINGAPORE": "SGP", "MALAYSIA": "MYS", "INDONESIA": "IDN",
        "CHINA": "CHN", "INDIA": "IND", "AUSTRALIA": "AUS",
        "UNITED STATES": "USA", "UNITED KINGDOM": "GBR", "JAPAN": "JPN",
        "PHILIPPINES": "PHL", "SOUTH KOREA": "KOR", "THAILAND": "THA",
        "VIETNAM": "VNM", "HONG KONG": "HKG", "TAIWAN": "TWN",
        "NEW ZEALAND": "NZL", "MYANMAR": "MMR", "OTHER": "OTH",
    },
}

CATEGORICAL_COLUMNS = [
    "hotel", "meal", "country", "market_segment", "distribution_channel",
    "reserved_room_type", "assigned_room_type", "deposit_type",
    "customer_type", "reservation_status", "promo_code", "agent", "company",
]

INTEGER_COLUMNS = [
    "lead_time", "arrival_date_year", "arrival_date_week_number",
    "arrival_date_day_of_month", "stays_in_weekend_nights",
    "stays_in_week_nights", "adults", "children", "babies",
    "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "booking_changes",
    "days_in_waiting_list", "required_car_parking_spaces",
    "total_of_special_requests",
]

FLOAT_COLUMNS = ["adr"]

# Declarative contract, rendered as a table in the notebook and README.
CLEANING_RULES: list[dict[str, str]] = [
    {"step": "drop_index_column", "columns": "Unnamed: 0",
     "issue": "Exported pandas index, no analytical meaning",
     "action": "Drop the column"},
    {"step": "drop_duplicate_bookings", "columns": "all",
     "issue": "353 rows duplicate an existing booking_id exactly",
     "action": "Drop exact duplicates, keep first occurrence"},
    {"step": "normalise_text", "columns": "all text columns",
     "issue": "Leading/trailing/internal whitespace and mixed casing inflate "
              "cardinality (deposit_type 26 variants, market_segment 66)",
     "action": "Strip, collapse internal whitespace, uppercase"},
    {"step": "unify_missing_tokens", "columns": "all text columns",
     "issue": "Missing values encoded as '', '-', 'NA', 'N/A', 'NULL', "
              "'UNKNOWN' interchangeably",
     "action": "Map all such tokens to NaN. 'UNDEFINED' is preserved: the "
               "data dictionary lists it as a real level"},
    {"step": "apply_aliases", "columns": "market_segment, customer_type, "
                                         "deposit_type, reservation_status, country",
     "issue": "Typos ('No Depost', 'Onlin TA', 'Transiant', 'Offline TA/T0') "
              "and country names given as both ISO codes and full names",
     "action": "Map each variant onto a single canonical value"},
    {"step": "binarise_target", "columns": "is_canceled",
     "issue": "Six spellings across three encodings (0/1, No/Yes, false/true)",
     "action": "Map to integer 0/1"},
    {"step": "coerce_numeric", "columns": "17 numeric columns",
     "issue": "Every column stored as TEXT in SQLite",
     "action": "Coerce to Int64/float; unparseable values become NaN"},
    {"step": "parse_month", "columns": "arrival_date_month",
     "issue": "Month given as both full name and three-letter abbreviation",
     "action": "Map to month number 1-12"},
    {"step": "build_dates", "columns": "arrival_date, booking_date",
     "issue": "No usable date column exists; arrival is split across three "
              "fields and booking date is only implied by lead_time",
     "action": "Construct arrival_date from Y/M/D; booking_date = "
               "arrival_date - lead_time"},
    {"step": "parse_status_date", "columns": "reservation_status_date",
     "issue": "931 rows use one of three non-ISO formats (MM-DD-YY, "
              "DD/MM/YYYY, 'DD Mon YYYY'); naive parsing silently drops them",
     "action": "Parse per format family. Each family's convention is "
               "determined by rows where one reading is impossible, so no row "
               "is left ambiguous"},
    {"step": "derive_stay_fields", "columns": "total_nights, total_guests",
     "issue": "Stay length and party size are split across several columns",
     "action": "Sum weekend+week nights; sum adults+children+babies"},
    {"step": "flag_implausible", "columns": "adr, total_nights, total_guests",
     "issue": "ADR up to $7,108 and <= $0 in 1,905 rows; 621 zero-night "
              "bookings; 190 zero-guest bookings; adults up to 55",
     "action": "Add boolean quality flags. Rows are NOT dropped here"},
    {"step": "flag_bad_lead_time", "columns": "lead_time",
     "issue": "298 bookings have a negative lead time (booked after arrival) "
              "and 98 exceed two years, up to 4,989 days; these push derived "
              "booking dates back as far as 2010",
     "action": "Flag lead_time < 0 or > 730 days as implausible"},
]


# `reservation_status_date` is written in four different formats. Each family
# has a single self-consistent convention, established by counting the rows
# where one reading is impossible (a component > 12 forces day- or
# month-first). Dash + 2-digit year is month-first: 187 rows force it, 0
# contradict. Slash + 4-digit year is day-first: 168 force it, 0 contradict.
# So no row is genuinely ambiguous once the family is identified.
STATUS_DATE_FORMATS = (
    ("%Y-%m-%d", r"^\d{4}-\d{2}-\d{2}$"),
    ("%m-%d-%y", r"^\d{2}-\d{2}-\d{2}$"),
    ("%d/%m/%Y", r"^\d{2}/\d{2}/\d{4}$"),
    ("%d %b %Y", r"^\d{2} [A-Za-z]{3} \d{4}$"),
)


def parse_status_date(s: pd.Series) -> pd.Series:
    """Parse `reservation_status_date` across all four observed formats."""
    text = s.astype("string").str.strip().str.slice(0, 11).str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for fmt, pattern in STATUS_DATE_FORMATS:
        todo = out.isna() & text.str.match(pattern, na=False)
        if todo.any():
            out.loc[todo] = pd.to_datetime(text[todo], format=fmt, errors="coerce")
    return out


def _norm_series(s: pd.Series) -> pd.Series:
    """Strip, collapse internal whitespace, uppercase; missing tokens to NaN."""
    out = (
        s.astype("string")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.upper()
    )
    return out.mask(out.isin(MISSING_TOKENS))


def normalise_text(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Apply whitespace/case normalisation and alias mapping to text columns."""
    df = df.copy()
    columns = columns or [c for c in CATEGORICAL_COLUMNS if c in df.columns]
    for col in columns:
        norm = _norm_series(df[col])
        # Separators are unified only for alias lookup, so room-type codes and
        # values like "TA/TO" keep their punctuation.
        if col in CATEGORICAL_ALIASES:
            lookup = norm.str.replace(r"[-_]", " ", regex=True).str.replace(
                r"\s+", " ", regex=True
            )
            norm = lookup.replace(CATEGORICAL_ALIASES[col]).fillna(norm)
        df[col] = norm
    return df


def binarise(s: pd.Series) -> pd.Series:
    """Map mixed boolean spellings onto nullable integer 0/1."""
    up = s.astype("string").str.strip().str.upper()
    return (
        up.map(lambda v: 1 if v in TRUTHY else (0 if v in FALSY else pd.NA))
        .astype("Int64")
    )


def clean_bookings(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full cleaning contract to the raw bookings table."""
    df = df.drop(columns=[c for c in ["Unnamed: 0"] if c in df.columns])
    df = df.drop_duplicates(subset="booking_id", keep="first").copy()

    df = normalise_text(df)

    df["is_canceled"] = binarise(df["is_canceled"])
    for col in (c for c in INTEGER_COLUMNS if c in df.columns):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in (c for c in FLOAT_COLUMNS if c in df.columns):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["arrival_month_num"] = (
        df["arrival_date_month"].astype("string").str.strip().str.upper().str[:3]
        .map(MONTHS).astype("Int64")
    )
    df["arrival_date"] = pd.to_datetime(
        {
            "year": df["arrival_date_year"],
            "month": df["arrival_month_num"],
            "day": df["arrival_date_day_of_month"],
        },
        errors="coerce",
    )
    df["booking_date"] = df["arrival_date"] - pd.to_timedelta(
        df["lead_time"].astype("float"), unit="D"
    )
    df["status_date"] = parse_status_date(df["reservation_status_date"])
    # Positive = status set before arrival (a cancellation with notice);
    # negative = set after arrival (a check-out, i.e. the departure date).
    df["status_days_before_arrival"] = (
        df["arrival_date"] - df["status_date"]
    ).dt.days

    df["total_nights"] = (
        df["stays_in_weekend_nights"].fillna(0) + df["stays_in_week_nights"].fillna(0)
    ).astype("Int64")
    df["total_guests"] = (
        df["adults"].fillna(0) + df["children"].fillna(0) + df["babies"].fillna(0)
    ).astype("Int64")
    df["booking_value"] = df["adr"] * df["total_nights"].astype("float")

    df["flag_bad_adr"] = ~(df["adr"] > 0)
    df["flag_zero_nights"] = ~(df["total_nights"] > 0)
    df["flag_zero_guests"] = ~(df["total_guests"] > 0)
    df["flag_bad_date"] = df["arrival_date"].isna()
    # A negative lead time means the booking was made after arrival; anything
    # beyond two years is not a real reservation horizon for a city hotel.
    lead = df["lead_time"].astype("float")
    df["flag_bad_lead_time"] = (lead < 0) | (lead > MAX_PLAUSIBLE_LEAD_DAYS)
    df["flag_any"] = df[
        ["flag_bad_adr", "flag_zero_nights", "flag_zero_guests", "flag_bad_date",
         "flag_bad_lead_time"]
    ].any(axis=1)
    return df.reset_index(drop=True)


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the reviews table; flag ratings outside the documented 1-5 range."""
    df = df.copy()
    df["client_email"] = df["client_email"].astype("string").str.strip().str.lower()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").astype("Int64")
    df["comments"] = df["comments"].astype("string").str.strip()
    df["flag_rating_out_of_range"] = ~df["rating"].between(1, 5)
    return df.reset_index(drop=True)


def filter_modelling_rows(
    df: pd.DataFrame, require_valid_date: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into rows usable for modelling and rows excluded, with reasons.

    Returned separately so the notebook can quantify exactly what each
    exclusion costs rather than silently shrinking the dataset.
    """
    reasons = pd.DataFrame(index=df.index)
    reasons["non_positive_adr"] = df["flag_bad_adr"]
    reasons["zero_nights"] = df["flag_zero_nights"]
    reasons["zero_guests"] = df["flag_zero_guests"]
    reasons["implausible_lead_time"] = df["flag_bad_lead_time"]
    if require_valid_date:
        reasons["unparseable_date"] = df["flag_bad_date"]
    excluded = reasons.any(axis=1)
    return df.loc[~excluded].reset_index(drop=True), reasons.loc[excluded]


def cleaning_contract() -> pd.DataFrame:
    """The cleaning contract as a table, for the notebook and README."""
    return pd.DataFrame(CLEANING_RULES)
