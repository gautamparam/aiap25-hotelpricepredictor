"""Temporal and capacity reconstruction.

`hotel.db` records bookings, not occupancy, and holds no room-inventory data
at all. This module rebuilds the night-level view the analysis needs:

  * `explode_room_nights` — one row per occupied room-night
  * `infer_capacity`      — hotel capacity, read off the occupancy plateau
  * `occupancy_panel`     — the hotel x date panel used for fill rates
  * `booking_curves`      — when each night sold out, and demand arriving after

The room-night explosion is the expensive step, so callers should build the
panel once and reuse it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Stays longer than this are treated as data errors rather than real stays:
# the 99.9th percentile is 25 nights, and the handful above 30 (max 69) would
# each smear one booking across months of the occupancy curve.
MAX_PLAUSIBLE_NIGHTS = 30


def stayable(df: pd.DataFrame, honoured_only: bool = True) -> pd.DataFrame:
    """Rows that can be placed on a calendar: real dates, 1..30 nights."""
    mask = (
        df["arrival_date"].notna()
        & df["total_nights"].between(1, MAX_PLAUSIBLE_NIGHTS)
    )
    if honoured_only:
        mask &= df["is_canceled"] == 0
    return df.loc[mask]


def explode_room_nights(
    df: pd.DataFrame, columns: list[str] | None = None, honoured_only: bool = True
) -> pd.DataFrame:
    """Expand each stay into one row per occupied night.

    Vectorised via `repeat` + `cumcount` rather than a Python loop: the naive
    loop over ~75k stays is the slowest step in the whole analysis.
    """
    base = stayable(df, honoured_only=honoured_only)
    keep = ["hotel", "arrival_date", "total_nights"] + (columns or [])
    keep = list(dict.fromkeys(c for c in keep if c in base.columns))
    nights = base["total_nights"].astype(int)
    out = base[keep].loc[base.index.repeat(nights)].copy()
    offset = out.groupby(level=0).cumcount()
    out["date"] = out["arrival_date"] + pd.to_timedelta(offset, unit="D")
    return out.reset_index(drop=True)


def infer_capacity(
    room_nights: pd.DataFrame, min_recurrence: int = 5, round_to: int = 5
) -> dict[str, int]:
    """Infer each hotel's room count from the plateau in daily occupancy.

    Daily occupancy piles up just below a hard ceiling and stops, which is the
    signature of a capacity constraint. The ceiling is taken as the highest
    occupancy level that recurs on at least `min_recurrence` nights: a real
    capacity is hit repeatedly, whereas values above it occur once or twice
    each. Those one-off days are the March 2024 event, when both hotels
    demonstrably overbooked, so a naive max or high quantile would mistake an
    overbooking spike for the room count.

    This is an *estimate*. It cannot be distinguished from a cap applied by
    whatever generated the data, and the intended corroboration using
    `days_in_waiting_list` failed. See notebook section 5.
    """
    daily = room_nights.groupby(["hotel", "date"]).size()
    caps: dict[str, int] = {}
    for hotel, series in daily.groupby(level="hotel"):
        counts = series.value_counts()
        recurring = counts[counts >= min_recurrence]
        if recurring.empty:            # degenerate fallback
            ceiling = float(series.max())
        else:
            ceiling = float(recurring.index.max())
        caps[hotel] = int(round_to * round(ceiling / round_to))
    return caps


def occupancy_panel(
    df: pd.DataFrame,
    capacities: dict[str, int] | None = None,
    include_demand: bool = True,
) -> pd.DataFrame:
    """Build the hotel x date panel: occupancy, fill, revenue, ADR, demand.

    `occupied` counts honoured bookings only. `requests` counts every booking
    made for that night including those later cancelled, so the two together
    separate realised occupancy from gross demand.
    """
    sold = explode_room_nights(df, columns=["adr"], honoured_only=True)
    if capacities is None:
        capacities = infer_capacity(sold)

    panel = (
        sold.groupby(["hotel", "date"])
        .agg(occupied=("adr", "size"), revenue=("adr", "sum"))
        .reset_index()
    )
    if include_demand:
        allrn = explode_room_nights(df, columns=["adr"], honoured_only=False)
        demand = (
            allrn.groupby(["hotel", "date"]).size().rename("requests").reset_index()
        )
        panel = panel.merge(demand, on=["hotel", "date"], how="left")

    panel["capacity"] = panel["hotel"].map(capacities)
    panel["fill_rate"] = panel["occupied"] / panel["capacity"]
    panel["empty_rooms"] = (panel["capacity"] - panel["occupied"]).clip(lower=0)
    panel["adr_night"] = panel["revenue"] / panel["occupied"]
    panel["revpar"] = panel["revenue"] / panel["capacity"]
    panel["month"] = panel["date"].dt.month
    panel["dow"] = panel["date"].dt.dayofweek
    panel["is_weekend"] = panel["dow"].isin([4, 5])
    return panel.sort_values(["hotel", "date"]).reset_index(drop=True)


def trim_to_stable_window(
    panel: pd.DataFrame, start: str = "2023-08-01", end: str = "2025-09-13"
) -> pd.DataFrame:
    """Drop the ramp-up/ramp-down edges where coverage is partial.

    Outside this window occupancy is an artefact of the extract's boundaries,
    not of demand, and would drag every mean fill rate downwards.
    """
    return panel[panel["date"].between(start, end)].copy()


def booking_curves(df: pd.DataFrame, capacities: dict[str, int]) -> pd.DataFrame:
    """For each night: when it hit capacity, and how much demand came after.

    Demand arriving *after* a night is already full is the cleanest available
    evidence of excess demand, because unlike a raw booking count it is not
    inflated by the 36% cancellation rate.
    """
    allrn = explode_room_nights(
        df, columns=["adr", "is_canceled", "booking_date"], honoured_only=False
    )
    allrn = allrn[allrn["booking_date"].notna()].copy()
    allrn["days_out"] = (allrn["date"] - allrn["booking_date"]).dt.days
    allrn = allrn[allrn["days_out"].between(0, 500)]

    rows = []
    for (hotel, date), g in allrn.groupby(["hotel", "date"], sort=False):
        cap = capacities[hotel]
        if len(g) < cap:
            continue
        g = g.sort_values("days_out", ascending=False)
        sellout = g["days_out"].to_numpy()[cap - 1]
        after = g[g["days_out"] < sellout]
        rows.append(
            {
                "hotel": hotel,
                "date": date,
                "gross_requests": len(g),
                "sellout_days_out": sellout,
                "excess_all": len(after),
                "excess_honoured": int((after["is_canceled"] == 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def cancellation_timing(df: pd.DataFrame) -> pd.DataFrame:
    """Days between cancellation and arrival, for cancelled bookings only."""
    c = df[(df["is_canceled"] == 1) & df["status_days_before_arrival"].notna()]
    return c[
        ["booking_id", "hotel", "reservation_status", "arrival_date",
         "status_date", "status_days_before_arrival", "adr", "total_nights",
         "booking_value"]
    ].copy()


# ---------------------------------------------------------------------------
# Recoverability: was a cancelled room-night actually lost, or was it re-sold?
# ---------------------------------------------------------------------------

def lost_room_nights(
    df: pd.DataFrame, panel: pd.DataFrame, extra_columns: list[str] | None = None
) -> pd.DataFrame:
    """Attribute cancelled room-nights to 'lost' vs 're-sold'.

    **Definition.** For a hotel-night with capacity K, honoured occupancy O and
    C cancelled room-nights, occupancy in the absence of those cancellations
    would have been ``min(O + C, K)`` — the hotel sells to capacity if demand
    allows. The room-nights genuinely lost are therefore

        lost = min(O + C, K) - O = min(C, K - O) = min(C, E)

    where ``E`` is the night's empty rooms. This needs no threshold, and every
    cancelled room-night is either lost or re-sold, so the two shares sum to 1.

    **What it cannot do.** The counterfactual does not identify *which* of the
    C cancellations were the unfilled ones — within a night they are
    exchangeable. Two columns are returned to bracket that:

    * ``p_lost``      — the exchangeable share ``min(C, E) / C``, applied
      equally to every cancellation on the night. Assumption-free.
    * ``lost_latest_first`` — a 0/1 flag marking the ``min(C, E)`` cancellations
      closest to arrival, on the reasoning that the hotel back-fills over time.
      An upper bound on the loss attributed to short-notice cancellations.

    Returns one row per cancelled room-night.
    """
    cols = ["adr", "status_days_before_arrival"] + (extra_columns or [])
    cancelled = df[df["is_canceled"] == 1]
    rn = explode_room_nights(cancelled, columns=cols, honoured_only=False)
    rn = rn.merge(
        panel[["hotel", "date", "occupied", "capacity", "empty_rooms", "month"]],
        on=["hotel", "date"], how="inner",
    )

    per_night = (
        rn.groupby(["hotel", "date"])
        .agg(C=("adr", "size"), E=("empty_rooms", "first"))
        .reset_index()
    )
    per_night["lost"] = np.minimum(per_night["C"], per_night["E"])
    per_night["p_lost"] = per_night["lost"] / per_night["C"]

    rn = rn.merge(per_night[["hotel", "date", "C", "E", "lost", "p_lost"]],
                  on=["hotel", "date"])
    rn = rn.sort_values(["hotel", "date", "status_days_before_arrival"])
    rank = rn.groupby(["hotel", "date"]).cumcount()
    rn["lost_latest_first"] = (rank < rn["lost"]).astype(int)
    return rn.reset_index(drop=True)


def loss_summary(lost_rn: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Summarise lost room-nights and value, under both attribution rules."""
    def _agg(g: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "cancelled_room_nights": len(g),
            "share_of_cancellations": len(g) / len(lost_rn),
            "pct_lost_exchangeable": g["p_lost"].mean(),
            "pct_lost_latest_first": g["lost_latest_first"].mean(),
            "lost_value_exchangeable": (g["p_lost"] * g["adr"]).sum(),
            "lost_value_latest_first": (g["lost_latest_first"] * g["adr"]).sum(),
        })
    if by is None:
        return _agg(lost_rn).to_frame("all").T
    return lost_rn.groupby(by, observed=True).apply(_agg, include_groups=False)
