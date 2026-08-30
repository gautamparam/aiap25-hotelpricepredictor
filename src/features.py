"""Feature engineering for the cancellation-risk model.

This module turns a *cleaned* bookings frame (see :mod:`src.cleaning`) into the
feature matrix the models consume. It knows nothing about models, metrics or
splitting — that separation is what lets the same features feed a logistic
regression and a gradient-boosted tree without either leaking into the other.

Two principles govern everything here, both inherited from the EDA:

**Admissibility.** Only columns whose value is settled at the moment a booking
is created may be used. Six columns are excluded for leakage even though three
of them carry the strongest apparent signal in the dataset — see
``eda.ipynb`` section 8. ``EXCLUDED_COLUMNS`` is enforced, not merely
documented: :func:`assert_no_leakage` fails loudly if one reappears.

**Train-only fitting.** Winsorisation caps and category vocabularies are
learned in ``fit`` and merely applied in ``transform``. Fitting them on the
full dataset would leak the test period's price distribution into training —
which matters here because the March 2024 event (section 6.2) sits at the far
right of the ADR distribution and would drag a global quantile upwards.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# ---------------------------------------------------------------------------
# Column contracts
# ---------------------------------------------------------------------------

#: Columns that must never reach a model. Each is a *consequence* of the
#: outcome rather than a cause, so a model trained on them scores well in
#: validation and fails in production. Established in eda.ipynb section 8.
EXCLUDED_COLUMNS: tuple[str, ...] = (
    "deposit_type",            # 99.4% pure; marks tour-operator allotment releases (8.1)
    "assigned_room_type",      # only assigned once a guest actually arrives (8.4)
    "booking_changes",         # accrues over the booking's life; zero at creation (8.5)
    "reservation_status",      # the target, restated
    "reservation_status_date", # set when the outcome occurs
    "client_email",            # unique per booking, so an identifier not a feature (1.2)
)

#: Numeric columns used as-is. All are known at booking time and already
#: numeric after cleaning, so no transformation earns its complexity.
NUMERIC_PASSTHROUGH: tuple[str, ...] = (
    "lead_time",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "is_repeated_guest",
    "required_car_parking_spaces",
    "total_of_special_requests",
)

#: Categorical columns taken directly from the cleaned frame. Cleaning has
#: already collapsed these (deposit_type 26 -> 3, market_segment 66 -> 8,
#: country 140 -> 18); without that step a one-hot encoder would emit hundreds
#: of near-empty columns. See eda.ipynb section 1.6.
CATEGORICAL_BASE: tuple[str, ...] = (
    "hotel",
    "market_segment",
    "distribution_channel",
    "customer_type",
    "meal",
    "reserved_room_type",
)

#: The March 2024 demand event: arrivals 2-4x baseline, ADR ~10x, cancellation
#: rate roughly halved. Flagged rather than dropped, because it is real demand
#: and a model that has never seen a surge cannot recognise the next one.
#: eda.ipynb section 6.2.
EVENT_WINDOW: tuple[str, str] = ("2024-02-28", "2024-03-12")

#: Lead-time bands. Cancellation climbs monotonically 9.8% -> 56.2% across
#: these (section 3.5). Trees can find such a split unaided; a linear model
#: cannot, so the banded version is supplied for the shared feature set.
LEAD_TIME_BINS: tuple[float, ...] = (-np.inf, 6, 29, 89, 179, np.inf)
LEAD_TIME_LABELS: tuple[str, ...] = ("0-6d", "7-29d", "30-89d", "90-179d", "180d+")

#: Months grouped by how recoverable a lost room-night is. Fill rate runs
#: 44-65% in Nov-Feb against 89-95% in May-Sep (section 5.3), which makes an
#: identical cancellation roughly four times more costly in the trough.
PEAK_MONTHS: frozenset[int] = frozenset({5, 6, 7, 8, 9})
TROUGH_MONTHS: frozenset[int] = frozenset({11, 12, 1, 2})

#: Placeholder for structurally-absent identifiers. `company` is missing on
#: 70.6% of rows, `agent` on 13.7%, `promo_code` on 40.9% — but a booking
#: simply may not involve a company, an agent or a promotion. The absence is
#: informative, so it becomes a level rather than something to impute (1.3).
MISSING_LEVEL = "NONE"


def assert_no_leakage(frame: pd.DataFrame) -> None:
    """Raise if any excluded column has found its way into a feature matrix.

    Cheap insurance against the most damaging mistake this pipeline can make.
    A leaked column would inflate every metric while producing a model that
    cannot be deployed, and the failure is silent without a check like this.

    Raises:
        ValueError: naming the offending columns.
    """
    leaked = sorted(set(frame.columns) & set(EXCLUDED_COLUMNS))
    if leaked:
        raise ValueError(
            f"Leakage guard tripped: {leaked} must not reach the model. "
            "See eda.ipynb section 8 for why each is inadmissible."
        )


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the stateless features — everything needing no fitted parameter.

    Kept separate from :class:`FeatureBuilder` so it can be called directly
    (for example from the notebook) and unit-tested without fitting anything.

    Args:
        df: a cleaned bookings frame from :func:`src.cleaning.clean_bookings`.

    Returns:
        A copy of ``df`` with the derived columns added. Nothing is dropped
        here; column selection happens in :meth:`FeatureBuilder.transform`.
    """
    out = df.copy()

    # `arrival_date` and the component sums are produced by cleaning, but the
    # notebook may hand us a frame that predates them, so derive defensively.
    if "total_nights" not in out:
        out["total_nights"] = (out["stays_in_weekend_nights"].fillna(0)
                               + out["stays_in_week_nights"].fillna(0))
    if "total_guests" not in out:
        out["total_guests"] = (out["adults"].fillna(0) + out["children"].fillna(0)
                               + out["babies"].fillna(0))

    # Exposure: what the booking is actually worth. This is half of the
    # adopted ranking rule, P(cancel) x value (section 9.5), so it is a
    # first-class feature rather than something applied afterwards.
    out["booking_value"] = out["adr"] * out["total_nights"].astype(float)

    arrival = out["arrival_date"]
    out["arrival_month"] = arrival.dt.month.astype("Int64")
    out["arrival_week"] = arrival.dt.isocalendar().week.astype("Int64")
    out["arrival_dow"] = arrival.dt.dayofweek.astype("Int64")
    # Friday and Saturday arrivals begin a weekend stay; Sunday does not.
    out["is_weekend_arrival"] = arrival.dt.dayofweek.isin([4, 5]).astype("Int64")

    month = out["arrival_month"]
    out["season_band"] = np.select(
        [month.isin(list(TROUGH_MONTHS)), month.isin(list(PEAK_MONTHS))],
        ["trough", "peak"],
        default="shoulder",
    )

    start, end = EVENT_WINDOW
    out["is_event_period"] = arrival.between(start, end).astype("Int64")

    out["lead_time_band"] = pd.cut(
        out["lead_time"].astype(float),
        bins=list(LEAD_TIME_BINS),
        labels=list(LEAD_TIME_LABELS),
    ).astype(object)

    # Presence flags carry the signal that the identifier itself is too
    # high-cardinality to express: a corporate-billed booking behaves
    # differently from a direct one regardless of *which* company it is.
    out["has_agent"] = out["agent"].notna().astype("Int64")
    out["has_company"] = out["company"].notna().astype("Int64")

    return out


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Build the model-ready feature matrix, learning only from training data.

    Implemented as a scikit-learn transformer so that train-only fitting is
    enforced by the framework rather than by remembering to do it. Two things
    are learned:

    * the ADR winsorisation cap, and
    * the vocabularies of the high-cardinality categoricals (``country``,
      ``agent``).

    Args:
        winsorise_adr_quantile: quantile at which ADR is clipped. Clipping
            rather than dropping is deliberate — the tail above $2,000 is
            almost entirely the March 2024 event (section 6.2), which is real
            demand. Dropping it would delete the clearest evidence in the
            dataset of what pricing power looks like.
        country_top_n: countries outside the top N by training frequency are
            folded into ``OTH``. Cleaning reduces 140 raw values to 18, but the
            tail is still thin enough to be noise.
        agent_top_n: same treatment for travel-agent identifiers.
        include_waiting_list: whether to keep ``days_in_waiting_list``. It is
            admissible (known at confirmation) but ~98% of waitlisted bookings
            sit in two market segments, so it is largely a proxy for those.
            Exposed as a switch so the ablation is one config change
            (section 8.6).
    """

    def __init__(
        self,
        winsorise_adr_quantile: float = 0.99,
        country_top_n: int = 15,
        agent_top_n: int = 20,
        include_waiting_list: bool = True,
    ) -> None:
        self.winsorise_adr_quantile = winsorise_adr_quantile
        self.country_top_n = country_top_n
        self.agent_top_n = agent_top_n
        self.include_waiting_list = include_waiting_list

    # -- scikit-learn API ---------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureBuilder":
        """Learn the winsorisation cap and category vocabularies from ``X``."""
        engineered = engineer(X)

        self.adr_cap_ = float(engineered["adr"].quantile(self.winsorise_adr_quantile))
        self.country_vocab_ = self._top_levels(engineered["country"], self.country_top_n)
        self.agent_vocab_ = self._top_levels(engineered["agent"], self.agent_top_n)

        # Freeze the level set of every categorical. Without this, casting to
        # `category` would infer levels independently from each frame, so a
        # level absent from the test split would silently shift every category
        # code and the model would read the wrong values.
        derived = self._apply_fitted(engineered)
        self.category_levels_ = {
            col: sorted(set(derived[col].dropna().astype(str)) | {MISSING_LEVEL})
            for col in self.categorical_features()
        }
        self.feature_names_ = self._column_plan()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the learned parameters and select the admissible columns."""
        derived = self._apply_fitted(engineer(X))
        out = derived[self._column_plan()].copy()

        for col in self.categorical_features():
            values = out[col].astype("object").where(out[col].notna(), MISSING_LEVEL).astype(str)
            # A level unseen during training becomes NONE rather than a new
            # category: production must score a booking from a brand-new travel
            # agent without crashing, and without shifting the encoding.
            known = self.category_levels_[col]
            out[col] = pd.Categorical(values.where(values.isin(known), MISSING_LEVEL),
                                      categories=known)
        for col in self.numeric_features():
            out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)

        assert_no_leakage(out)
        return out

    def _apply_fitted(self, engineered: pd.DataFrame) -> pd.DataFrame:
        """Add the columns that depend on parameters learned in ``fit``.

        Shared by ``fit`` and ``transform`` so the two cannot drift apart —
        the levels frozen during fitting are exactly those transform produces.
        """
        out = engineered.copy()

        # Winsorise both the rate and the exposure. `booking_value` is derived
        # from ADR, so clipping only one would leave the outlier alive in the
        # other. Clipping rather than dropping is deliberate: the tail above
        # $2,000 is almost entirely the March 2024 event (section 6.2), which
        # is real demand, not error.
        out["adr_winsorised"] = out["adr"].clip(upper=self.adr_cap_)
        out["booking_value"] = out["adr_winsorised"] * out["total_nights"].astype(float)

        out["country_grouped"] = self._apply_vocab(out["country"], self.country_vocab_,
                                                   other="OTH")
        out["agent_grouped"] = self._apply_vocab(out["agent"], self.agent_vocab_,
                                                 other="OTHER_AGENT")
        out["promo_code"] = out["promo_code"].astype("object").fillna(MISSING_LEVEL)
        return out

    # -- column plan --------------------------------------------------------

    def numeric_features(self) -> list[str]:
        """Numeric columns in the output, in a stable order."""
        cols = list(NUMERIC_PASSTHROUGH) + [
            "adr_winsorised", "booking_value", "total_nights", "total_guests",
            "arrival_month", "arrival_week", "arrival_dow",
            "is_weekend_arrival", "is_event_period", "has_agent", "has_company",
        ]
        if self.include_waiting_list:
            cols.append("days_in_waiting_list")
        return cols

    def categorical_features(self) -> list[str]:
        """Categorical columns in the output, in a stable order."""
        return list(CATEGORICAL_BASE) + [
            "country_grouped", "agent_grouped", "promo_code",
            "lead_time_band", "season_band",
        ]

    def _column_plan(self) -> list[str]:
        return self.numeric_features() + self.categorical_features()

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _top_levels(series: pd.Series, n: int) -> set[str]:
        """The n most frequent non-missing levels, as a set for O(1) lookup."""
        return set(series.dropna().value_counts().head(n).index.astype(str))

    @staticmethod
    def _apply_vocab(series: pd.Series, vocab: set[str], other: str) -> pd.Series:
        """Map a series onto a fixed vocabulary, folding the rest into ``other``.

        Unseen levels at transform time land in ``other`` rather than raising,
        which is the behaviour production needs: a new travel agent should not
        crash scoring.
        """
        text = series.astype("object").where(series.notna(), MISSING_LEVEL).astype(str)
        return text.where(text.isin(vocab | {MISSING_LEVEL}), other)
