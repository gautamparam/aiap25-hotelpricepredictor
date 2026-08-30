"""Time-ordered train / validation / test splitting.

The single most consequential choice in this module is that splits are made on
``booking_date`` and never at random. The reasoning is in ``eda.ipynb``
section 12, and it is worth restating because a random split is the default
almost everywhere else:

A booking is scored *when it is created*, so the deployment reality is "train
on bookings taken up to today, score bookings taken tomorrow". A random split
breaks that in two ways. It lets the model learn a specific night's demand
conditions from some of that night's bookings and then be tested on the rest,
and it hides concept drift by mixing periods together. The EDA measured that
drift directly: the cancellation rate ran 37.5% / 40.6% / 29.9% across the
three time-ordered periods. A model has to survive that, not be protected
from it.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


@dataclass(frozen=True)
class Split:
    """Row indices for one temporal split, plus the boundaries that made it.

    Boundaries are carried alongside the indices so they can be written into
    ``metrics.json`` — a reader should be able to see exactly which dates the
    reported numbers refer to without re-running anything.
    """

    train_idx: pd.Index
    val_idx: pd.Index
    test_idx: pd.Index
    train_end: pd.Timestamp
    val_end: pd.Timestamp

    def summary(self, df: pd.DataFrame, target: str, date_column: str) -> pd.DataFrame:
        """Per-split row counts, date ranges and base rates.

        The base rate column is the point of this table: it exposes the drift
        that a random split would have concealed.
        """
        rows = []
        for name, idx in (("train", self.train_idx), ("validation", self.val_idx),
                          ("test", self.test_idx)):
            part = df.loc[idx]
            rows.append({
                "split": name,
                "bookings": len(part),
                "from": part[date_column].min(),
                "to": part[date_column].max(),
                "cancellation_rate": part[target].mean(),
            })
        return pd.DataFrame(rows)


def temporal_split(
    df: pd.DataFrame,
    date_column: str = "booking_date",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Split:
    """Split chronologically into train / validation / test.

    Cut points come from quantiles of ``date_column`` rather than hard-coded
    dates, so the split still makes sense if the extract is refreshed or
    re-cut. Rows missing a date are dropped from all three splits — they
    cannot be placed on the timeline, and silently assigning them to training
    would be worse than losing them.

    Args:
        df: cleaned bookings, one row per booking.
        date_column: the column defining chronology. Must be ``booking_date``
            for this problem; ``arrival_date`` would leak, because a booking
            made today for next year's stay would land in the test split while
            its outcome was already known during training.
        train_frac: proportion of rows, by date order, used for fitting.
        val_frac: proportion used for tuning, calibration and threshold choice.

    Returns:
        A :class:`Split` with disjoint, chronologically ordered indices.

    Raises:
        ValueError: if the fractions do not leave a non-empty test split.
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError(
            f"train_frac={train_frac} and val_frac={val_frac} leave no test split"
        )

    dated = df[df[date_column].notna()]
    train_end = dated[date_column].quantile(train_frac)
    val_end = dated[date_column].quantile(train_frac + val_frac)

    dates = dated[date_column]
    return Split(
        train_idx=dated.index[dates <= train_end],
        val_idx=dated.index[(dates > train_end) & (dates <= val_end)],
        test_idx=dated.index[dates > val_end],
        train_end=train_end,
        val_end=val_end,
    )


def tuning_cv(n_splits: int = 4) -> TimeSeriesSplit:
    """Cross-validator for hyper-parameter search.

    ``TimeSeriesSplit`` rather than ``KFold`` for the same reason the outer
    split is temporal: every fold must train on the past and validate on the
    future. Using KFold here would tune against a leak even though the final
    hold-out was clean, and would select hyper-parameters that flatter a
    model which cannot generalise forward.

    Callers are expected to pass rows already sorted by date.
    """
    return TimeSeriesSplit(n_splits=n_splits)


def sort_by_date(df: pd.DataFrame, date_column: str = "booking_date") -> pd.DataFrame:
    """Return ``df`` ordered by date, as :func:`tuning_cv` requires."""
    return df.sort_values(date_column, kind="stable")
