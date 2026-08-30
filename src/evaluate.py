"""Evaluation metrics for the cancellation-risk model.

The metric hierarchy here is deliberate and is the part most worth reading.

**Accuracy is reported only to show that it is useless.** The modelled
population is 72% honoured bookings, so "predict nobody cancels" scores 0.72
while identifying nobody. It appears in the output next to the majority
baseline that achieves it, and nowhere else.

**PR-AUC is the primary statistical metric.** At a 28% base rate the precision
recall curve describes what the business cares about — finding the minority
class — without being flattered by the large, easy negative class the way
ROC-AUC is.

**Loss captured at budget is the primary business metric.** The reservations
desk can only contact so many guests, so the operationally meaningful question
is: of the money actually lost in the test period, what share sits inside the
top *k*% of bookings by score? This is reported against the ``value_sort``
baseline every time, because a model that cannot beat "sort by value
descending" should not be deployed however good its AUC looks
(``eda.ipynb`` section 9.6).

**Brier score and calibration are load-bearing, not decorative.** The adopted
ranking rule multiplies the predicted probability by dollars, so a
miscalibrated probability produces a wrong ordering rather than merely a
shifted one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from .temporal import lost_room_nights

#: Intervention budgets at which loss capture is reported, as a share of
#: bookings flagged. 10% is the default operating point; the others bracket it
#: so the shape of the curve is visible rather than a single point estimate.
DEFAULT_BUDGETS: tuple[float, ...] = (0.05, 0.10, 0.20, 0.30)


def realised_loss_per_booking(modelled: pd.DataFrame, panel: pd.DataFrame) -> pd.Series:
    """Money actually lost per booking, as the ground truth for loss capture.

    Uses Rule B from ``eda.ipynb`` section 7.1: for a hotel-night with capacity
    K, honoured occupancy O and C cancelled room-nights, the room-nights
    genuinely lost are ``min(C, K - O)`` — because occupancy in the absence of
    those cancellations would have been ``min(O + C, K)``. Roughly three
    quarters of cancelled room-nights were re-sold and cost nothing, so scoring
    against gross cancelled value would reward the model for finding
    cancellations that did not matter.

    Within a night the cancellations are exchangeable — the counterfactual
    cannot say *which* ones went unfilled — so each carries the night's share
    ``min(C, E) / C``. Honoured bookings get zero.

    Args:
        modelled: the modelled population, one row per booking.
        panel: hotel-night occupancy panel from :func:`src.temporal.occupancy_panel`.

    Returns:
        A Series indexed like ``modelled``, in SGD, zero for honoured bookings.
    """
    cancelled = modelled[modelled["is_canceled"] == 1]
    if cancelled.empty:
        return pd.Series(0.0, index=modelled.index)

    room_nights = lost_room_nights(modelled, panel)
    # Group on the booking index carried through the explosion. Rebuilding it
    # positionally would misattribute: `stayable` and the panel merge each drop
    # rows, so the room-night order no longer lines up with the booking order.
    per_booking = (room_nights.groupby("booking_index")
                   .apply(lambda g: (g["p_lost"] * g["adr"]).sum(), include_groups=False))
    return modelled.index.map(per_booking).to_series(index=modelled.index).fillna(0.0)


def loss_captured_at_budget(
    scores: np.ndarray | pd.Series,
    losses: np.ndarray | pd.Series,
    budgets: tuple[float, ...] = DEFAULT_BUDGETS,
) -> pd.DataFrame:
    """Share of realised loss sitting in the top-scoring k% of bookings.

    Args:
        scores: the ranking score. Any monotone transform gives the same
            answer, so this works equally for probabilities and for the
            unnormalised ``P(cancel) x value`` product.
        losses: realised loss per booking, aligned with ``scores``.
        budgets: fractions of the population to flag.

    Returns:
        One row per budget: bookings flagged, loss captured, and lift over
        flagging at random (for which capture equals the budget itself).
    """
    scores = np.asarray(scores, dtype=float)
    losses = np.asarray(losses, dtype=float)
    total = losses.sum()
    order = np.argsort(-scores, kind="stable")

    rows = []
    for budget in budgets:
        n = max(int(round(len(scores) * budget)), 1)
        captured = losses[order[:n]].sum()
        share = captured / total if total > 0 else float("nan")
        rows.append({
            "budget": budget,
            "bookings_flagged": n,
            "loss_captured": captured,
            "share_of_loss": share,
            "lift_over_random": share / budget if budget > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


def classification_metrics(
    y_true: pd.Series, proba: np.ndarray, threshold: float
) -> dict[str, float]:
    """Threshold-free and threshold-dependent classification metrics.

    ``accuracy`` is included solely so the README can show it being beaten by
    the majority baseline; it is never used to select a model.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "brier": float(brier_score_loss(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
        "base_rate": float(y_true.mean()),
        "positives_predicted": int(y_pred.sum()),
    }


def calibration_table(y_true: pd.Series, proba: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Predicted versus observed cancellation rate, by probability decile.

    A reliability check in tabular form. If predicted and observed diverge, the
    ``P(cancel) x value`` ranking is distorted — a booking predicted at 0.6 but
    truly at 0.3 is over-weighted by exactly the calibration error.
    """
    frame = pd.DataFrame({"y": np.asarray(y_true, dtype=int), "p": proba})
    frame["bucket"] = pd.qcut(frame["p"], bins, labels=False, duplicates="drop")
    out = (frame.groupby("bucket")
           .agg(bookings=("y", "size"), mean_predicted=("p", "mean"),
                observed_rate=("y", "mean"))
           .reset_index(drop=True))
    out["gap"] = out["mean_predicted"] - out["observed_rate"]
    return out


def per_group_metrics(
    y_true: pd.Series, proba: np.ndarray, groups: pd.Series
) -> pd.DataFrame:
    """PR-AUC and ROC-AUC computed separately per group.

    Used for the two properties, whose base rates differ materially
    (City 29.8% against Airport 24.7%). A blended figure can hide a model that
    works well on one and poorly on the other, which matters because the
    reservations teams are separate.
    """
    frame = pd.DataFrame({"y": np.asarray(y_true, dtype=int), "p": proba,
                          "group": np.asarray(groups)})
    rows = []
    for name, part in frame.groupby("group"):
        # AUC is undefined when a group contains a single class.
        if part["y"].nunique() < 2:
            continue
        rows.append({
            "group": name, "bookings": len(part), "base_rate": part["y"].mean(),
            "pr_auc": average_precision_score(part["y"], part["p"]),
            "roc_auc": roc_auc_score(part["y"], part["p"]),
        })
    return pd.DataFrame(rows)


#: Notice-period bands used for the recoverability diagnostic. Chosen to match
#: eda.ipynb section 9.4, where short-notice cancellations were found to be
#: 1.5-3.7x less recoverable than early ones.
NOTICE_BANDS: tuple[float, ...] = (-1, 3, 7, 14, 30, 60, 10_000)
NOTICE_LABELS: tuple[str, ...] = ("0-3 d", "4-7 d", "8-14 d", "15-30 d",
                                  "31-60 d", "60+ d")


def loss_capture_by_notice(
    scores: np.ndarray | pd.Series,
    losses: pd.Series,
    notice_days: pd.Series,
    budget: float = 0.10,
) -> pd.DataFrame:
    """Does the model find the *costly* cancellations, not merely the frequent ones?

    Section 9.4 established that short-notice cancellations are markedly less
    recoverable — 1.5-3.7x more likely to leave a room unsold — but section 9.3
    found their *timing* far less predictable than the fact of cancellation
    (ROC-AUC 0.716 against 0.787). The model is therefore not given any
    notice-period feature, and cannot be: at booking time nobody knows when a
    guest will change their mind.

    That makes this a diagnostic rather than an objective. It answers the
    question the omission raises: **is the model systematically blind to the
    most damaging cancellations?** If capture in the 0-7 day bands were far
    below the others, the ranking would be finding cheap cancellations and
    missing expensive ones, and the intervention design would need to change.

    Args:
        scores: the ranking score actually used to prioritise.
        losses: realised loss per booking.
        notice_days: days between cancellation and arrival; NaN for honoured
            bookings, which are excluded since they carry no notice period.
        budget: share of bookings flagged.

    Returns:
        One row per notice band: realised loss in that band, and how much of it
        the flagged set captures.
    """
    frame = pd.DataFrame({
        "score": np.asarray(scores, dtype=float),
        "loss": np.asarray(losses, dtype=float),
        "notice": np.asarray(notice_days, dtype=float),
    })
    n_flagged = max(int(round(len(frame) * budget)), 1)
    flagged = frame.nlargest(n_flagged, "score").index

    cancelled = frame[frame["notice"].notna()].copy()
    cancelled["band"] = pd.cut(cancelled["notice"], bins=list(NOTICE_BANDS),
                               labels=list(NOTICE_LABELS))
    rows = []
    for band, part in cancelled.groupby("band", observed=True):
        band_loss = part["loss"].sum()
        captured = part.loc[part.index.intersection(flagged), "loss"].sum()
        rows.append({
            "notice_band": str(band),
            "cancellations": len(part),
            "realised_loss": band_loss,
            "loss_captured": captured,
            "share_captured": captured / band_loss if band_loss > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


def choose_threshold(
    y_true: pd.Series, proba: np.ndarray, budget: float
) -> float:
    """Threshold that flags exactly ``budget`` of bookings.

    Expressed as a budget rather than a probability because that is the form
    the business constraint actually takes — the desk has capacity for so many
    contacts per week, not a tolerance for a particular probability. Chosen on
    validation and then applied unchanged to test.
    """
    return float(np.quantile(proba, 1 - budget))


def evaluate(
    y_true: pd.Series,
    proba: np.ndarray,
    losses: pd.Series,
    values: pd.Series,
    groups: pd.Series | None = None,
    notice_days: pd.Series | None = None,
    budgets: tuple[float, ...] = DEFAULT_BUDGETS,
    threshold: float = 0.5,
) -> dict[str, object]:
    """Assemble the full evaluation for one model on one split.

    The ranking used for loss capture is ``proba x value`` — the adopted
    decision rule from section 9.5 — not ``proba`` alone. Section 9.5 tested
    both and found probability-only ranking gives up roughly twelve points of
    loss captured, because cancelled bookings are systematically more valuable
    than honoured ones.

    Returns:
        A JSON-serialisable dict. Tables are converted to records so the whole
        thing can be written to ``metrics.json`` without further handling.
    """
    proba = np.asarray(proba, dtype=float)
    exposure = proba * np.asarray(values, dtype=float)

    result: dict[str, object] = {
        "classification": classification_metrics(y_true, proba, threshold),
        "loss_capture_by_expected_value": loss_captured_at_budget(
            exposure, losses, budgets).to_dict("records"),
        "loss_capture_by_probability_only": loss_captured_at_budget(
            proba, losses, budgets).to_dict("records"),
        "loss_capture_by_value_only": loss_captured_at_budget(
            values, losses, budgets).to_dict("records"),
        "calibration": calibration_table(y_true, proba).to_dict("records"),
        "total_realised_loss": float(np.asarray(losses, dtype=float).sum()),
    }
    if groups is not None:
        result["per_group"] = per_group_metrics(y_true, proba, groups).to_dict("records")
    if notice_days is not None:
        # Diagnostic, not an objective — see loss_capture_by_notice.
        result["loss_capture_by_notice"] = loss_capture_by_notice(
            exposure, losses, notice_days).to_dict("records")
    return result


def summarise_comparison(results: dict[str, dict[str, object]],
                         budget: float = 0.10) -> pd.DataFrame:
    """One row per model, for the README's model-comparison table."""
    rows = []
    for name, res in results.items():
        cls = res["classification"]
        capture = next(r for r in res["loss_capture_by_expected_value"]
                       if abs(r["budget"] - budget) < 1e-9)
        rows.append({
            "model": name,
            "pr_auc": cls["pr_auc"],
            "roc_auc": cls["roc_auc"],
            "brier": cls["brier"],
            "accuracy": cls["accuracy"],
            f"loss_captured@{budget:.0%}": capture["share_of_loss"],
        })
    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False).reset_index(drop=True)
