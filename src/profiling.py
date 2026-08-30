"""Before/after reporting for the data-processing stage.

The cleaning contract in :mod:`src.cleaning` makes a lot of consequential
changes — 26 spellings of ``deposit_type`` collapse to 3, 140 country values to
18, 931 dates in three non-ISO formats are recovered. None of that is visible
from the pipeline's log line "cleaned 118,196 rows", which is exactly the
situation in which a silent regression survives for months.

This module produces an auditable comparison of the raw table against the
processed one, written to ``artifacts/data_processing_report.md`` on every run.
It answers four questions a reviewer would otherwise have to open the notebook
to check:

1. How many rows were lost at each step, and to which rule?
2. Which columns changed, and by how much?
3. Did the target survive cleaning intact?
4. Were bad values repaired, or quietly dropped?

The module is pure reporting: it computes nothing the pipeline depends on, and
removing it would not change a single prediction.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

#: Columns worth a detailed before/after comparison. Restricted to those the
#: cleaning contract actually touches, so the table stays readable rather than
#: repeating 36 near-identical rows.
PROFILED_COLUMNS: tuple[str, ...] = (
    "is_canceled", "hotel", "deposit_type", "market_segment", "distribution_channel",
    "customer_type", "meal", "country", "reserved_room_type", "assigned_room_type",
    "agent", "company", "promo_code", "lead_time", "adr", "arrival_date_month",
    "reservation_status", "reservation_status_date",
)

#: Tokens the raw extract uses interchangeably to mean "no value". Counted as
#: missing in the "before" column so the comparison is like-for-like — the
#: cleaned frame represents all of these as NaN.
_RAW_MISSING = {"", "-", "na", "n/a", "nan", "null", "none", "unknown"}


def row_accounting(
    raw: pd.DataFrame,
    deduplicated: pd.DataFrame,
    filtered: pd.DataFrame,
    exclusion_reasons: pd.DataFrame,
    modelled: pd.DataFrame,
) -> pd.DataFrame:
    """Row counts at each stage, with the cost of every exclusion rule.

    Presented as a funnel so the cumulative cost is obvious. Rows are dropped
    in four distinct places for four distinct reasons, and conflating them
    would hide which rule is expensive.
    """
    rows = [
        {"stage": "raw bookings table", "rows": len(raw), "removed": 0,
         "reason": "-"},
        {"stage": "after de-duplication", "rows": len(deduplicated),
         "removed": len(raw) - len(deduplicated),
         "reason": "duplicate booking_id (section 1.2)"},
    ]
    for rule in exclusion_reasons.columns:
        rows.append({
            "stage": f"  ...excluded: {rule}", "rows": None,
            "removed": int(exclusion_reasons[rule].sum()),
            "reason": "implausible value (section 3.1)",
        })
    rows.append({
        "stage": "usable for modelling", "rows": len(filtered),
        "removed": len(deduplicated) - len(filtered),
        "reason": "union of the rules above",
    })
    rows.append({
        "stage": "modelled population", "rows": len(modelled),
        "removed": len(filtered) - len(modelled),
        "reason": "block bookings, handled by rule not model (section 8.1)",
    })
    out = pd.DataFrame(rows)
    out["retained_%"] = (out["rows"] / len(raw) * 100).round(2)
    return out


def column_comparison(raw: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    """Per-column before/after: dtype, distinct values and missing count.

    The ``distinct_before`` / ``distinct_after`` pair is the headline. A drop
    from 26 to 3 means whitespace, casing and typo variants were absorbed; if
    that number ever stops collapsing, a normalisation rule has broken.
    """
    rows = []
    for col in PROFILED_COLUMNS:
        if col not in raw.columns:
            continue
        before = raw[col].astype("string").str.strip()
        before_missing = before.isna() | before.str.lower().isin(_RAW_MISSING)

        record = {
            "column": col,
            "dtype_before": "TEXT",  # every column is TEXT in the source database
            "distinct_before": int(before.nunique(dropna=True)),
            "missing_before": int(before_missing.sum()),
        }
        if col in cleaned.columns:
            after = cleaned[col]
            record.update({
                "dtype_after": str(after.dtype),
                "distinct_after": int(after.nunique(dropna=True)),
                "missing_after": int(after.isna().sum()),
            })
        else:
            # Dropped or renamed by cleaning — recorded rather than omitted, so
            # the disappearance is deliberate and visible.
            record.update({"dtype_after": "(dropped)", "distinct_after": 0,
                           "missing_after": 0})
        rows.append(record)

    out = pd.DataFrame(rows)
    out["distinct_collapsed"] = out["distinct_before"] - out["distinct_after"]
    return out.sort_values("distinct_collapsed", ascending=False).reset_index(drop=True)


def target_integrity(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Cross-tabulate the two independent encodings of the outcome.

    ``is_canceled`` and ``reservation_status`` are stored separately in the
    source. If cleaning has damaged the label, they will disagree. Perfect
    agreement across ~118k rows is meaningful evidence that the target is
    trustworthy — see ``eda.ipynb`` section 2.1.
    """
    return pd.crosstab(cleaned["reservation_status"], cleaned["is_canceled"],
                       margins=True, margins_name="total")


def numeric_repairs(raw: pd.DataFrame, cleaned: pd.DataFrame) -> pd.DataFrame:
    """Count values that failed numeric coercion, per column.

    Distinguishes "repaired" from "lost". A value that would not coerce becomes
    NaN, and the row may still be usable; this table shows how often that
    happened so the reader can judge whether coercion was benign.
    """
    numeric_cols = ["lead_time", "adr", "arrival_date_year", "adults", "children",
                    "babies", "days_in_waiting_list", "booking_changes"]
    rows = []
    for col in numeric_cols:
        if col not in raw.columns or col not in cleaned.columns:
            continue
        coerced = pd.to_numeric(raw[col], errors="coerce")
        raw_missing = raw[col].astype("string").str.strip().isin([""]) | raw[col].isna()
        rows.append({
            "column": col,
            "unparseable_as_number": int(coerced.isna().sum() - raw_missing.sum()),
            "blank_in_source": int(raw_missing.sum()),
            "null_after_cleaning": int(cleaned[col].isna().sum()),
        })
    return pd.DataFrame(rows)


def build_report(
    raw: pd.DataFrame,
    cleaned: pd.DataFrame,
    filtered: pd.DataFrame,
    exclusion_reasons: pd.DataFrame,
    modelled: pd.DataFrame,
) -> str:
    """Assemble the full before/after report as Markdown.

    Returns:
        A Markdown document. Written to disk by :func:`write_report` and echoed
        to the log by the pipeline, so the comparison is available both in the
        artifacts directory and in CI output.
    """
    accounting = row_accounting(raw, cleaned, filtered, exclusion_reasons, modelled)
    columns = column_comparison(raw, cleaned)
    integrity = target_integrity(cleaned)
    repairs = numeric_repairs(raw, cleaned)

    collapsed = columns[columns["distinct_collapsed"] > 0]
    parts = [
        "# Data processing — before and after",
        "",
        "Generated by `src/profiling.py` on every pipeline run. Section references "
        "point to `eda.ipynb`, where each rule is justified.",
        "",
        "## 1. Row accounting",
        "",
        accounting.to_markdown(index=False),
        "",
        f"**{len(modelled):,} of {len(raw):,} rows reach the model "
        f"({len(modelled) / len(raw):.1%}).**",
        "",
        "## 2. Column-level comparison",
        "",
        "Every column in the source database is stored as `TEXT`, so *all* typing "
        "is done by the cleaning contract. `distinct_collapsed` counts the spelling, "
        "casing and whitespace variants absorbed by normalisation.",
        "",
        columns.to_markdown(index=False),
        "",
        f"**{len(collapsed)} columns had their cardinality reduced; "
        f"{int(columns['distinct_collapsed'].sum()):,} spurious distinct values were "
        "absorbed in total.**",
        "",
        "## 3. Target integrity",
        "",
        "`is_canceled` and `reservation_status` are stored independently in the "
        "source. Perfect agreement is evidence the label survived cleaning intact.",
        "",
        integrity.to_markdown(),
        "",
        "## 4. Numeric coercion",
        "",
        repairs.to_markdown(index=False),
        "",
    ]
    return "\n".join(parts)


def write_report(report: str, output_dir: str | Path) -> Path:
    """Write the report to ``<output_dir>/data_processing_report.md``."""
    path = Path(output_dir) / "data_processing_report.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return path
