"""End-to-end pipeline: ingest, clean, engineer, split, fit, evaluate, persist.

This is the only module with a ``__main__`` guard and the only one that writes
to disk. Every other module is import-safe and side-effect free, which is what
lets ``eda.ipynb`` reuse the cleaning and temporal code without inheriting the
pipeline's logging configuration or output directory.

Run it with::

    bash run.sh                                    # defaults from config.yaml
    python -m src.pipeline --set model.name=logistic
    python -m src.pipeline --compare-all           # every model in the registry

Stages, in order:

1. **Ingest** — read ``bookings`` from SQLite (:mod:`src.data_access`).
2. **Clean** — apply the declarative contract (:mod:`src.cleaning`).
3. **Filter** — drop implausible rows; separate the block-booking population.
4. **Attribute loss** — reconstruct occupancy and compute realised loss per
   booking, the ground truth for the business metric (:mod:`src.temporal`).
5. **Split** — chronologically on ``booking_date`` (:mod:`src.splits`).
6. **Fit** — build and train the configured model (:mod:`src.models`).
7. **Calibrate** — optional isotonic calibration, fitted over time-ordered CV
   folds across the combined train + validation slice.
8. **Evaluate** — statistical and business metrics (:mod:`src.evaluate`).
9. **Persist** — model, metrics, feature importance, data-processing report.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from . import __version__
from .cleaning import clean_bookings, filter_modelling_rows
from .config import Config, PROJECT_ROOT
from .data_access import load_bookings
from .evaluate import (DEFAULT_BUDGETS, choose_threshold, evaluate,
                       realised_loss_per_booking, summarise_comparison)
from .features import FeatureBuilder
from .models import CALIBRATABLE, REGISTRY, build_model, calibrate
from .profiling import build_report, write_report
from .splits import sort_by_date, temporal_split, tuning_cv
from .temporal import (explode_room_nights, infer_capacity, occupancy_panel,
                       trim_to_stable_window)

LOGGER = logging.getLogger("pipeline")

#: Marks the tour-operator allotment population. Held out of modelling because
#: its outcome is ~99.4% determined, so it needs an operational rule rather
#: than a classifier — see ``eda.ipynb`` section 8.1.
BLOCK_BOOKING_LEVEL = "NON REFUND"


def configure_logging(verbose: bool = True) -> None:
    """Send timestamped logs to stdout so CI captures the whole run."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


class CancellationPipeline:
    """Orchestrates the stages above for one configuration.

    State is held on the instance so stages can be run individually from a
    notebook or a test, rather than only as one monolithic call.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.output_dir = Path(config.get("output.dir", "artifacts"))
        if not self.output_dir.is_absolute():
            self.output_dir = PROJECT_ROOT / self.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -- stages -------------------------------------------------------------

    def ingest_and_clean(self) -> None:
        """Stages 1-3: read, clean, filter, and split off block bookings."""
        LOGGER.info("Loading bookings from SQLite")
        self.raw = load_bookings(self.config.db_path)
        LOGGER.info("  raw rows: %s", f"{len(self.raw):,}")

        self.cleaned = clean_bookings(self.raw)
        self.filtered, self.exclusions = filter_modelling_rows(self.cleaned)
        LOGGER.info("  cleaned: %s | usable: %s (%.1f%% retained)",
                    f"{len(self.cleaned):,}", f"{len(self.filtered):,}",
                    100 * len(self.filtered) / len(self.raw))

        if self.config.get("population.exclude_block_bookings", True):
            is_block = self.filtered["deposit_type"] == BLOCK_BOOKING_LEVEL
            self.modelled = self.filtered.loc[~is_block].copy()
            LOGGER.info("  excluded %s block bookings (deposit_type=%s); "
                        "outcome is ~99.4%% determined, handled by rule",
                        f"{int(is_block.sum()):,}", BLOCK_BOOKING_LEVEL)
        else:
            self.modelled = self.filtered.copy()

        if self.config.get("population.exclude_event_period", False):
            before = len(self.modelled)
            self.modelled = self.modelled[self.modelled["is_event_period"] != 1]
            LOGGER.info("  excluded %s event-period bookings", f"{before - len(self.modelled):,}")

        LOGGER.info("  modelled population: %s | base rate %.1f%%",
                    f"{len(self.modelled):,}", 100 * self.modelled["is_canceled"].mean())

    def attribute_loss(self) -> None:
        """Stage 4: realised loss per booking, the business-metric ground truth.

        Requires the *full* filtered population, not just the modelled subset:
        occupancy on a night depends on every booking held for it, including
        the block bookings excluded from modelling.
        """
        LOGGER.info("Reconstructing occupancy and attributing realised loss")
        capacities = infer_capacity(explode_room_nights(self.filtered))
        LOGGER.info("  inferred capacity: %s", capacities)
        self.panel = trim_to_stable_window(occupancy_panel(self.filtered, capacities))

        self.modelled["realised_loss"] = realised_loss_per_booking(self.modelled, self.panel)
        self.modelled["booking_value"] = (
            self.modelled["adr"] * self.modelled["total_nights"].astype(float))
        LOGGER.info("  realised loss across the modelled population: $%s",
                    f"{self.modelled['realised_loss'].sum():,.0f}")

    def split(self) -> None:
        """Stage 5: chronological train / validation / test."""
        self.splits = temporal_split(
            self.modelled,
            date_column=self.config.get("split.date_column", "booking_date"),
            train_frac=self.config.get("split.train_frac", 0.70),
            val_frac=self.config.get("split.val_frac", 0.15),
        )
        summary = self.splits.summary(self.modelled, "is_canceled",
                                      self.config.get("split.date_column", "booking_date"))
        LOGGER.info("Temporal split (base rates show the drift a random split would hide):")
        # `to_dict` rather than `itertuples`: "from" is a Python keyword, so
        # itertuples silently renames the column and the attribute vanishes.
        for row in summary.to_dict("records"):
            LOGGER.info("  %-10s %7s bookings  %s -> %s  base rate %.1f%%",
                        row["split"], f"{row['bookings']:,}",
                        row["from"].date(), row["to"].date(),
                        100 * row["cancellation_rate"])
        self.split_summary = summary

    def fit(self, model_name: str) -> Any:
        """Stages 6-7: build, optionally tune, fit and calibrate one model."""
        # `model.params` belongs to the *configured* model. Under
        # --compare-all the other models fall back to their own defaults,
        # because hyper-parameters are not transferable between algorithms
        # (a learning rate means nothing to a logistic regression).
        configured = self.config.get("model.name", "hist_gbm")
        params = (dict(self.config.get("model.params", {}) or {})
                  if model_name == configured else {})
        feature_params = dict(self.config.get("features", {}) or {})
        # `features` in config also carries keys FeatureBuilder does not accept.
        feature_params = {k: v for k, v in feature_params.items()
                          if k in FeatureBuilder().get_params()}

        pipeline = build_model(model_name, params, feature_params)
        train = self.modelled.loc[self.splits.train_idx]
        y_train = train["is_canceled"].astype(int)

        if self.config.get("tuning.enabled", False) and model_name in CALIBRATABLE:
            pipeline = self._tune(pipeline, model_name)

        LOGGER.info("Fitting %s on %s training bookings", model_name, f"{len(train):,}")
        pipeline.fit(train, y_train)

        if self.config.get("model.calibrate", True) and model_name in CALIBRATABLE:
            LOGGER.info("  calibrating probabilities (isotonic, time-series folds)")
            # Calibration matters here because the decision rule multiplies the
            # probability by dollars; a miscalibrated score distorts the
            # ranking rather than merely shifting it (section 9.5).
            fit_slice = sort_by_date(
                self.modelled.loc[self.splits.train_idx.union(self.splits.val_idx)])
            calibrated = calibrate(
                build_model(model_name, params, feature_params),
                cv=tuning_cv(self.config.get("tuning.cv_splits", 4)),
            )
            calibrated.fit(fit_slice, fit_slice["is_canceled"].astype(int))
            return calibrated
        return pipeline

    def _tune(self, pipeline: Any, model_name: str) -> Any:
        """Randomised hyper-parameter search over time-ordered folds."""
        from sklearn.model_selection import RandomizedSearchCV

        grid = self.config.get(f"tuning.grids.{model_name}", {}) or {}
        if not grid:
            LOGGER.info("  tuning enabled but no grid for %s; using defaults", model_name)
            return pipeline

        LOGGER.info("  tuning %s over %s candidates", model_name,
                    self.config.get("tuning.n_iter", 25))
        search = RandomizedSearchCV(
            pipeline, {f"model__{k}": v for k, v in grid.items()},
            n_iter=self.config.get("tuning.n_iter", 25),
            cv=tuning_cv(self.config.get("tuning.cv_splits", 4)),
            scoring="average_precision",
            random_state=self.config.random_seed,
            n_jobs=-1, refit=True,
        )
        train = sort_by_date(self.modelled.loc[self.splits.train_idx])
        search.fit(train, train["is_canceled"].astype(int))
        LOGGER.info("  best params: %s", search.best_params_)
        self.best_params_ = search.best_params_
        return search.best_estimator_

    def evaluate_model(self, model: Any, name: str) -> dict[str, object]:
        """Stage 8: score on validation, pick a threshold, report on test."""
        budgets = tuple(self.config.get("evaluation.budgets", list(DEFAULT_BUDGETS)))
        budget = self.config.get("intervention.budget", 0.10)

        val = self.modelled.loc[self.splits.val_idx]
        test = self.modelled.loc[self.splits.test_idx]

        # Threshold is chosen on validation and applied unchanged to test —
        # tuning it on test would report an operating point that does not exist.
        val_proba = model.predict_proba(val)[:, 1]
        threshold = choose_threshold(val["is_canceled"].astype(int), val_proba, budget)

        test_proba = model.predict_proba(test)[:, 1]
        result = evaluate(
            test["is_canceled"].astype(int), test_proba,
            test["realised_loss"], test["booking_value"],
            groups=test["hotel"],
            # NaN for honoured bookings, which have no notice period.
            notice_days=test["status_days_before_arrival"].where(test["is_canceled"] == 1),
            budgets=budgets, threshold=threshold,
        )
        result["model"] = name
        result["n_test"] = int(len(test))

        cls = result["classification"]
        at_budget = next(r for r in result["loss_capture_by_expected_value"]
                         if abs(r["budget"] - budget) < 1e-9)
        LOGGER.info("  %-14s PR-AUC %.3f  ROC-AUC %.3f  Brier %.3f  "
                    "loss captured @%.0f%%: %.1f%%",
                    name, cls["pr_auc"], cls["roc_auc"], cls["brier"],
                    100 * budget, 100 * at_budget["share_of_loss"])
        return result

    def persist(self, model: Any, results: dict[str, dict[str, object]],
                chosen: str) -> None:
        """Stage 9: write model, metrics and reports to the artifacts directory."""
        joblib.dump(model, self.output_dir / "model.joblib")

        report = build_report(self.raw, self.cleaned, self.filtered,
                              self.exclusions, self.modelled)
        report_path = write_report(report, self.output_dir)

        metrics = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "pipeline_version": __version__,
            "chosen_model": chosen,
            "config": self.config.raw,
            "population": {
                "raw_rows": int(len(self.raw)),
                "cleaned_rows": int(len(self.cleaned)),
                "usable_rows": int(len(self.filtered)),
                "modelled_rows": int(len(self.modelled)),
                "base_rate": float(self.modelled["is_canceled"].mean()),
                "total_realised_loss": float(self.modelled["realised_loss"].sum()),
            },
            "split": {
                "train_end": str(self.splits.train_end.date()),
                "val_end": str(self.splits.val_end.date()),
                "summary": json.loads(self.split_summary.to_json(orient="records",
                                                                 date_format="iso")),
            },
            "results": results,
        }
        (self.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2,
                                                                 default=str))

        comparison = summarise_comparison(
            results, budget=self.config.get("intervention.budget", 0.10))
        comparison.to_csv(self.output_dir / "model_comparison.csv", index=False)

        LOGGER.info("Artifacts written to %s", self.output_dir)
        LOGGER.info("  model.joblib · metrics.json · model_comparison.csv · %s",
                    report_path.name)
        LOGGER.info("\n%s", comparison.to_string(index=False))

    # -- entry point --------------------------------------------------------

    def run(self, compare_all: bool = False) -> dict[str, object]:
        """Execute every stage and return the metrics for the chosen model."""
        self.ingest_and_clean()
        self.attribute_loss()
        self.split()

        chosen = self.config.get("model.name", "hist_gbm")
        # The baseline set is always evaluated. Without it the headline number
        # has no reference point, and section 9.6 showed a value sort alone
        # captures 35.6% of the loss with no model at all.
        baselines = ["majority", "dummy_stratified", "lead_time_rule", "value_sort"]
        names = list(REGISTRY) if compare_all else baselines + [chosen]
        names = list(dict.fromkeys(names))

        LOGGER.info("Training and evaluating: %s", ", ".join(names))
        results, fitted = {}, {}
        for name in names:
            fitted[name] = self.fit(name)
            results[name] = self.evaluate_model(fitted[name], name)

        self.persist(fitted[chosen], results, chosen)
        return results[chosen]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line interface. ``--set`` accepts repeated dotted overrides."""
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        description="Travelers' Inn cancellation-risk pipeline.",
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config.yaml"),
                        help="path to the YAML configuration file")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set model.name=logistic")
    parser.add_argument("--compare-all", action="store_true",
                        help="train and evaluate every model in the registry")
    parser.add_argument("--quiet", action="store_true", help="warnings only")
    return parser.parse_args(argv)


def apply_overrides(config: Config, overrides: list[str]) -> Config:
    """Apply ``key.path=value`` overrides onto a loaded config.

    Values are parsed as YAML scalars so ``true``, ``0.15`` and ``[1,2]`` all
    arrive with the right type rather than as strings.
    """
    import yaml

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Malformed override '{item}'; expected KEY=VALUE")
        key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        node = config.raw
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
        LOGGER.info("Config override: %s = %r", key, value)
    return config


def main(argv: list[str] | None = None) -> int:
    """Entry point used by ``run.sh``. Returns a process exit code."""
    args = parse_args(argv)
    configure_logging(verbose=not args.quiet)

    config = apply_overrides(Config.load(args.config), args.set)
    np.random.seed(config.random_seed)

    LOGGER.info("Travelers' Inn cancellation pipeline v%s", __version__)
    CancellationPipeline(config).run(compare_all=args.compare_all)
    LOGGER.info("Done.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
