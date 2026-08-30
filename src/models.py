"""Model registry and construction.

Every entry in :data:`REGISTRY` returns a fitted-able scikit-learn ``Pipeline``
that takes the *cleaned bookings frame* as input and produces a cancellation
probability. Feature engineering lives inside the pipeline rather than beside
it, which is what makes train-only fitting automatic: calling ``fit`` on a
training slice cannot see validation or test rows.

Two entries are not machine-learning models at all:

``majority``
    Predicts the base rate for everyone. Present to demonstrate why accuracy
    is a useless metric here — at a 72% majority class it scores 0.72 while
    identifying nobody.

``value_sort``
    Ranks by booking value with no model whatsoever. This is the bar that
    matters: ``eda.ipynb`` section 9.6 measured it capturing 30.3% of realised
    loss at a 10% intervention budget, against 38.6% for the fitted model. A
    model that cannot beat it is not worth deploying, whatever its AUC, so it
    belongs in the registry rather than in a footnote.

Gradient boosting is the expected winner and the EDA says why: no numeric
feature exceeds |0.30| Spearman correlation with the target (section 3.6),
while categorical levels differ four-fold — so the signal lives in
interactions between categoricals, which is precisely what a boosted tree
ensemble captures and a linear model does not.

``HistGradientBoostingClassifier`` is chosen over XGBoost or LightGBM
deliberately. It ships inside scikit-learn, so ``requirements.txt`` stays
small and the GitHub Action cannot fail on a compiled dependency.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .features import FeatureBuilder


class MajorityBaseline(BaseEstimator, ClassifierMixin):
    """Predict the training base rate for every booking.

    Deliberately useless as a ranker — every score is identical, so its ROC-AUC
    is 0.5 by construction. Its purpose is to put a number on the accuracy
    trap: it will beat most real models on accuracy while being commercially
    worthless.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MajorityBaseline":
        self.classes_ = np.array([0, 1])
        self.base_rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.full(len(X), self.base_rate_)
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class StratifiedDummy(BaseEstimator, ClassifierMixin):
    """Random scores drawn from the training base rate.

    The canonical statistical floor. ``MajorityBaseline`` pins the *threshold*
    metrics (accuracy, recall) but its constant score makes its loss-capture
    number degenerate: a constant probability multiplied by value reproduces
    the value ranking exactly, so it silently duplicates ``value_sort``. This
    one produces genuinely uninformative *rankings*, so loss capture should
    land near the intervention budget itself — which is the number every other
    model has to beat.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StratifiedDummy":
        self.classes_ = np.array([0, 1])
        self.base_rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        p = rng.random(len(X))
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 1 - self.base_rate_).astype(int)


class LeadTimeRule(BaseEstimator, ClassifierMixin):
    """Rank by lead time alone — the rule a revenue manager would write.

    ``eda.ipynb`` section 3.5 found cancellation climbing monotonically from
    9.8% to 56.2% across lead-time bands, making "the further ahead it was
    booked, the riskier it is" the obvious non-model heuristic. Including it
    keeps the comparison honest: a machine-learning pipeline should have to
    beat the single most predictive column used on its own, not just chance.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LeadTimeRule":
        self.classes_ = np.array([0, 1])
        lead = pd.to_numeric(X["lead_time"], errors="coerce")
        self.lo_, self.hi_ = float(lead.min()), float(lead.max())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        lead = pd.to_numeric(X["lead_time"], errors="coerce").fillna(self.lo_)
        span = max(self.hi_ - self.lo_, 1e-9)
        p = ((lead - self.lo_) / span).clip(0, 1).to_numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class ValueSortBaseline(BaseEstimator, ClassifierMixin):
    """Rank purely by booking value; no learning of any kind.

    Scores are min-max scaled into [0, 1] so the same evaluation code can
    consume them, but they are *not* probabilities and the Brier score for this
    model is meaningless. It exists to be beaten on the loss-captured metric.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ValueSortBaseline":
        self.classes_ = np.array([0, 1])
        values = pd.to_numeric(X["booking_value"], errors="coerce")
        self.lo_, self.hi_ = float(values.min()), float(values.max())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        values = pd.to_numeric(X["booking_value"], errors="coerce").fillna(self.lo_)
        span = max(self.hi_ - self.lo_, 1e-9)
        p = ((values - self.lo_) / span).clip(0, 1).to_numpy()
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _one_hot_encoder(builder: FeatureBuilder) -> ColumnTransformer:
    """Encoding for models that cannot consume raw categoricals.

    ``handle_unknown="infrequent_if_exist"`` matters in production: a category
    absent from training must not raise at scoring time. Numerics are scaled
    because logistic regression's regularisation penalty is scale-sensitive —
    ``booking_value`` spans thousands while ``is_repeated_guest`` is 0/1.
    """
    return ColumnTransformer(
        [
            ("num", StandardScaler(), builder.numeric_features()),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  min_frequency=20, sparse_output=False),
             builder.categorical_features()),
        ],
        remainder="drop",
    )


def _ordinal_encoder(builder: FeatureBuilder) -> ColumnTransformer:
    """Encoding for tree ensembles that need numeric input but not one-hot.

    Trees split on thresholds, so an arbitrary integer coding is harmless — and
    it avoids the dimensionality blow-up one-hot would cause on the grouped
    country and agent columns.
    """
    return ColumnTransformer(
        [
            ("num", "passthrough", builder.numeric_features()),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=-1), builder.categorical_features()),
        ],
        remainder="drop",
    )


def _hist_gbm(params: dict[str, Any], builder: FeatureBuilder) -> Pipeline:
    """Gradient boosting with native categorical support — the primary model."""
    defaults = dict(max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
                    min_samples_leaf=40, l2_regularization=0.0, random_state=42)
    defaults.update(params)
    return Pipeline([
        ("features", builder),
        # "from_dtype" reads the pandas `category` dtype that FeatureBuilder
        # sets, so no encoder is needed and the tree splits on categories
        # directly rather than on an arbitrary integer coding.
        ("model", HistGradientBoostingClassifier(categorical_features="from_dtype",
                                                 **defaults)),
    ])


def _logistic(params: dict[str, Any], builder: FeatureBuilder) -> Pipeline:
    """Interpretable linear baseline.

    ``class_weight="balanced"`` because the 28% base rate otherwise pulls the
    decision boundary towards the majority class. Coefficients are a useful
    sanity check on the feature set: a wildly dominant one usually means
    leakage has crept back in.
    """
    defaults = dict(max_iter=2000, C=1.0, class_weight="balanced", random_state=42)
    defaults.update(params)
    return Pipeline([
        ("features", builder),
        ("encode", _one_hot_encoder(builder)),
        ("model", LogisticRegression(**defaults)),
    ])


def _random_forest(params: dict[str, Any], builder: FeatureBuilder) -> Pipeline:
    """Bagged trees — a different inductive bias from boosting, as a check."""
    defaults = dict(n_estimators=300, min_samples_leaf=5, n_jobs=-1,
                    class_weight="balanced_subsample", random_state=42)
    defaults.update(params)
    return Pipeline([
        ("features", builder),
        ("encode", _ordinal_encoder(builder)),
        ("model", RandomForestClassifier(**defaults)),
    ])


def _majority(params: dict[str, Any], builder: FeatureBuilder) -> Pipeline:
    return Pipeline([("features", builder), ("model", MajorityBaseline())])


def _value_sort(params: dict[str, Any], builder: FeatureBuilder) -> Pipeline:
    return Pipeline([("features", builder), ("model", ValueSortBaseline())])


def _dummy_stratified(params: dict[str, Any], builder: FeatureBuilder) -> Pipeline:
    return Pipeline([("features", builder), ("model", StratifiedDummy(**params))])


def _lead_time_rule(params: dict[str, Any], builder: FeatureBuilder) -> Pipeline:
    return Pipeline([("features", builder), ("model", LeadTimeRule())])


#: Name -> constructor. Adding a model means adding one entry here; nothing
#: else in the pipeline needs to change, which is what makes experimentation a
#: single config edit.
REGISTRY: dict[str, Callable[[dict[str, Any], FeatureBuilder], Pipeline]] = {
    "majority": _majority,
    "dummy_stratified": _dummy_stratified,
    "lead_time_rule": _lead_time_rule,
    "value_sort": _value_sort,
    "logistic": _logistic,
    "random_forest": _random_forest,
    "hist_gbm": _hist_gbm,
}

#: Models for which probability calibration is meaningful. The four baselines
#: are excluded because their outputs are rankings, not probability estimates.
CALIBRATABLE: frozenset[str] = frozenset({"logistic", "random_forest", "hist_gbm"})


def build_model(
    name: str,
    params: dict[str, Any] | None = None,
    feature_params: dict[str, Any] | None = None,
) -> Pipeline:
    """Construct an unfitted pipeline by registry name.

    Args:
        name: a key of :data:`REGISTRY`.
        params: estimator hyper-parameters, merged over the defaults.
        feature_params: keyword arguments for :class:`FeatureBuilder`.

    Raises:
        KeyError: if ``name`` is not registered, listing the valid options.
    """
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(REGISTRY)}")
    builder = FeatureBuilder(**(feature_params or {}))
    return REGISTRY[name](params or {}, builder)


def calibrate(pipeline: Pipeline, cv: Any, method: str = "isotonic") -> CalibratedClassifierCV:
    """Wrap a pipeline so its probabilities are calibrated, not merely ranked.

    This matters more here than in a typical classification problem. The
    adopted decision rule is ``P(cancel) x booking_value`` (section 9.5), so a
    miscalibrated probability multiplied by dollars produces a meaningless
    ordering — the error is not a monotone rescaling and it does not cancel.

    Isotonic rather than sigmoid because the dataset is large enough (~76k
    training rows) for the non-parametric fit not to overfit, and the
    miscalibration in boosted trees is not reliably sigmoid-shaped.

    The caller should pass a ``TimeSeriesSplit`` so calibration folds respect
    chronology, exactly as tuning does.
    """
    return CalibratedClassifierCV(pipeline, method=method, cv=cv)
