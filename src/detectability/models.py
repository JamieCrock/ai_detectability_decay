"""Detector definitions.

Each detector is a scikit-learn ``Pipeline`` that takes raw strings and
outputs ``P(AI)``. Four detectors are provided so results can be compared
across model families and feature representations:

==================  ===============================  ==========================
name                features                         classifier
==================  ===============================  ==========================
``random_forest``   31 linguistic features           RandomForestClassifier
``gradient_boost``  31 linguistic features           HistGradientBoosting
``logistic``        31 linguistic features (scaled)  LogisticRegression
``tfidf_logistic``  word 1-2gram TF-IDF              LogisticRegression
==================  ===============================  ==========================

The random forest on linguistic features is the primary detector used by the
web app; the others act as baselines for the decay comparison.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES, feature_matrix


class LinguisticFeatures(BaseEstimator, TransformerMixin):
    """scikit-learn transformer wrapping :func:`feature_matrix`."""

    def fit(self, X, y=None):  # noqa: N803 - sklearn naming convention
        return self

    def transform(self, X):  # noqa: N803
        return feature_matrix(X)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(FEATURE_NAMES, dtype=object)


def _random_forest(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("features", LinguisticFeatures()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=seed,
                ),
            ),
        ]
    )


def _gradient_boost(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("features", LinguisticFeatures()),
            ("clf", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=seed)),
        ]
    )


def _logistic(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("features", LinguisticFeatures()),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=seed)),
        ]
    )


def _tfidf_logistic(seed: int) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50_000, sublinear_tf=True),
            ),
            ("clf", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced", random_state=seed)),
        ]
    )


DETECTORS: dict[str, Callable[[int], Pipeline]] = {
    "random_forest": _random_forest,
    "gradient_boost": _gradient_boost,
    "logistic": _logistic,
    "tfidf_logistic": _tfidf_logistic,
}


def build_detector(name: str, seed: int = 42) -> Pipeline:
    """Return an unfitted detector pipeline by name."""
    try:
        return DETECTORS[name](seed)
    except KeyError as exc:
        valid = ", ".join(sorted(DETECTORS))
        raise ValueError(f"Unknown detector '{name}'. Choose from: {valid}") from exc


@dataclass
class ModelBundle:
    """A fitted detector plus the metadata the web app needs to explain it."""

    name: str
    pipeline: Pipeline
    trained_on: str
    n_train: int
    class_means: dict[str, dict[str, float]] = field(default_factory=dict)
    feature_importances: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return ``P(AI)`` for each text."""
        return self.pipeline.predict_proba(texts)[:, 1]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> ModelBundle:
        bundle = joblib.load(path)
        if not isinstance(bundle, ModelBundle):
            raise TypeError(f"{path} does not contain a ModelBundle")
        return bundle


def fit_bundle(
    name: str,
    texts: list[str],
    labels: np.ndarray,
    trained_on: str,
    seed: int = 42,
) -> ModelBundle:
    """Fit a detector and capture explanation metadata alongside it."""
    labels = np.asarray(labels)
    pipeline = build_detector(name, seed).fit(texts, labels)

    class_means: dict[str, dict[str, float]] = {}
    importances: dict[str, float] = {}
    if "features" in pipeline.named_steps:
        X = feature_matrix(texts)
        for label, key in ((0, "human"), (1, "ai")):
            rows = X[labels == label]
            if len(rows):
                class_means[key] = dict(zip(FEATURE_NAMES, rows.mean(axis=0).round(5).tolist(), strict=False))
        clf = pipeline.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            importances = dict(zip(FEATURE_NAMES, clf.feature_importances_.round(5).tolist(), strict=False))
        elif hasattr(clf, "coef_"):
            importances = dict(zip(FEATURE_NAMES, np.abs(clf.coef_[0]).round(5).tolist(), strict=False))

    return ModelBundle(
        name=name,
        pipeline=pipeline,
        trained_on=trained_on,
        n_train=len(texts),
        class_means=class_means,
        feature_importances=importances,
    )
