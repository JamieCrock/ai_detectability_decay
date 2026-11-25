"""Classification metrics and side-by-side detector comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .models import ModelBundle, fit_bundle


def classification_metrics(y_true, p_ai, threshold: float = 0.5) -> dict[str, float]:
    """Accuracy, precision, recall, F1 and ROC-AUC with AI as the positive class."""
    y_true = np.asarray(y_true)
    p_ai = np.asarray(p_ai, dtype=float)
    y_pred = (p_ai >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, p_ai)
    return {k: float(v) for k, v in metrics.items()}


def compare_detectors(
    train: pd.DataFrame,
    test: pd.DataFrame,
    names: list[str],
    trained_on: str,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, ModelBundle]]:
    """Fit every named detector on ``train`` and score it on ``test``."""
    rows = []
    bundles: dict[str, ModelBundle] = {}
    for name in names:
        bundle = fit_bundle(name, train["text"].tolist(), train["label"].to_numpy(), trained_on, seed)
        metrics = classification_metrics(test["label"], bundle.predict_proba(test["text"].tolist()))
        bundle.metrics = metrics
        bundles[name] = bundle
        rows.append({"detector": name, **metrics})
    table = pd.DataFrame(rows).set_index("detector").sort_values("f1", ascending=False)
    return table, bundles
