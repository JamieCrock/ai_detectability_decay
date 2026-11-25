import numpy as np
import pytest

from detectability.evaluate import classification_metrics
from detectability.features import FEATURE_NAMES
from detectability.models import DETECTORS, ModelBundle, build_detector


def test_all_detectors_build():
    for name in DETECTORS:
        assert build_detector(name).steps


def test_unknown_detector_raises():
    with pytest.raises(ValueError, match="Unknown detector"):
        build_detector("magic")


def test_detectors_separate_synthetic_classes(trained):
    table, _ = trained
    assert (table["f1"] > 0.9).all(), table


def test_bundle_carries_explanation_metadata(trained):
    _, bundles = trained
    rf = bundles["random_forest"]
    assert set(rf.feature_importances) == set(FEATURE_NAMES)
    assert sum(rf.feature_importances.values()) == pytest.approx(1.0, abs=1e-3)
    assert set(rf.class_means) == {"human", "ai"}
    assert rf.metrics["accuracy"] > 0.9


def test_tfidf_bundle_has_no_feature_metadata(trained):
    _, bundles = trained
    assert bundles["tfidf_logistic"].class_means == {}


def test_bundle_round_trip(tmp_path, trained):
    _, bundles = trained
    path = bundles["random_forest"].save(tmp_path / "rf.joblib")
    loaded = ModelBundle.load(path)
    texts = ["I don't know, honestly it was fine! Would I go again? Probably, yeah. My mate loved it though."]
    assert np.allclose(loaded.predict_proba(texts), bundles["random_forest"].predict_proba(texts))


def test_classification_metrics_known_values():
    y = [1, 1, 0, 0]
    p = [0.9, 0.2, 0.1, 0.7]
    m = classification_metrics(y, p)
    assert m["accuracy"] == 0.5
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["false_positive_rate"] == 0.5
    assert m["roc_auc"] == 0.75
