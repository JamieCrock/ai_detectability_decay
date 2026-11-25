import math

import numpy as np
import pytest

from detectability.decay import decay_curve, fit_decay, plot_decay, run_decay
from detectability.transforms import TransformationSchedule


def test_fit_recovers_known_parameters():
    t = np.arange(0, 10)
    y = decay_curve(t, 0.95, 0.30, 0.45)
    fit = fit_decay(t, y)
    assert fit.d0 == pytest.approx(0.95, abs=1e-3)
    assert fit.d_inf == pytest.approx(0.30, abs=1e-3)
    assert fit.decay_constant == pytest.approx(0.45, abs=1e-3)
    assert fit.half_life == pytest.approx(math.log(2) / 0.45, rel=1e-3)
    assert fit.r2 == pytest.approx(1.0)


def test_flat_curve_has_infinite_half_life():
    fit = fit_decay([0, 1, 2, 3], [0.9, 0.9, 0.9, 0.9])
    assert fit.decay_constant == 0.0
    assert math.isinf(fit.half_life)
    assert fit.d0 == pytest.approx(0.9, abs=1e-3)


def test_fit_needs_three_points():
    with pytest.raises(ValueError):
        fit_decay([0, 1], [0.9, 0.5])


def test_run_decay_end_to_end(trained, splits, tmp_path):
    _, bundles = trained
    ai_texts = splits.test.loc[splits.test.label == 1, "text"].tolist()
    schedule = TransformationSchedule(per_step=3, strength=0.9, seed=3)
    result = run_decay(bundles, ai_texts, schedule, generations=5)

    assert set(result.curves["detector"]) == set(bundles)
    assert result.curves["generation"].nunique() == 6
    assert len(result.fits) == len(bundles)
    assert len(result.examples) == 6
    assert result.curves["detection_rate"].between(0, 1).all()

    # Successive transformations should make the AI text look more human for
    # the linguistic-feature detectors.
    rf = result.curves[result.curves.detector == "random_forest"].set_index("generation")["mean_p_ai"]
    assert rf.iloc[-1] < rf.iloc[0]
    drift = result.feature_drift.set_index("generation")
    assert drift["discourse_marker_rate"].iloc[-1] < drift["discourse_marker_rate"].iloc[0]

    out = result.save(tmp_path)
    for name in ("decay_curves.csv", "decay_fits.csv", "feature_drift.csv", "decay_summary.md"):
        assert (out / name).exists()
    assert "| random_forest |" in (out / "decay_summary.md").read_text()
    assert plot_decay(result, tmp_path / "plot.png").stat().st_size > 0


def test_run_decay_rejects_too_few_generations(trained):
    _, bundles = trained
    with pytest.raises(ValueError):
        run_decay(bundles, ["text"], TransformationSchedule(), generations=1)
