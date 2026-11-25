"""Temporal decay of detectability under successive transformations.

Experiment
----------
1. Train each detector on clean human vs AI text (generation 0).
2. Take the held-out AI texts and transform them repeatedly with a
   :class:`~detectability.transforms.TransformationSchedule`, so generation
   ``t`` is derived from generation ``t-1``.
3. At every generation record each detector's *detection rate* (recall on the
   AI class) and mean ``P(AI)``.
4. Fit an exponential decay model to each detector's curve::

       D(t) = D_inf + (D_0 - D_inf) * exp(-lambda * t)

   ``lambda`` is the decay constant, ``D_inf`` the asymptotic floor the
   detector settles at, and ``ln(2) / lambda`` the *detectability half-life*
   (generations until half of the eventual loss has happened).

Evasion effectiveness at generation ``t`` is ``1 - D(t) / D(0)``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from .features import FEATURE_NAMES, feature_matrix
from .models import ModelBundle
from .transforms import TransformationSchedule

TRACKED_FEATURES = (
    "burstiness",
    "contraction_rate",
    "discourse_marker_rate",
    "type_token_ratio",
    "mean_word_len",
    "flesch_reading_ease",
)


@dataclass(frozen=True)
class DecayFit:
    detector: str
    d0: float
    d_inf: float
    decay_constant: float
    half_life: float
    r2: float

    def predict(self, t) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        return decay_curve(t, self.d0, self.d_inf, self.decay_constant)


def decay_curve(t, d0: float, d_inf: float, lam: float):
    return d_inf + (d0 - d_inf) * np.exp(-lam * t)


def fit_decay(generations: Sequence[float], rates: Sequence[float], detector: str = "") -> DecayFit:
    """Least-squares fit of the exponential decay model with bounded parameters."""
    t = np.asarray(generations, dtype=float)
    y = np.asarray(rates, dtype=float)
    if len(t) < 3:
        raise ValueError("Need at least three generations to fit a decay curve")

    d0_guess = float(y[0])
    dinf_guess = float(min(y[-1], y[0]))
    p0 = (d0_guess, dinf_guess, 0.3)
    try:
        params, _ = curve_fit(
            decay_curve,
            t,
            y,
            p0=p0,
            bounds=([0.0, 0.0, 0.0], [1.0, 1.0, 10.0]),
            maxfev=20_000,
        )
    except RuntimeError:
        params = np.array([d0_guess, dinf_guess, 0.0])

    d0, d_inf, lam = (float(p) for p in params)
    if abs(d0 - d_inf) < 1e-3:
        # No measurable decay: the rate constant is unidentifiable, so report none.
        lam = 0.0
    residuals = y - decay_curve(t, d0, d_inf, lam)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float((residuals**2).sum()) / ss_tot if ss_tot > 0 else 1.0
    half_life = math.log(2) / lam if lam > 1e-9 else math.inf
    return DecayFit(detector, d0, d_inf, lam, half_life, r2)


@dataclass
class DecayResult:
    curves: pd.DataFrame  # generation x detector -> detection_rate, mean_p_ai
    fits: pd.DataFrame  # one row per detector
    feature_drift: pd.DataFrame  # generation x tracked feature means
    examples: pd.DataFrame  # one document followed through every generation

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.curves.to_csv(out / "decay_curves.csv", index=False)
        self.fits.to_csv(out / "decay_fits.csv", index=False)
        self.feature_drift.to_csv(out / "feature_drift.csv", index=False)
        self.examples.to_csv(out / "example_trajectory.csv", index=False)
        (out / "decay_summary.md").write_text(self.summary_markdown(), encoding="utf-8")
        return out

    def summary_markdown(self) -> str:
        last = int(self.curves["generation"].max())
        gen0 = self.curves[self.curves.generation == 0].set_index("detector")["detection_rate"]
        genN = self.curves[self.curves.generation == last].set_index("detector")["detection_rate"]
        lines = [
            f"| Detector | Detection @ gen 0 | Detection @ gen {last} | Evasion effectiveness | "
            "Decay constant (lambda) | Half-life (gens) | Floor (D_inf) | Fit R^2 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for row in self.fits.itertuples():
            d0, dn = gen0[row.detector], genN[row.detector]
            evasion = 1 - dn / d0 if d0 else 0.0
            hl = "inf" if math.isinf(row.half_life) else f"{row.half_life:.2f}"
            lines.append(
                f"| {row.detector} | {d0:.3f} | {dn:.3f} | {evasion:.1%} | "
                f"{row.decay_constant:.3f} | {hl} | {row.d_inf:.3f} | {row.r2:.3f} |"
            )
        return "\n".join(lines) + "\n"


def run_decay(
    bundles: Mapping[str, ModelBundle],
    ai_texts: Sequence[str],
    schedule: TransformationSchedule,
    generations: int = 8,
    threshold: float = 0.5,
) -> DecayResult:
    """Run the successive-transformation experiment for every detector."""
    if generations < 2:
        raise ValueError("generations must be at least 2")
    texts = list(ai_texts)
    curve_rows: list[dict] = []
    drift_rows: list[dict] = []
    trajectory: list[dict] = [{"generation": 0, "transforms": "", "text": texts[0]}] if texts else []
    tracked_idx = [FEATURE_NAMES.index(f) for f in TRACKED_FEATURES]

    for gen in range(generations + 1):
        if gen > 0:
            if texts:
                first, applied = schedule.apply(texts[0], gen, 0)
                trajectory.append({"generation": gen, "transforms": "+".join(applied), "text": first})
            texts = schedule.apply_all(texts, gen)

        X = feature_matrix(texts)
        drift_rows.append(
            {
                "generation": gen,
                **{name: float(X[:, i].mean()) for name, i in zip(TRACKED_FEATURES, tracked_idx, strict=False)},
            }
        )
        for name, bundle in bundles.items():
            p_ai = bundle.predict_proba(texts)
            curve_rows.append(
                {
                    "generation": gen,
                    "detector": name,
                    "detection_rate": float((p_ai >= threshold).mean()),
                    "mean_p_ai": float(p_ai.mean()),
                }
            )

    curves = pd.DataFrame(curve_rows)
    fits = pd.DataFrame(
        [
            asdict(fit_decay(group["generation"], group["detection_rate"], detector=name))
            for name, group in curves.groupby("detector", sort=False)
        ]
    )
    return DecayResult(curves, fits, pd.DataFrame(drift_rows), pd.DataFrame(trajectory))


def plot_decay(result: DecayResult, path: str | Path) -> Path:
    """Detection rate per generation with fitted decay curves overlaid."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    fits = {row.detector: row for row in result.fits.itertuples()}
    t_smooth = np.linspace(0, result.curves["generation"].max(), 200)
    for name, group in result.curves.groupby("detector", sort=False):
        (line,) = ax.plot(group["generation"], group["detection_rate"], "o", label=name)
        fit = fits[name]
        ax.plot(
            t_smooth,
            decay_curve(t_smooth, fit.d0, fit.d_inf, fit.decay_constant),
            "-",
            color=line.get_color(),
            alpha=0.7,
        )
    ax.set_xlabel("Transformation generation")
    ax.set_ylabel("Detection rate (recall on AI text)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Decay of AI detectability under successive transformations")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
