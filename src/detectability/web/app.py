"""FastAPI web app serving the AI-text detector.

Run with ``detectability serve`` (or ``uvicorn detectability.web.app:app``).
The model path comes from ``--model`` or the ``DETECTABILITY_MODEL``
environment variable and defaults to ``models/random_forest.joblib``.
"""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ..features import extract_features, tokenize
from ..models import ModelBundle

DEFAULT_MODEL_PATH = Path("models/random_forest.joblib")
MIN_WORDS = 15
MAX_CHARS = 20_000


class PredictRequest(BaseModel):
    text: str = Field(..., max_length=MAX_CHARS)


class FeatureExplanation(BaseModel):
    name: str
    value: float
    human_mean: float
    ai_mean: float
    leans: str
    importance: float


class PredictResponse(BaseModel):
    p_ai: float
    label: str
    verdict: str
    n_words: int
    top_features: list[FeatureExplanation]


def _verdict(p_ai: float) -> str:
    if p_ai >= 0.85:
        return "Very likely AI-generated"
    if p_ai >= 0.6:
        return "Likely AI-generated"
    if p_ai > 0.4:
        return "Uncertain"
    if p_ai > 0.15:
        return "Likely human-written"
    return "Very likely human-written"


def explain(bundle: ModelBundle, text: str, top_k: int = 6) -> list[FeatureExplanation]:
    """Rank the most important features and say which class each value resembles."""
    if not bundle.feature_importances or not bundle.class_means:
        return []
    feats = extract_features(text)
    human, ai = bundle.class_means["human"], bundle.class_means["ai"]
    ranked = sorted(bundle.feature_importances.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    out = []
    for name, importance in ranked:
        value = feats[name]
        leans = "ai" if abs(value - ai[name]) < abs(value - human[name]) else "human"
        out.append(
            FeatureExplanation(
                name=name,
                value=round(value, 4),
                human_mean=round(human[name], 4),
                ai_mean=round(ai[name], 4),
                leans=leans,
                importance=round(importance, 4),
            )
        )
    return out


def create_app(bundle: ModelBundle | None = None, model_path: str | Path | None = None) -> FastAPI:
    """App factory - pass a fitted bundle directly (tests) or a path to load."""
    if bundle is None:
        path = Path(model_path or os.environ.get("DETECTABILITY_MODEL", DEFAULT_MODEL_PATH))
        if not path.exists():
            raise FileNotFoundError(f"No trained model at {path}. Run `detectability train` (or `make demo`) first.")
        bundle = ModelBundle.load(path)

    app = FastAPI(title="AI Detectability", version="1.0.0")
    page = resources.files("detectability.web").joinpath("static/index.html").read_text(encoding="utf-8")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return page

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/model")
    def model_info() -> dict:
        return {
            "detector": bundle.name,
            "trained_on": bundle.trained_on,
            "n_train": bundle.n_train,
            "metrics": {
                k: round(v, 4)
                for k, v in bundle.metrics.items()
                if k in {"accuracy", "precision", "recall", "f1", "roc_auc"}
            },
        }

    @app.post("/api/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        n_words = len(tokenize(req.text))
        if n_words < MIN_WORDS:
            raise HTTPException(status_code=422, detail=f"Please provide at least {MIN_WORDS} words.")
        p_ai = float(bundle.predict_proba([req.text])[0])
        return PredictResponse(
            p_ai=round(p_ai, 4),
            label="ai" if p_ai >= 0.5 else "human",
            verdict=_verdict(p_ai),
            n_words=n_words,
            top_features=explain(bundle, req.text),
        )

    return app


def __getattr__(name: str):
    # Lazily build ``app`` so ``uvicorn detectability.web.app:app`` works
    # without loading a model at import time.
    if name == "app":
        return create_app()
    raise AttributeError(name)
