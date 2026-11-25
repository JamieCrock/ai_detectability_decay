"""Dataset loading and splitting.

Supported inputs
----------------
* **HC3** (``all.jsonl`` or any per-domain file from
  `Hello-SimpleAI/HC3 <https://huggingface.co/datasets/Hello-SimpleAI/HC3>`_):
  each record holds a question with lists of ``human_answers`` and
  ``chatgpt_answers``.
* **Generic CSV / JSONL** with a ``text`` column and a ``label`` column, where
  the label is ``1``/``ai``/``machine`` for AI text and ``0``/``human`` for
  human text. Extra columns (for example ``source`` or ``model``) are kept.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

AI_LABELS = {"1", "ai", "machine", "generated", "chatgpt", "llm", "true"}
HUMAN_LABELS = {"0", "human", "false"}


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    test: pd.DataFrame


def _normalise_label(value) -> int:
    key = str(value).strip().lower()
    if key in AI_LABELS:
        return 1
    if key in HUMAN_LABELS:
        return 0
    raise ValueError(f"Unrecognised label value: {value!r}")


def load_hc3(path: str | Path) -> pd.DataFrame:
    """Flatten an HC3 jsonl file into one row per answer."""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            source = rec.get("source", Path(path).stem)
            for answer in rec.get("human_answers") or []:
                rows.append({"text": answer, "label": 0, "source": source})
            for answer in rec.get("chatgpt_answers") or []:
                rows.append({"text": answer, "label": 1, "source": source})
    return pd.DataFrame(rows)


def load_table(path: str | Path) -> pd.DataFrame:
    """Load a CSV or JSONL file with ``text`` and ``label`` columns."""
    path = Path(path)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix in {".jsonl", ".json"}:
        df = pd.read_json(path, lines=path.suffix == ".jsonl")
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    missing = {"text", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df = df.copy()
    df["label"] = df["label"].map(_normalise_label)
    return df


def load_dataset(path: str | Path, min_words: int = 20, max_samples: int | None = None, seed: int = 42) -> pd.DataFrame:
    """Load HC3 or a generic table, drop very short texts and optionally subsample.

    Subsampling is stratified so the class balance is preserved.
    """
    path = Path(path)
    if path.suffix == ".jsonl" and _looks_like_hc3(path):
        df = load_hc3(path)
    else:
        df = load_table(path)

    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.split().str.len() >= min_words]
    df = df.drop_duplicates(subset="text").reset_index(drop=True)

    if max_samples is not None and len(df) > max_samples:
        df, _ = train_test_split(df, train_size=max_samples, stratify=df["label"], random_state=seed)
        df = df.reset_index(drop=True)
    return df


def _looks_like_hc3(path: Path) -> bool:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                return "human_answers" in rec or "chatgpt_answers" in rec
    return False


def split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> Split:
    """Stratified train/test split."""
    train, test = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)
    return Split(train.reset_index(drop=True), test.reset_index(drop=True))


def class_balance(labels) -> dict[str, int]:
    labels = np.asarray(labels)
    return {"human": int((labels == 0).sum()), "ai": int((labels == 1).sum())}
