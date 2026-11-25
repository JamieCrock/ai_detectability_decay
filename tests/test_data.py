import json

import pandas as pd
import pytest

from detectability import data

LONG = " ".join(["word"] * 25)


def test_load_hc3_flattens_answers(tmp_path):
    path = tmp_path / "hc3.jsonl"
    records = [
        {
            "question": "q1",
            "human_answers": [LONG + " a"],
            "chatgpt_answers": [LONG + " b", LONG + " c"],
            "source": "finance",
        },
        {"question": "q2", "human_answers": [LONG + " d"], "chatgpt_answers": [], "source": "medicine"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in records))
    df = data.load_dataset(path)
    assert len(df) == 4
    assert data.class_balance(df["label"]) == {"human": 2, "ai": 2}
    assert set(df["source"]) == {"finance", "medicine"}


def test_load_csv_normalises_labels_and_filters_short(tmp_path):
    path = tmp_path / "d.csv"
    pd.DataFrame(
        {"text": [LONG + " x", LONG + " y", "too short", LONG + " x"], "label": ["AI", "human", "ai", "AI"]}
    ).to_csv(path, index=False)
    df = data.load_dataset(path)
    assert list(df["label"]) == [1, 0]  # short text and duplicate dropped


def test_bad_label_and_missing_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"text": [LONG], "label": ["maybe"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="Unrecognised label"):
        data.load_dataset(bad)

    missing = tmp_path / "missing.csv"
    pd.DataFrame({"body": [LONG]}).to_csv(missing, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        data.load_dataset(missing)


def test_stratified_subsample_and_split(corpus):
    split = data.split(corpus, test_size=0.25, seed=0)
    assert len(split.test) == pytest.approx(len(corpus) * 0.25, abs=1)
    assert split.train["label"].mean() == pytest.approx(0.5, abs=0.02)
