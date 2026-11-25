import json

import pytest
from fastapi.testclient import TestClient

from detectability.cli import main
from detectability.web.app import create_app

HUMAN = (
    "Honestly I've been doing remote work for about two years now and it's fine. Not gonna lie, video calls "
    "drive me up the wall! Can't complain though. Would I do it again? Probably, yeah."
)
AI = (
    "Remote work has become a significant aspect of modern life. Additionally, it is important to note that "
    "video calls can significantly enhance productivity. Furthermore, numerous individuals utilize video calls "
    "in order to improve productivity. In conclusion, remote work offers various benefits."
)


@pytest.fixture(scope="module")
def client(trained):
    _, bundles = trained
    return TestClient(create_app(bundle=bundles["random_forest"]))


def test_index_and_health(client):
    assert client.get("/health").json() == {"status": "ok"}
    page = client.get("/")
    assert page.status_code == 200
    assert "AI Detectability" in page.text


def test_model_info(client):
    info = client.get("/api/model").json()
    assert info["detector"] == "random_forest"
    assert "synthetic" in info["trained_on"]
    assert 0 <= info["metrics"]["f1"] <= 1


def test_predict_ai_and_human(client):
    ai = client.post("/api/predict", json={"text": AI}).json()
    human = client.post("/api/predict", json={"text": HUMAN}).json()
    assert ai["label"] == "ai" and human["label"] == "human"
    assert ai["p_ai"] > human["p_ai"]
    assert len(ai["top_features"]) == 6
    assert {f["leans"] for f in ai["top_features"]} <= {"ai", "human"}


def test_predict_rejects_short_text(client):
    res = client.post("/api/predict", json={"text": "too short"})
    assert res.status_code == 422


def test_create_app_without_model_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="detectability train"):
        create_app(model_path=tmp_path / "missing.joblib")


def test_cli_demo_train_predict_and_decay(tmp_path, corpus, capsys):
    models = tmp_path / "models"
    assert main(["demo", "--out", str(models)]) == 0
    assert (models / "random_forest.joblib").exists()

    assert main(["predict", AI, "--model", str(models / "random_forest.joblib")]) == 0
    out = capsys.readouterr().out.strip().splitlines()[-1]
    assert json.loads(out)["label"] == "ai"

    csv = tmp_path / "corpus.csv"
    corpus.to_csv(csv, index=False)
    assert (
        main(["train", "--data", str(csv), "--compare", "--out", str(models), "--results", str(tmp_path / "res")]) == 0
    )
    assert (tmp_path / "res" / "detector_comparison.csv").exists()

    assert (
        main(
            [
                "decay",
                "--data",
                str(csv),
                "--detectors",
                "random_forest",
                "logistic",
                "--generations",
                "3",
                "--out",
                str(tmp_path / "decay"),
            ]
        )
        == 0
    )
    assert (tmp_path / "decay" / "decay_curves.png").exists()
