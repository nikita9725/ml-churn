from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from src.main import app
from src.model.repository import ModelRepository
from src.model.training import train_churn_model

REAL_DATA_PATH = Path(__file__).parent.parent / "data" / "churn_dataset.csv"


def test_root(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "ml churn service is running"}


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model_available" in data
    assert "dataset_loaded" in data


def test_health_model_not_trained(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["model_available"] is False


def test_health_model_trained(client: TestClient, model_path: Path) -> None:
    df = pd.read_csv(REAL_DATA_PATH)
    pipeline, metrics, config = train_churn_model(df)

    model_repo = ModelRepository(model_path)
    model_repo.save(pipeline, metrics, config)
    app.state.model_repo = model_repo

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["model_available"] is True
    assert data["status"] == "ok"
