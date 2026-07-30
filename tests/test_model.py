from collections.abc import Callable, Generator
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.dataset.repository import DatasetRepository
from src.main import app, get_dataset_repo
from src.model.training import ModelMetrics, train_churn_model

REAL_DATA_PATH = Path(__file__).parent.parent / "data" / "churn_dataset.csv"


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.read_csv(REAL_DATA_PATH)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def override_repo() -> Generator[Callable[[DatasetRepository | object], None]]:
    def _override(repo: DatasetRepository | object) -> None:
        app.dependency_overrides[get_dataset_repo] = lambda: repo

    yield _override
    app.dependency_overrides.clear()


def test_train_churn_model_returns_pipeline_and_metrics(
    sample_df: pd.DataFrame,
) -> None:
    pipeline, metrics = train_churn_model(sample_df)
    assert pipeline is not None
    assert isinstance(metrics, ModelMetrics)


def test_train_churn_model_metrics_are_valid(sample_df: pd.DataFrame) -> None:
    _, metrics = train_churn_model(sample_df)
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.f1 <= 1.0


def test_train_churn_model_pipeline_can_predict(sample_df: pd.DataFrame) -> None:
    pipeline, _ = train_churn_model(sample_df)
    X_test = sample_df.drop(columns=["churn"]).head(2)
    predictions = pipeline.predict(X_test)
    assert len(predictions) == 2
    assert all(p in [0, 1] for p in predictions)


def test_model_train_endpoint(
    client: TestClient, override_repo: Callable[[DatasetRepository | object], None]
) -> None:
    override_repo(DatasetRepository(REAL_DATA_PATH))
    response = client.post("/model/train")
    assert response.status_code == 200
    data = response.json()
    assert "accuracy" in data
    assert "f1" in data
    assert 0.0 <= data["accuracy"] <= 1.0
    assert 0.0 <= data["f1"] <= 1.0


def test_model_train_endpoint_empty_dataset(
    client: TestClient, override_repo: Callable[[object], None]
) -> None:
    class EmptyRepo:
        @property
        def df(self) -> pd.DataFrame:
            return pd.DataFrame()

    override_repo(EmptyRepo())
    response = client.post("/model/train")
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()
