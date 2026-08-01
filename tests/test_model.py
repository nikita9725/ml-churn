from collections.abc import Callable, Generator
from pathlib import Path

import pandas as pd
import pytest
from fastapi import Depends
from fastapi.testclient import TestClient

from src.api.dependencies import get_loaded_model_repo
from src.dataset.repository import DatasetRepository
from src.main import app
from src.model.repository import ModelNotTrainedError, ModelRepository
from src.model.training import ModelMetrics, train_churn_model

REAL_DATA_PATH = Path(__file__).parent.parent / "data" / "churn_dataset.csv"


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.read_csv(REAL_DATA_PATH)


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
    client: TestClient,
    override_repo: Callable[[DatasetRepository | object], None],
    override_model_repo: Callable[[ModelRepository | object], None],
    model_path: Path,
) -> None:
    override_repo(DatasetRepository(REAL_DATA_PATH))
    override_model_repo(ModelRepository(model_path))
    response = client.post("/model/train")
    assert response.status_code == 200
    data = response.json()
    assert "accuracy" in data
    assert "f1" in data
    assert 0.0 <= data["accuracy"] <= 1.0
    assert 0.0 <= data["f1"] <= 1.0


def test_model_train_endpoint_empty_dataset(
    client: TestClient,
    override_repo: Callable[[object], None],
    override_model_repo: Callable[[ModelRepository | object], None],
    model_path: Path,
) -> None:
    class EmptyRepo:
        @property
        def df(self) -> pd.DataFrame:
            return pd.DataFrame()

    override_repo(EmptyRepo())
    override_model_repo(ModelRepository(model_path))
    response = client.post("/model/train")
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    return tmp_path / "churn_model.joblib"


@pytest.fixture
def override_model_repo() -> Generator[Callable[[ModelRepository | object], None]]:
    def _override(repo: ModelRepository | object) -> None:
        app.dependency_overrides[get_loaded_model_repo] = lambda: repo

    yield _override
    app.dependency_overrides.clear()


def test_save_and_load_model(sample_df: pd.DataFrame, model_path: Path) -> None:
    pipeline, metrics = train_churn_model(sample_df)
    repo = ModelRepository(model_path)
    repo.save(pipeline, metrics)

    loaded_repo = ModelRepository(model_path)
    loaded_repo.load()

    assert loaded_repo.is_trained is True
    assert loaded_repo.metrics is not None
    assert loaded_repo.metrics.accuracy == metrics.accuracy
    assert loaded_repo.metrics.f1 == metrics.f1
    assert loaded_repo.trained_at is not None

    X_test = sample_df.drop(columns=["churn"]).head(2)
    predictions = loaded_repo.pipeline.predict(X_test)
    assert len(predictions) == 2


def test_load_nonexistent_file(model_path: Path) -> None:
    repo = ModelRepository(model_path)
    repo.load()
    assert repo.is_trained is False
    assert repo.metrics is None
    assert repo.trained_at is None


def test_model_status_not_trained(
    client: TestClient,
    override_model_repo: Callable[[ModelRepository | object], None],
    tmp_path: Path,
) -> None:
    override_model_repo(ModelRepository(tmp_path / "nonexistent.joblib"))
    response = client.get("/model/status")
    assert response.status_code == 200
    data = response.json()
    assert data["is_trained"] is False
    assert data["accuracy"] is None
    assert data["f1"] is None


def test_model_status_after_train(
    client: TestClient,
    override_repo: Callable[[DatasetRepository | object], None],
    override_model_repo: Callable[[ModelRepository | object], None],
    model_path: Path,
) -> None:
    override_repo(DatasetRepository(REAL_DATA_PATH))
    override_model_repo(ModelRepository(model_path))

    response = client.post("/model/train")
    assert response.status_code == 200

    response = client.get("/model/status")
    assert response.status_code == 200
    data = response.json()
    assert data["is_trained"] is True
    assert data["accuracy"] is not None
    assert data["f1"] is not None
    assert data["trained_at"] is not None


def test_model_persists_after_restart(
    sample_df: pd.DataFrame, model_path: Path
) -> None:
    pipeline, metrics = train_churn_model(sample_df)
    repo = ModelRepository(model_path)
    repo.save(pipeline, metrics)

    new_repo = ModelRepository(model_path)
    new_repo.load()

    assert new_repo.is_trained is True
    assert new_repo.metrics is not None
    assert new_repo.metrics.accuracy == metrics.accuracy
    assert new_repo.metrics.f1 == metrics.f1


def test_pipeline_raises_model_not_trained_error(model_path: Path) -> None:
    repo = ModelRepository(model_path)
    with pytest.raises(ModelNotTrainedError, match="Model has not been trained yet"):
        _ = repo.pipeline


def test_model_not_trained_error_returns_400(
    client: TestClient,
    override_model_repo: Callable[[ModelRepository | object], None],
    model_path: Path,
) -> None:
    class RepoWithUntrainedPipeline:
        @property
        def pipeline(self) -> None:
            raise ModelNotTrainedError("Model has not been trained yet")

    override_model_repo(RepoWithUntrainedPipeline())

    @app.get("/test-pipeline")
    def test_endpoint(
        model_repo: ModelRepository = Depends(get_loaded_model_repo),
    ) -> str:
        _ = model_repo.pipeline
        return "ok"

    try:
        response = client.get("/test-pipeline")
        assert response.status_code == 400
        assert "Model has not been trained yet" in response.json()["detail"]
    finally:
        app.router.routes = [
            r for r in app.router.routes if getattr(r, "path", None) != "/test-pipeline"
        ]
