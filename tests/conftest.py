from collections.abc import Callable, Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import (
    get_dataset_repo,
    get_loaded_model_repo,
    get_training_history_repo,
)
from src.dataset.repository import DatasetRepository
from src.main import app
from src.model.history_repository import (
    InMemoryTrainingHistoryRepository,
    TrainingHistoryRepository,
)
from src.model.repository import ModelRepository


@pytest.fixture(autouse=True)
def init_model_repo(tmp_path: Path) -> None:
    model_path = tmp_path / "churn_model.joblib"
    app.state.model_repo = ModelRepository(model_path)
    app.state.training_history_repo = InMemoryTrainingHistoryRepository()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def override_repo() -> Generator[Callable[[DatasetRepository | object], None]]:
    def _override(repo: DatasetRepository | object) -> None:
        app.dependency_overrides[get_dataset_repo] = lambda: repo

    yield _override
    app.dependency_overrides.clear()


@pytest.fixture
def override_model_repo() -> Generator[Callable[[ModelRepository | object], None]]:
    def _override(repo: ModelRepository | object) -> None:
        app.dependency_overrides[get_loaded_model_repo] = lambda: repo

    yield _override
    app.dependency_overrides.clear()


@pytest.fixture
def override_history_repo() -> Generator[
    Callable[[TrainingHistoryRepository | object], None]
]:
    def _override(repo: TrainingHistoryRepository | object) -> None:
        app.dependency_overrides[get_training_history_repo] = lambda: repo

    yield _override
    app.dependency_overrides.clear()


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    return tmp_path / "churn_model.joblib"
