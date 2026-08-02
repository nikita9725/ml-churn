import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.dataset.repository import DatasetRepository
from src.model.history_repository import (
    InMemoryTrainingHistoryRepository,
    JsonTrainingHistoryRepository,
)
from src.model.repository import ModelRepository
from src.model.training import TrainingHistoryEntry

REAL_DATA_PATH = Path(__file__).parent.parent / "data" / "churn_dataset.csv"


def _make_entry(
    model_type: str = "logreg",
    accuracy: float = 0.8,
    f1: float = 0.7,
    roc_auc: float = 0.85,
    hyperparameters: dict | None = None,
    timestamp: datetime | None = None,
) -> TrainingHistoryEntry:
    return TrainingHistoryEntry(
        timestamp=timestamp or datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
        model_type=model_type,
        hyperparameters=hyperparameters or {},
        accuracy=accuracy,
        f1=f1,
        roc_auc=roc_auc,
    )


# ── InMemoryTrainingHistoryRepository ────────────────────────────────


class TestInMemoryTrainingHistoryRepository:
    def test_empty_on_creation(self) -> None:
        repo = InMemoryTrainingHistoryRepository()
        assert repo.get_all() == []
        assert repo.get_last() == []

    def test_add_and_get_all(self) -> None:
        repo = InMemoryTrainingHistoryRepository()
        entry = _make_entry()
        repo.add(entry)
        assert repo.get_all() == [entry]

    def test_get_all_filters_by_model_type(self) -> None:
        repo = InMemoryTrainingHistoryRepository()
        logreg = _make_entry(model_type="logreg", accuracy=0.8)
        rf = _make_entry(model_type="random_forest", accuracy=0.9)
        repo.add(logreg)
        repo.add(rf)

        assert repo.get_all(model_type="logreg") == [logreg]
        assert repo.get_all(model_type="random_forest") == [rf]
        assert len(repo.get_all()) == 2

    def test_get_last_returns_tail(self) -> None:
        repo = InMemoryTrainingHistoryRepository()
        entries = [
            _make_entry(
                accuracy=0.1 * i, timestamp=datetime(2026, 1, i, 12, 0, 0, tzinfo=UTC)
            )
            for i in range(1, 6)
        ]
        for e in entries:
            repo.add(e)

        last_3 = repo.get_last(3)
        assert len(last_3) == 3
        assert last_3 == entries[-3:]

    def test_get_last_on_empty(self) -> None:
        repo = InMemoryTrainingHistoryRepository()
        assert repo.get_last(5) == []

    def test_get_last_more_than_available(self) -> None:
        repo = InMemoryTrainingHistoryRepository()
        entry = _make_entry()
        repo.add(entry)
        assert repo.get_last(100) == [entry]


# ── JsonTrainingHistoryRepository ────────────────────────────────────


class TestJsonTrainingHistoryRepository:
    def test_load_nonexistent_file(self, tmp_path: Path) -> None:
        repo = JsonTrainingHistoryRepository(tmp_path / "history.json")
        repo.load()
        assert repo.get_all() == []

    def test_add_persists_to_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "history.json"
        repo = JsonTrainingHistoryRepository(path)
        entry = _make_entry()
        repo.add(entry)

        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data) == 1
        assert data[0]["model_type"] == "logreg"
        assert data[0]["accuracy"] == 0.8

    def test_load_roundtrip(self, tmp_path: Path) -> None:
        path = tmp_path / "history.json"
        repo = JsonTrainingHistoryRepository(path)

        entry1 = _make_entry(
            model_type="logreg",
            accuracy=0.8,
            timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC),
        )
        entry2 = _make_entry(
            model_type="random_forest",
            accuracy=0.9,
            hyperparameters={"n_estimators": 200},
            timestamp=datetime(2026, 1, 2, 12, 0, 0, tzinfo=UTC),
        )
        repo.add(entry1)
        repo.add(entry2)

        loaded = JsonTrainingHistoryRepository(path)
        loaded.load()

        entries = loaded.get_all()
        assert len(entries) == 2
        assert entries[0].model_type == "logreg"
        assert entries[0].accuracy == 0.8
        assert entries[1].model_type == "random_forest"
        assert entries[1].hyperparameters == {"n_estimators": 200}
        assert entries[1].roc_auc == 0.85

    def test_filter_after_reload(self, tmp_path: Path) -> None:
        path = tmp_path / "history.json"
        repo = JsonTrainingHistoryRepository(path)
        repo.add(_make_entry(model_type="logreg"))
        repo.add(_make_entry(model_type="random_forest"))

        loaded = JsonTrainingHistoryRepository(path)
        loaded.load()
        assert len(loaded.get_all(model_type="logreg")) == 1
        assert len(loaded.get_all(model_type="random_forest")) == 1


# ── Endpoint: POST /model/train saves history ────────────────────────


@pytest.fixture
def history_repo() -> InMemoryTrainingHistoryRepository:
    return InMemoryTrainingHistoryRepository()


def test_train_adds_history_entry(
    client: TestClient,
    override_repo: Callable,
    override_model_repo: Callable,
    override_history_repo: Callable,
    model_path: Path,
    history_repo: InMemoryTrainingHistoryRepository,
) -> None:
    override_repo(DatasetRepository(REAL_DATA_PATH))
    override_model_repo(ModelRepository(model_path))
    override_history_repo(history_repo)

    response = client.post("/model/train")
    assert response.status_code == 200

    entries = history_repo.get_all()
    assert len(entries) == 1
    assert entries[0].model_type == "logreg"
    assert 0.0 <= entries[0].accuracy <= 1.0
    assert 0.0 <= entries[0].f1 <= 1.0
    assert 0.0 <= entries[0].roc_auc <= 1.0


def test_train_multiple_times_appends_history(
    client: TestClient,
    override_repo: Callable,
    override_model_repo: Callable,
    override_history_repo: Callable,
    model_path: Path,
    history_repo: InMemoryTrainingHistoryRepository,
) -> None:
    override_repo(DatasetRepository(REAL_DATA_PATH))
    override_model_repo(ModelRepository(model_path))
    override_history_repo(history_repo)

    client.post("/model/train")
    client.post("/model/train", json={"model_type": "random_forest"})

    entries = history_repo.get_all()
    assert len(entries) == 2
    assert entries[0].model_type == "logreg"
    assert entries[1].model_type == "random_forest"


# ── Endpoint: GET /model/metrics ─────────────────────────────────────


def test_metrics_empty_history(client: TestClient) -> None:
    response = client.get("/model/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["latest"] is None
    assert data["history"] == []


def test_metrics_after_train(
    client: TestClient,
    override_repo: Callable,
    override_model_repo: Callable,
    override_history_repo: Callable,
    model_path: Path,
    history_repo: InMemoryTrainingHistoryRepository,
) -> None:
    override_repo(DatasetRepository(REAL_DATA_PATH))
    override_model_repo(ModelRepository(model_path))
    override_history_repo(history_repo)

    client.post("/model/train")

    response = client.get("/model/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["latest"] is not None
    assert "accuracy" in data["latest"]
    assert "f1" in data["latest"]
    assert "roc_auc" in data["latest"]
    assert data["latest"]["model_type"] == "logreg"
    assert len(data["history"]) == 1


def test_metrics_filter_by_model_type(
    client: TestClient,
    override_repo: Callable,
    override_model_repo: Callable,
    override_history_repo: Callable,
    model_path: Path,
    history_repo: InMemoryTrainingHistoryRepository,
) -> None:
    override_repo(DatasetRepository(REAL_DATA_PATH))
    override_model_repo(ModelRepository(model_path))
    override_history_repo(history_repo)

    client.post("/model/train")
    client.post("/model/train", json={"model_type": "random_forest"})
    client.post("/model/train")

    response = client.get("/model/metrics?model_type=random_forest")
    assert response.status_code == 200
    data = response.json()
    assert len(data["history"]) == 1
    assert data["history"][0]["model_type"] == "random_forest"
    assert data["latest"]["model_type"] == "random_forest"


def test_metrics_limit(
    client: TestClient,
    override_repo: Callable,
    override_model_repo: Callable,
    override_history_repo: Callable,
    model_path: Path,
    history_repo: InMemoryTrainingHistoryRepository,
) -> None:
    override_repo(DatasetRepository(REAL_DATA_PATH))
    override_model_repo(ModelRepository(model_path))
    override_history_repo(history_repo)

    for _ in range(5):
        client.post("/model/train")

    response = client.get("/model/metrics?limit=2")
    assert response.status_code == 200
    data = response.json()
    assert len(data["history"]) == 2


# ── Endpoint: GET /model/status includes roc_auc ─────────────────────


def test_status_includes_roc_auc(
    client: TestClient,
    override_repo: Callable,
    override_model_repo: Callable,
    model_path: Path,
) -> None:
    override_repo(DatasetRepository(REAL_DATA_PATH))
    override_model_repo(ModelRepository(model_path))

    client.post("/model/train")

    response = client.get("/model/status")
    assert response.status_code == 200
    data = response.json()
    assert data["roc_auc"] is not None
    assert 0.0 <= data["roc_auc"] <= 1.0


# ── train_churn_model returns roc_auc ────────────────────────────────


def test_train_churn_model_returns_roc_auc() -> None:
    from src.model.training import train_churn_model

    df = pd.read_csv(REAL_DATA_PATH)
    _, metrics, _ = train_churn_model(df)
    assert 0.0 <= metrics.roc_auc <= 1.0
