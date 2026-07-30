from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.dataset.repository import DatasetRepository

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "tiny_churn.csv"


@pytest.fixture
def repo():
    return DatasetRepository(FIXTURE_PATH)


def test_init_loads_dataframe(repo):
    assert len(repo._df) == 5
    assert "churn" in repo._df.columns
    assert "monthly_fee" in repo._df.columns


def test_get_preview_returns_rows(repo):
    rows = repo.get_preview(n=3)
    assert len(rows) == 3
    assert rows[0]["monthly_fee"] == 65.5
    assert rows[0]["churn"] == 1


def test_get_preview_default_n(repo):
    rows = repo.get_preview()
    assert len(rows) == 5


def test_get_preview_n_larger_than_dataset(repo):
    rows = repo.get_preview(n=100)
    assert len(rows) == 5


def test_get_info(repo):
    info = repo.get_info()
    assert info.row_count == 5
    assert info.column_count == 10
    assert "churn" not in info.feature_names
    assert "monthly_fee" in info.feature_names
    assert info.churn_distribution == {1: 3, 0: 2}


def test_get_info_feature_names_exclude_churn(repo):
    info = repo.get_info()
    assert "churn" not in info.feature_names
    assert len(info.feature_names) == info.column_count - 1


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        DatasetRepository(Path("/nonexistent/file.csv"))


from src.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_dataset_preview_endpoint(client, monkeypatch):
    monkeypatch.setattr("src.main.dataset_repo", DatasetRepository(FIXTURE_PATH))
    response = client.get("/dataset/preview?n=2")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["monthly_fee"] == 65.5


def test_dataset_info_endpoint(client, monkeypatch):
    monkeypatch.setattr("src.main.dataset_repo", DatasetRepository(FIXTURE_PATH))
    response = client.get("/dataset/info")
    assert response.status_code == 200
    data = response.json()
    assert data["row_count"] == 5
    assert data["column_count"] == 10
    assert "churn" not in data["feature_names"]
    assert data["churn_distribution"] == {"1": 3, "0": 2}
