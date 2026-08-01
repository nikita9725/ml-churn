from collections.abc import Callable, Generator
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import get_dataset_repo
from src.dataset.preprocessing import (
    TARGET_COLUMN,
    check_class_distribution,
    prepare_data,
    split_data,
)
from src.dataset.repository import DatasetRepository
from src.main import app

REAL_DATA_PATH = Path(__file__).parent.parent / "data" / "churn_dataset.csv"


@pytest.fixture
def repo() -> DatasetRepository:
    return DatasetRepository(REAL_DATA_PATH)


@pytest.fixture
def override_repo() -> Generator[Callable[[Path], DatasetRepository]]:
    def _override(repo_path: Path) -> DatasetRepository:
        repo = DatasetRepository(repo_path)
        app.dependency_overrides[get_dataset_repo] = lambda: repo
        return repo

    yield _override
    app.dependency_overrides.clear()


def test_init_loads_dataframe(repo: DatasetRepository) -> None:
    assert len(repo.df) == 2000
    assert "churn" in repo.df.columns
    assert "monthly_fee" in repo.df.columns


def test_get_preview_returns_rows(repo: DatasetRepository) -> None:
    rows = repo.get_preview(n=3)
    assert len(rows) == 3
    assert "monthly_fee" in rows[0]
    assert "churn" in rows[0]


def test_get_preview_default_n(repo: DatasetRepository) -> None:
    rows = repo.get_preview()
    assert len(rows) == 10


def test_get_preview_n_larger_than_dataset(repo: DatasetRepository) -> None:
    rows = repo.get_preview(n=5000)
    assert len(rows) == 2000


def test_get_info(repo: DatasetRepository) -> None:
    info = repo.get_info()
    assert info.row_count == 2000
    assert info.column_count == 10
    assert "churn" not in info.feature_names
    assert "monthly_fee" in info.feature_names
    assert info.churn_distribution[0] + info.churn_distribution[1] == 2000


def test_get_info_feature_names_exclude_churn(repo: DatasetRepository) -> None:
    info = repo.get_info()
    assert "churn" not in info.feature_names
    assert len(info.feature_names) == info.column_count - 1


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        DatasetRepository(Path("/nonexistent/file.csv"))


def test_dataset_preview_endpoint(
    client: TestClient, override_repo: Callable[[Path], DatasetRepository]
) -> None:
    override_repo(REAL_DATA_PATH)
    response = client.get("/dataset/preview?n=2")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2


def test_dataset_info_endpoint(
    client: TestClient, override_repo: Callable[[Path], DatasetRepository]
) -> None:
    override_repo(REAL_DATA_PATH)
    response = client.get("/dataset/info")
    assert response.status_code == 200
    data = response.json()
    assert data["row_count"] == 2000
    assert data["column_count"] == 10
    assert "churn" not in data["feature_names"]


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "monthly_fee": [
                10.0,
                20.0,
                None,
                40.0,
                50.0,
                60.0,
                70.0,
                80.0,
                90.0,
                100.0,
            ],
            "usage_hours": [1.0, 2.0, 3.0, None, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "support_requests": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1],
            "account_age_months": [12, 24, 36, 48, 12, 24, 36, 48, 12, 24],
            "failed_payments": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "region": [
                "america",
                "europe",
                None,
                "asia",
                "america",
                "europe",
                "asia",
                "america",
                "europe",
                "asia",
            ],
            "device_type": [
                "mobile",
                "desktop",
                "mobile",
                None,
                "tablet",
                "mobile",
                "desktop",
                "mobile",
                "tablet",
                "mobile",
            ],
            "payment_method": [
                "card",
                "paypal",
                "card",
                "paypal",
                None,
                "card",
                "paypal",
                "card",
                "paypal",
                "card",
            ],
            "autopay_enabled": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "churn": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )


def test_prepare_data_separates_x_and_y(sample_df: pd.DataFrame) -> None:
    X, y = prepare_data(sample_df)
    assert TARGET_COLUMN not in X.columns
    assert y.name == "churn"
    assert len(X) == 10
    assert len(y) == 10


def test_prepare_data_fills_numeric_na(sample_df: pd.DataFrame) -> None:
    X, _ = prepare_data(sample_df)
    assert X["monthly_fee"].isnull().sum() == 0
    assert X["usage_hours"].isnull().sum() == 0


def test_prepare_data_fills_categorical_na(sample_df: pd.DataFrame) -> None:
    X, _ = prepare_data(sample_df)
    assert X["region"].isnull().sum() == 0
    assert X["device_type"].isnull().sum() == 0
    assert X["payment_method"].isnull().sum() == 0


def test_prepare_data_does_not_modify_original(sample_df: pd.DataFrame) -> None:
    original_na = sample_df["monthly_fee"].isnull().sum()
    prepare_data(sample_df)
    assert sample_df["monthly_fee"].isnull().sum() == original_na


def test_split_data_sizes(sample_df: pd.DataFrame) -> None:
    prepared = prepare_data(sample_df)
    split = split_data(prepared, test_size=0.2)
    assert len(split.X_train) == 8
    assert len(split.X_test) == 2
    assert len(split.y_train) == 8
    assert len(split.y_test) == 2


def test_split_data_stratification(sample_df: pd.DataFrame) -> None:
    prepared = prepare_data(sample_df)
    split = split_data(prepared, test_size=0.2)
    train_ratio = split.y_train.mean()
    test_ratio = split.y_test.mean()
    assert abs(train_ratio - test_ratio) <= 0.01


def test_check_class_distribution_passes() -> None:
    y_train = pd.Series([0, 0, 0, 0, 1, 1])
    y_test = pd.Series([0, 1])
    result = check_class_distribution(y_train, y_test, tolerance=0.2)
    assert hasattr(result, "train")
    assert hasattr(result, "test")
    assert 0 in result.train
    assert 1 in result.train


def test_check_class_distribution_raises_on_imbalance() -> None:
    y_train = pd.Series([0, 0, 0, 0, 0, 1])
    y_test = pd.Series([1, 1])
    with pytest.raises(ValueError, match="distribution differs"):
        check_class_distribution(y_train, y_test)


def test_split_info_endpoint(
    client: TestClient, override_repo: Callable[[Path], DatasetRepository]
) -> None:
    override_repo(REAL_DATA_PATH)
    response = client.get("/dataset/split-info")
    assert response.status_code == 200
    data = response.json()
    assert data["train_size"] == 1600
    assert data["test_size"] == 400
    assert (
        data["train_distribution"]["count_0"] + data["train_distribution"]["count_1"]
        == 1600
    )
    assert (
        data["test_distribution"]["count_0"] + data["test_distribution"]["count_1"]
        == 400
    )
    assert (
        abs(
            data["train_distribution"]["ratio_0"]
            + data["train_distribution"]["ratio_1"]
            - 1.0
        )
        < 0.001
    )
    assert (
        abs(
            data["test_distribution"]["ratio_0"]
            + data["test_distribution"]["ratio_1"]
            - 1.0
        )
        < 0.001
    )
