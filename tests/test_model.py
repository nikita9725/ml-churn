from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest
from fastapi import Depends
from fastapi.testclient import TestClient

from src.api.dependencies import get_loaded_model_repo
from src.dataset.preprocessing import prepare_data, split_data
from src.dataset.repository import DatasetRepository
from src.exceptions import ErrorCode, ModelNotTrainedError
from src.main import app
from src.model.repository import ModelRepository
from src.model.training import ModelMetrics, train_churn_model
from src.schemas import TrainingConfigChurn

REAL_DATA_PATH = Path(__file__).parent.parent / "data" / "churn_dataset.csv"


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.read_csv(REAL_DATA_PATH)


def test_train_churn_model_returns_pipeline_and_metrics(
    sample_df: pd.DataFrame,
) -> None:
    pipeline, metrics, config = train_churn_model(sample_df)
    assert pipeline is not None
    assert isinstance(metrics, ModelMetrics)
    assert config is not None
    assert config.model_type == "logreg"  # Дефолтный тип


def test_train_churn_model_metrics_are_valid(sample_df: pd.DataFrame) -> None:
    _, metrics, _ = train_churn_model(sample_df)
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.f1 <= 1.0
    assert 0.0 <= metrics.roc_auc <= 1.0


def test_train_churn_model_pipeline_can_predict(sample_df: pd.DataFrame) -> None:
    pipeline, _, _ = train_churn_model(sample_df)
    X_test = sample_df.drop(columns=["churn"]).head(2)
    predictions = pipeline.predict(X_test)
    assert len(predictions) == 2
    assert all(p in [0, 1] for p in predictions)


def test_pipeline_rejects_unknown_category(sample_df: pd.DataFrame) -> None:
    pipeline, _, _ = train_churn_model(sample_df)
    X_unknown = sample_df.drop(columns=["churn"]).head(1).copy()
    X_unknown["region"] = "mars"
    with pytest.raises(ValueError, match="unknown"):
        pipeline.predict(X_unknown)


def test_pipeline_handles_missing_values(sample_df: pd.DataFrame) -> None:
    pipeline, _, _ = train_churn_model(sample_df)
    X_with_na = sample_df.drop(columns=["churn"]).head(2).copy()
    X_with_na.loc[0, "monthly_fee"] = None
    X_with_na.loc[1, "region"] = None
    predictions = pipeline.predict(X_with_na)
    assert len(predictions) == 2
    assert all(p in [0, 1] for p in predictions)


def test_imputer_uses_only_train_statistics() -> None:
    """
    Тест доказывает отсутствие утечки данных из test в train при импутации.

    Создаём датасет, где train и test имеют сильно разные распределения:
    - train monthly_fee: [10, 20, NaN, 40, 50, 60, 70, 80] → median = 50
    - test monthly_fee: [1000, 2000]

    Если бы импутация происходила на всём датасете (утечка):
    - overall median = median([10,20,40,50,60,70,80,1000,2000]) = 60

    При корректной импутации только на train:
    - train median = median([10,20,40,50,60,70,80]) = 50

    Проверяем, что SimpleImputer запомнил именно train median (50), а не overall (60).
    """

    df = pd.DataFrame(
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
                1000.0,
                2000.0,
            ],
            "usage_hours": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            "support_requests": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1],
            "account_age_months": [12, 24, 36, 48, 12, 24, 36, 48, 12, 24],
            "failed_payments": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "region": [
                "america",
                "europe",
                "asia",
                "america",
                "europe",
                "asia",
                "america",
                "europe",
                "asia",
                "america",
            ],
            "device_type": [
                "mobile",
                "desktop",
                "tablet",
                "mobile",
                "desktop",
                "tablet",
                "mobile",
                "desktop",
                "tablet",
                "mobile",
            ],
            "payment_method": [
                "card",
                "paypal",
                "crypto",
                "card",
                "paypal",
                "crypto",
                "card",
                "paypal",
                "crypto",
                "card",
            ],
            "autopay_enabled": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "churn": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    prepared = prepare_data(df)
    split = split_data(prepared, test_size=0.2, random_state=42)

    pipeline, _, _ = train_churn_model(df)

    # Получаем SimpleImputer из pipeline
    preprocessor = pipeline.named_steps["preprocessor"]
    numeric_pipeline = preprocessor.named_transformers_["num"]
    imputer = numeric_pipeline.named_steps["imputer"]

    # Проверяем, что imputer запомнил median из train (50.0), а не из всего датасета (60.0)
    # monthly_fee — первый числовой признак (индекс 0)
    monthly_fee_idx = list(preprocessor.transformers_[0][2]).index("monthly_fee")
    train_median = split.X_train["monthly_fee"].median()
    imputer_value = imputer.statistics_[monthly_fee_idx]

    assert imputer_value == train_median, (
        f"Imputer uses wrong statistic: got {imputer_value}, expected train median {train_median}. "
        f"This indicates data leakage from test to train."
    )
    assert imputer_value == 50.0, (
        f"Expected train median to be 50.0, got {imputer_value}"
    )


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
    assert "roc_auc" in data
    assert 0.0 <= data["accuracy"] <= 1.0
    assert 0.0 <= data["f1"] <= 1.0
    assert 0.0 <= data["roc_auc"] <= 1.0


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
    data = response.json()
    assert data["code"] == ErrorCode.EMPTY_DATASET.value
    assert "empty" in data["message"].lower()


def test_save_and_load_model(sample_df: pd.DataFrame, model_path: Path) -> None:
    pipeline, metrics, config = train_churn_model(sample_df)
    repo = ModelRepository(model_path)
    repo.save(pipeline, metrics, config)

    loaded_repo = ModelRepository(model_path)
    loaded_repo.load()

    assert loaded_repo.is_trained is True
    assert loaded_repo.metrics is not None
    assert loaded_repo.metrics.accuracy == metrics.accuracy
    assert loaded_repo.metrics.f1 == metrics.f1
    assert loaded_repo.metrics.roc_auc == metrics.roc_auc
    assert loaded_repo.trained_at is not None
    assert loaded_repo.config is not None
    assert loaded_repo.config.model_type == config.model_type

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
    assert data["roc_auc"] is not None
    assert data["trained_at"] is not None


def test_model_persists_after_restart(
    sample_df: pd.DataFrame, model_path: Path
) -> None:
    pipeline, metrics, config = train_churn_model(sample_df)
    repo = ModelRepository(model_path)
    repo.save(pipeline, metrics, config)

    new_repo = ModelRepository(model_path)
    new_repo.load()

    assert new_repo.is_trained is True
    assert new_repo.metrics is not None
    assert new_repo.metrics.accuracy == metrics.accuracy
    assert new_repo.metrics.f1 == metrics.f1
    assert new_repo.metrics.roc_auc == metrics.roc_auc
    assert new_repo.config is not None
    assert new_repo.config.model_type == config.model_type


def test_pipeline_raises_model_not_trained_error(model_path: Path) -> None:
    repo = ModelRepository(model_path)
    with pytest.raises(ModelNotTrainedError):
        _ = repo.pipeline


def test_model_not_trained_error_returns_400(
    client: TestClient,
    override_model_repo: Callable[[ModelRepository | object], None],
    model_path: Path,
) -> None:
    class RepoWithUntrainedPipeline:
        @property
        def pipeline(self) -> None:
            raise ModelNotTrainedError()

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
        data = response.json()
        assert data["code"] == ErrorCode.MODEL_NOT_TRAINED.value
        assert data["message"] == "Model has not been trained yet"
    finally:
        app.router.routes = [
            r for r in app.router.routes if getattr(r, "path", None) != "/test-pipeline"
        ]


def test_train_with_logreg_config(
    sample_df: pd.DataFrame,
) -> None:
    config = TrainingConfigChurn(model_type="logreg", hyperparameters={"C": 0.5})
    pipeline, _metrics, returned_config = train_churn_model(sample_df, config)
    assert pipeline is not None
    assert returned_config.model_type == "logreg"
    assert returned_config.hyperparameters == {"C": 0.5}


def test_train_with_random_forest_config(
    sample_df: pd.DataFrame,
) -> None:
    config = TrainingConfigChurn(
        model_type="random_forest", hyperparameters={"n_estimators": 50}
    )
    pipeline, _metrics, returned_config = train_churn_model(sample_df, config)
    assert pipeline is not None
    assert returned_config.model_type == "random_forest"
    assert returned_config.hyperparameters == {"n_estimators": 50}


def test_train_with_invalid_model_type(
    sample_df: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="Invalid model_type"):
        TrainingConfigChurn(model_type="invalid_model")


def test_model_status_shows_config(
    client: TestClient,
    override_repo: Callable[[DatasetRepository | object], None],
    override_model_repo: Callable[[ModelRepository | object], None],
    model_path: Path,
) -> None:
    override_repo(DatasetRepository(REAL_DATA_PATH))
    override_model_repo(ModelRepository(model_path))

    # Обучаем с дефолтным конфигом
    response = client.post("/model/train")
    assert response.status_code == 200

    # Проверяем, что статус показывает конфиг
    response = client.get("/model/status")
    assert response.status_code == 200
    data = response.json()
    assert data["is_trained"] is True
    assert data["model_type"] == "logreg"
    assert data["hyperparameters"] == {}
