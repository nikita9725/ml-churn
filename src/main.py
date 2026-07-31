from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.dataset.preprocessing import check_class_distribution, prepare_data, split_data
from src.dataset.repository import DatasetRepository
from src.model.repository import ModelNotTrainedError, ModelRepository
from src.model.training import train_churn_model

_dataset_path = Path(__file__).resolve().parent.parent / "data" / "churn_dataset.csv"
_model_path = Path(__file__).resolve().parent.parent / "models" / "churn_model.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    repo = ModelRepository(_model_path)
    repo.load()
    app.state.model_repo = repo
    yield


app = FastAPI(lifespan=lifespan)


@app.exception_handler(ModelNotTrainedError)
async def model_not_trained_handler(
    request: Request, exc: ModelNotTrainedError
) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


class FeatureVectorChurn(BaseModel):
    monthly_fee: float
    usage_hours: float
    support_requests: int
    account_age_months: int
    failed_payments: int
    region: str
    device_type: str
    payment_method: str
    autopay_enabled: int


class DatasetRowChurn(FeatureVectorChurn):
    churn: int


class HealthResponse(BaseModel):
    message: str


class DatasetInfoResponse(BaseModel):
    row_count: int
    column_count: int
    feature_names: list[str]
    churn_distribution: dict[str, int]


class ClassDistributionResponse(BaseModel):
    count_0: int
    count_1: int
    ratio_0: float
    ratio_1: float


class SplitInfoResponse(BaseModel):
    train_size: int
    test_size: int
    train_distribution: ClassDistributionResponse
    test_distribution: ClassDistributionResponse


class ModelMetricsResponse(BaseModel):
    accuracy: float
    f1: float


class ModelStatusResponse(BaseModel):
    is_trained: bool
    trained_at: datetime | None
    accuracy: float | None
    f1: float | None


def get_dataset_repo() -> DatasetRepository:
    return DatasetRepository(_dataset_path)


def get_loaded_model_repo() -> ModelRepository:
    return app.state.model_repo


@app.get("/", response_model=HealthResponse)
async def root() -> HealthResponse:
    return HealthResponse(message="ml churn service is running")


@app.post("/predict", response_model=FeatureVectorChurn)
async def predict(features: FeatureVectorChurn) -> FeatureVectorChurn:
    return features


@app.get("/dataset/preview", response_model=list[DatasetRowChurn])
async def dataset_preview(
    n: int = 10, repo: DatasetRepository = Depends(get_dataset_repo)
) -> list[DatasetRowChurn]:
    rows = repo.get_preview(n)
    return [DatasetRowChurn(**row) for row in rows]


@app.get("/dataset/info", response_model=DatasetInfoResponse)
async def dataset_info(
    repo: DatasetRepository = Depends(get_dataset_repo),
) -> DatasetInfoResponse:
    info = repo.get_info()
    return DatasetInfoResponse(
        row_count=info.row_count,
        column_count=info.column_count,
        feature_names=info.feature_names,
        churn_distribution={str(k): v for k, v in info.churn_distribution.items()},
    )


@app.get("/dataset/split-info", response_model=SplitInfoResponse)
async def dataset_split_info(
    repo: DatasetRepository = Depends(get_dataset_repo),
) -> SplitInfoResponse:
    df = repo.df
    prepared = prepare_data(df)
    split = split_data(prepared)
    distribution = check_class_distribution(split.y_train, split.y_test)

    train_counts = split.y_train.value_counts()
    test_counts = split.y_test.value_counts()

    return SplitInfoResponse(
        train_size=len(split.X_train),
        test_size=len(split.X_test),
        train_distribution=ClassDistributionResponse(
            count_0=int(train_counts.get(0, 0)),
            count_1=int(train_counts.get(1, 0)),
            ratio_0=distribution.train.get(0, 0.0),
            ratio_1=distribution.train.get(1, 0.0),
        ),
        test_distribution=ClassDistributionResponse(
            count_0=int(test_counts.get(0, 0)),
            count_1=int(test_counts.get(1, 0)),
            ratio_0=distribution.test.get(0, 0.0),
            ratio_1=distribution.test.get(1, 0.0),
        ),
    )


@app.post("/model/train")
async def model_train(
    repo: DatasetRepository = Depends(get_dataset_repo),
    model_repo: ModelRepository = Depends(get_loaded_model_repo),
) -> ModelMetricsResponse:
    """
    Запускает обучение модели на данных из churn_dataset.csv.

    Эндпоинт:
    1. Загружает данные из репозитория
    2. Проверяет что данные не пустые
    3. Обучает модель через train_churn_model
    4. Сохраняет модель на диск
    5. Возвращает метрики accuracy и f1 на тестовой выборке
    """
    df = repo.df

    if df.empty:
        raise HTTPException(status_code=400, detail="Dataset is empty")

    try:
        pipeline, metrics = train_churn_model(df)
        model_repo.save(pipeline, metrics)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Model training failed: {e!s}"
        ) from e

    return ModelMetricsResponse(
        accuracy=metrics.accuracy,
        f1=metrics.f1,
    )


@app.get("/model/status", response_model=ModelStatusResponse)
async def model_status(
    model_repo: ModelRepository = Depends(get_loaded_model_repo),
) -> ModelStatusResponse:
    return ModelStatusResponse(
        is_trained=model_repo.is_trained,
        trained_at=model_repo.trained_at,
        accuracy=model_repo.metrics.accuracy if model_repo.metrics else None,
        f1=model_repo.metrics.f1 if model_repo.metrics else None,
    )
