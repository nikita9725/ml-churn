from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from src.dataset.preprocessing import check_class_distribution, prepare_data, split_data
from src.dataset.repository import DatasetRepository

app = FastAPI()

_dataset_path = Path(__file__).resolve().parent.parent / "data" / "churn_dataset.csv"
dataset_repo = DatasetRepository(_dataset_path)


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


@app.get("/", response_model=HealthResponse)
async def root() -> HealthResponse:
    return HealthResponse(message="ml churn service is running")


@app.post("/predict", response_model=FeatureVectorChurn)
async def predict(features: FeatureVectorChurn) -> FeatureVectorChurn:
    return features


@app.get("/dataset/preview", response_model=list[DatasetRowChurn])
async def dataset_preview(n: int = 10) -> list[DatasetRowChurn]:
    rows = dataset_repo.get_preview(n)
    return [DatasetRowChurn(**row) for row in rows]


@app.get("/dataset/info", response_model=DatasetInfoResponse)
async def dataset_info() -> DatasetInfoResponse:
    info = dataset_repo.get_info()
    return DatasetInfoResponse(
        row_count=info.row_count,
        column_count=info.column_count,
        feature_names=info.feature_names,
        churn_distribution={str(k): v for k, v in info.churn_distribution.items()},
    )


@app.get("/dataset/split-info", response_model=SplitInfoResponse)
async def dataset_split_info() -> SplitInfoResponse:
    df = dataset_repo.df
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
