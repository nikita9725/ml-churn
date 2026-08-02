from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator


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


class HealthCheckResponse(BaseModel):
    status: str
    model_available: bool
    dataset_loaded: bool


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
    roc_auc: float


class ModelStatusResponse(BaseModel):
    is_trained: bool
    trained_at: datetime | None
    accuracy: float | None
    f1: float | None
    roc_auc: float | None = None
    model_type: str | None = None
    hyperparameters: dict[str, Any] | None = None


class TrainingHistoryEntryResponse(BaseModel):
    timestamp: datetime
    model_type: str
    hyperparameters: dict[str, Any]
    accuracy: float
    f1: float
    roc_auc: float


class ModelMetricsHistoryResponse(BaseModel):
    latest: TrainingHistoryEntryResponse | None
    history: list[TrainingHistoryEntryResponse]


class PredictionResponseChurn(BaseModel):
    prediction: int = Field(
        description="Предсказанный класс: 0 (не churn) или 1 (churn)"
    )
    probability_churn_0: float = Field(description="Вероятность класса 0 (не churn)")
    probability_churn_1: float = Field(description="Вероятность класса 1 (churn)")
    features: FeatureVectorChurn = Field(description="Входные признаки клиента")


class TrainingConfigChurn(BaseModel):
    model_type: str = Field(
        default="logreg",
        description="Тип модели: logreg или random_forest",
    )
    hyperparameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Гиперпараметры модели",
    )

    @model_validator(mode="after")
    def validate_model_type(self) -> "TrainingConfigChurn":
        valid_types = {"logreg", "random_forest"}
        if self.model_type not in valid_types:
            raise ValueError(
                f"Invalid model_type '{self.model_type}'. Must be one of: {valid_types}"
            )
        return self


class FeatureInfo(BaseModel):
    name: str = Field(description="Название признака")
    type: str = Field(description="Тип данных: numeric или categorical")
    description: str = Field(description="Описание признака")


class ModelSchemaResponse(BaseModel):
    features: list[FeatureInfo] = Field(description="Список признаков модели")
