from fastapi import APIRouter, Body, Depends

from src.api.dependencies import get_dataset_repo, get_loaded_model_repo
from src.dataset.preprocessing import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from src.dataset.repository import DatasetRepository
from src.exceptions import EmptyDatasetError, TrainingFailedError
from src.model.repository import ModelRepository
from src.model.training import train_churn_model
from src.schemas import (
    FeatureInfo,
    ModelMetricsResponse,
    ModelSchemaResponse,
    ModelStatusResponse,
    TrainingConfigChurn,
)

router = APIRouter(prefix="/model", tags=["model"])


@router.post("/train")
def model_train(
    repo: DatasetRepository = Depends(get_dataset_repo),
    model_repo: ModelRepository = Depends(get_loaded_model_repo),
    config: TrainingConfigChurn = Body(default=TrainingConfigChurn()),
) -> ModelMetricsResponse:
    """
    Запускает обучение модели на данных из churn_dataset.csv.

    Эндпоинт:
    1. Загружает данные из репозитория
    2. Проверяет что данные не пустые
    3. Обучает модель через train_churn_model с учётом конфига
    4. Сохраняет модель на диск вместе с конфигом
    5. Возвращает метрики accuracy и f1 на тестовой выборке
    """
    df = repo.df

    if df.empty:
        raise EmptyDatasetError()

    try:
        pipeline, metrics, config = train_churn_model(df, config)
        model_repo.save(pipeline, metrics, config)
    except Exception as e:
        raise TrainingFailedError(str(e)) from e

    return ModelMetricsResponse(
        accuracy=metrics.accuracy,
        f1=metrics.f1,
    )


@router.get("/status")
def model_status(
    model_repo: ModelRepository = Depends(get_loaded_model_repo),
) -> ModelStatusResponse:
    return ModelStatusResponse(
        is_trained=model_repo.is_trained,
        trained_at=model_repo.trained_at,
        accuracy=model_repo.metrics.accuracy if model_repo.metrics else None,
        f1=model_repo.metrics.f1 if model_repo.metrics else None,
        model_type=model_repo.config.model_type if model_repo.config else None,
        hyperparameters=model_repo.config.hyperparameters
        if model_repo.config
        else None,
    )


# Описания признаков для документации
_FEATURE_DESCRIPTIONS = {
    "monthly_fee": "Ежемесячная плата за сервис",
    "usage_hours": "Часы использования сервиса",
    "support_requests": "Количество обращений в поддержку",
    "account_age_months": "Возраст аккаунта в месяцах",
    "failed_payments": "Количество неудачных платежей",
    "autopay_enabled": "Включён ли автоплатёж (0/1)",
    "region": "Регион клиента (america, europe, asia, africa)",
    "device_type": "Тип устройства (mobile, desktop, tablet)",
    "payment_method": "Метод оплаты (card, paypal, crypto)",
}


@router.get("/schema")
def model_schema() -> ModelSchemaResponse:
    """
    Возвращает схему признаков модели.

    Полезно для клиентов, чтобы понять, какие признаки ожидает модель
    и какие типы данных нужно передавать в /predict.
    """
    features = []

    # Числовые признаки
    for col in NUMERIC_COLUMNS:
        features.append(
            FeatureInfo(
                name=col,
                type="numeric",
                description=_FEATURE_DESCRIPTIONS.get(col, ""),
            )
        )

    # Категориальные признаки
    for col in CATEGORICAL_COLUMNS:
        features.append(
            FeatureInfo(
                name=col,
                type="categorical",
                description=_FEATURE_DESCRIPTIONS.get(col, ""),
            )
        )

    return ModelSchemaResponse(features=features)
