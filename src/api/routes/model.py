import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Body, Depends, Query

from src.api.dependencies import (
    get_dataset_repo,
    get_loaded_model_repo,
    get_training_history_repo,
)
from src.dataset.preprocessing import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from src.dataset.repository import DatasetRepository
from src.exceptions import EmptyDatasetError, TrainingFailedError
from src.model.history_repository import TrainingHistoryRepository
from src.model.repository import ModelRepository
from src.model.training import TrainingHistoryEntry, train_churn_model
from src.schemas import (
    FeatureInfo,
    ModelMetricsHistoryResponse,
    ModelMetricsResponse,
    ModelSchemaResponse,
    ModelStatusResponse,
    TrainingConfigChurn,
    TrainingHistoryEntryResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/model", tags=["model"])


@router.post("/train")
def model_train(
    repo: DatasetRepository = Depends(get_dataset_repo),
    model_repo: ModelRepository = Depends(get_loaded_model_repo),
    history_repo: TrainingHistoryRepository = Depends(get_training_history_repo),
    config: TrainingConfigChurn = Body(default=TrainingConfigChurn()),
) -> ModelMetricsResponse:
    """
    Запускает обучение модели на данных из churn_dataset.csv.

    Эндпоинт:
    1. Загружает данные из репозитория
    2. Проверяет что данные не пустые
    3. Обучает модель через train_churn_model с учётом конфига
    4. Сохраняет модель на диск вместе с конфигом
    5. Добавляет запись в историю обучений
    6. Возвращает метрики accuracy, f1 и roc_auc на тестовой выборке
    """
    df = repo.df

    if df.empty:
        raise EmptyDatasetError()

    logger.info(
        "Starting model training (type=%s, rows=%d)", config.model_type, len(df)
    )

    try:
        pipeline, metrics, config = train_churn_model(df, config)
        model_repo.save(pipeline, metrics, config)
    except Exception as e:
        logger.error("Model training failed: %s", e)
        raise TrainingFailedError(str(e)) from e

    logger.info(
        "Training complete — accuracy=%.4f, f1=%.4f, roc_auc=%.4f",
        metrics.accuracy,
        metrics.f1,
        metrics.roc_auc,
    )

    history_repo.add(
        TrainingHistoryEntry(
            timestamp=datetime.now(UTC),
            model_type=config.model_type,
            hyperparameters=config.hyperparameters,
            accuracy=metrics.accuracy,
            f1=metrics.f1,
            roc_auc=metrics.roc_auc,
        )
    )

    return ModelMetricsResponse(
        accuracy=metrics.accuracy,
        f1=metrics.f1,
        roc_auc=metrics.roc_auc,
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
        roc_auc=model_repo.metrics.roc_auc if model_repo.metrics else None,
        model_type=model_repo.config.model_type if model_repo.config else None,
        hyperparameters=model_repo.config.hyperparameters
        if model_repo.config
        else None,
    )


@router.get("/metrics")
def model_metrics(
    model_type: str | None = Query(default=None, description="Фильтр по типу модели"),
    limit: int = Query(default=10, ge=1, description="Количество последних записей"),
    history_repo: TrainingHistoryRepository = Depends(get_training_history_repo),
) -> ModelMetricsHistoryResponse:
    """
    Возвращает метрики последнего обучения и историю обучений.

    Можно фильтровать по типу модели (model_type=logreg / random_forest)
    и ограничить количество записей через limit.
    """
    entries = history_repo.get_all(model_type=model_type)
    limited = entries[-limit:]

    latest = limited[-1] if limited else None

    return ModelMetricsHistoryResponse(
        latest=_to_entry_response(latest) if latest else None,
        history=[_to_entry_response(e) for e in limited],
    )


def _to_entry_response(entry: TrainingHistoryEntry) -> TrainingHistoryEntryResponse:
    return TrainingHistoryEntryResponse(
        timestamp=entry.timestamp,
        model_type=entry.model_type,
        hyperparameters=entry.hyperparameters,
        accuracy=entry.accuracy,
        f1=entry.f1,
        roc_auc=entry.roc_auc,
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
