from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_dataset_repo, get_loaded_model_repo
from src.dataset.repository import DatasetRepository
from src.model.repository import ModelRepository
from src.model.training import train_churn_model
from src.schemas import ModelMetricsResponse, ModelStatusResponse

router = APIRouter(prefix="/model", tags=["model"])


@router.post("/train")
def model_train(
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


@router.get("/status")
def model_status(
    model_repo: ModelRepository = Depends(get_loaded_model_repo),
) -> ModelStatusResponse:
    return ModelStatusResponse(
        is_trained=model_repo.is_trained,
        trained_at=model_repo.trained_at,
        accuracy=model_repo.metrics.accuracy if model_repo.metrics else None,
        f1=model_repo.metrics.f1 if model_repo.metrics else None,
    )
