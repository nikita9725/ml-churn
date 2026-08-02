import logging

import pandas as pd
from fastapi import APIRouter, Depends

from src.api.dependencies import get_loaded_model_repo
from src.dataset.preprocessing import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS
from src.exceptions import PredictionFailedError
from src.model.repository import ModelRepository
from src.schemas import FeatureVectorChurn, PredictionResponseChurn

logger = logging.getLogger(__name__)

router = APIRouter(tags=["predict"])

# Все признаки в порядке, который использовался при обучении
ALL_FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS


@router.post("/predict")
def predict(
    features: FeatureVectorChurn | list[FeatureVectorChurn],
    model_repo: ModelRepository = Depends(get_loaded_model_repo),
) -> PredictionResponseChurn | list[PredictionResponseChurn]:
    """
    Предсказывает churn для нового клиента или списка клиентов.

    Принимает признаки клиента и возвращает:
    - Предсказанный класс (0 или 1)
    - Вероятности для обоих классов
    - Входные признаки (для удобства)

    Если модель не обучена, возвращает ошибку 400.

    Пример запроса:
    ```json
    {
        "monthly_fee": 85.0,
        "usage_hours": 150.0,
        "support_requests": 3,
        "account_age_months": 12,
        "failed_payments": 2,
        "region": "europe",
        "device_type": "mobile",
        "payment_method": "credit_card",
        "autopay_enabled": 0
    }
    ```

    Пример ответа:
    ```json
    {
        "prediction": 1,
        "probability_churn_0": 0.3,
        "probability_churn_1": 0.7,
        "features": {...}
    }
    ```
    """
    pipeline = model_repo.pipeline

    if isinstance(features, FeatureVectorChurn):
        features_list = [features]
        is_single = True
    else:
        features_list = features
        is_single = False

    logger.info("Prediction request for %d client(s)", len(features_list))

    df = pd.DataFrame(
        [f.model_dump() for f in features_list], columns=ALL_FEATURE_COLUMNS
    )

    try:
        predictions = pipeline.predict(df)
        probabilities = pipeline.predict_proba(df)
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise PredictionFailedError(str(e)) from e

    results = [
        PredictionResponseChurn(
            prediction=int(predictions[i]),
            probability_churn_0=float(probabilities[i][0]),
            probability_churn_1=float(probabilities[i][1]),
            features=features_list[i],
        )
        for i in range(len(features_list))
    ]

    return results[0] if is_single else results
