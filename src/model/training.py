from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.dataset.preprocessing import (
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    prepare_data,
    split_data,
)
from src.schemas import TrainingConfigChurn


@dataclass
class ModelMetrics:
    accuracy: float
    f1: float
    roc_auc: float


@dataclass
class TrainingHistoryEntry:
    timestamp: datetime
    model_type: str
    hyperparameters: dict[str, Any]
    accuracy: float
    f1: float
    roc_auc: float


def _build_preprocessor() -> ColumnTransformer:
    """
    Создаём ColumnTransformer — он применяет разные трансформации к разным колонкам.

    Для числовых признаков (NUMERIC_COLUMNS):
    - StandardScaler приводит признаки к одному масштабу (среднее=0, стандартное отклонение=1).
      Это важно для LogisticRegression, потому что градиентный спуск работает лучше,
      когда все признаки в одном масштабе.

    Для категориальных признаков (CATEGORICAL_COLUMNS):
    - OneHotEncoder преобразует категории в бинарные колонки (0/1).
      Например, region="america" станет [1, 0, 0], region="europe" → [0, 1, 0].
      set_output(transform="pandas") возвращает DataFrame с понятными именами колонок.
    """
    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(handle_unknown="error", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_COLUMNS),
            ("cat", categorical_transformer, CATEGORICAL_COLUMNS),
        ]
    )
    preprocessor.set_output(transform="pandas")

    return preprocessor


def _build_pipeline(config: TrainingConfigChurn) -> Pipeline:
    """
    Создаём Pipeline — он объединяет предобработку и модель в единую цепочку.

    Pipeline гарантирует, что:
    1. При обучении: данные проходят через предобработку, затем через модель
    2. При предсказании: новые данные проходят через ТОТ ЖЕ предобработчик,
       что исключает ошибку "забыли масштабировать тестовые данные"

    Args:
        config: Конфигурация обучения с типом модели и гиперпараметрами
    """
    preprocessor = _build_preprocessor()

    # Выбираем модель в зависимости от model_type
    if config.model_type == "logreg":
        # Используем hardcoded дефолты для обратной совместимости
        default_params = {"random_state": 42, "max_iter": 1000}
        # Объединяем дефолты с переданными гиперпараметрами
        params = {**default_params, **config.hyperparameters}
        classifier = LogisticRegression(**params)
    elif config.model_type == "random_forest":
        # Используем hardcoded дефолты
        default_params = {"random_state": 42, "n_estimators": 100}
        # Объединяем дефолты с переданными гиперпараметрами
        params = {**default_params, **config.hyperparameters}
        classifier = RandomForestClassifier(**params)
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return pipeline


def train_churn_model(
    df: pd.DataFrame, config: TrainingConfigChurn | None = None
) -> tuple[Pipeline, ModelMetrics, TrainingConfigChurn]:
    """
    Обучает модель прогнозирования churn.

    Args:
        df: DataFrame с признаками и целевой переменной churn
        config: Конфигурация обучения. Если None, используется дефолтный config.

    Returns:
        tuple[Pipeline, ModelMetrics, TrainingConfigChurn]: обученный pipeline, метрики и конфиг

    Как это работает:
    1. prepare_data — заполняет пропуски, разделяет X (признаки) и y (целевая переменная)
    2. split_data — разделяет данные на тренировочную и тестовую выборки (80/20)
       Стратификация гарантирует, что пропорция классов (churn/no churn) одинакова в обеих выборках
    3. Pipeline обучается на тренировочных данных
    4. Метрики вычисляются на тестовых данных — это честная оценка,
       потому что модель не видела эти данные во время обучения
    """
    # Если config не передан, используем дефолтный
    if config is None:
        config = TrainingConfigChurn()

    prepared = prepare_data(df)
    split = split_data(prepared)

    pipeline = _build_pipeline(config)
    pipeline.fit(split.X_train, split.y_train)

    y_pred = pipeline.predict(split.X_test)
    y_prob = pipeline.predict_proba(split.X_test)[:, 1]

    metrics = ModelMetrics(
        accuracy=accuracy_score(split.y_test, y_pred),
        f1=f1_score(split.y_test, y_pred),
        roc_auc=roc_auc_score(split.y_test, y_prob),
    )

    return pipeline, metrics, config
