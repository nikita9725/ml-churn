from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from src.exceptions import ModelNotTrainedError
from src.model.training import ModelMetrics
from src.schemas import TrainingConfigChurn


class ModelRepository:
    def __init__(self, model_path: str | Path) -> None:
        self._model_path = Path(model_path)
        self._pipeline: Pipeline | None = None
        self._metrics: ModelMetrics | None = None
        self._trained_at: datetime | None = None
        self._config: TrainingConfigChurn | None = None

    def save(
        self, pipeline: Pipeline, metrics: ModelMetrics, config: TrainingConfigChurn
    ) -> None:
        self._pipeline = pipeline
        self._metrics = metrics
        self._config = config
        self._trained_at = datetime.now(UTC)
        self._model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            (self._pipeline, self._metrics, self._trained_at, self._config),
            self._model_path,
        )

    def load(self) -> None:
        if not self._model_path.exists():
            return
        loaded = joblib.load(self._model_path)
        # Обратная совместимость: если файл содержит tuple из 3 элементов, используем дефолтный config
        if len(loaded) == 3:
            self._pipeline, self._metrics, self._trained_at = loaded
            self._config = TrainingConfigChurn()  # Дефолтный config
        else:
            self._pipeline, self._metrics, self._trained_at, self._config = loaded

    @property
    def is_trained(self) -> bool:
        return self._pipeline is not None

    @property
    def pipeline(self) -> Pipeline:
        if self._pipeline is None:
            raise ModelNotTrainedError()
        return self._pipeline

    @property
    def metrics(self) -> ModelMetrics | None:
        return self._metrics

    @property
    def trained_at(self) -> datetime | None:
        return self._trained_at

    @property
    def config(self) -> TrainingConfigChurn | None:
        return self._config
