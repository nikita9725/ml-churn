from pathlib import Path

from fastapi import Request

from src.dataset.repository import DatasetRepository
from src.model.history_repository import TrainingHistoryRepository
from src.model.repository import ModelRepository

_dataset_path = (
    Path(__file__).resolve().parent.parent.parent / "data" / "churn_dataset.csv"
)
_model_path = (
    Path(__file__).resolve().parent.parent.parent / "models" / "churn_model.joblib"
)
_history_path = (
    Path(__file__).resolve().parent.parent.parent / "models" / "training_history.json"
)


def get_dataset_repo() -> DatasetRepository:
    return DatasetRepository(_dataset_path)


def get_loaded_model_repo(request: Request) -> ModelRepository:
    return request.app.state.model_repo


def get_training_history_repo(request: Request) -> TrainingHistoryRepository:
    return request.app.state.training_history_repo
