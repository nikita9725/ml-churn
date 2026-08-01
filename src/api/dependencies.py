from pathlib import Path

from fastapi import Request

from src.dataset.repository import DatasetRepository
from src.model.repository import ModelRepository

_dataset_path = (
    Path(__file__).resolve().parent.parent.parent / "data" / "churn_dataset.csv"
)
_model_path = (
    Path(__file__).resolve().parent.parent.parent / "models" / "churn_model.joblib"
)


def get_dataset_repo() -> DatasetRepository:
    return DatasetRepository(_dataset_path)


def get_loaded_model_repo(request: Request) -> ModelRepository:
    return request.app.state.model_repo
