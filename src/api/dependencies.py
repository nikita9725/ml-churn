from fastapi import Request

from src.config import DATASET_PATH
from src.dataset.repository import DatasetRepository
from src.model.history_repository import TrainingHistoryRepository
from src.model.repository import ModelRepository


def get_dataset_repo() -> DatasetRepository:
    return DatasetRepository(DATASET_PATH)


def get_loaded_model_repo(request: Request) -> ModelRepository:
    return request.app.state.model_repo


def get_training_history_repo(request: Request) -> TrainingHistoryRepository:
    return request.app.state.training_history_repo
