from enum import StrEnum

from pydantic import BaseModel, Field


class ErrorCode(StrEnum):
    """Коды ошибок сервиса."""

    EMPTY_DATASET = "EMPTY_DATASET"
    DATASET_NOT_FOUND = "DATASET_NOT_FOUND"
    DATASET_EMPTY_FILE = "DATASET_EMPTY_FILE"
    DATASET_INVALID_STRUCTURE = "DATASET_INVALID_STRUCTURE"
    MODEL_NOT_TRAINED = "MODEL_NOT_TRAINED"
    TRAINING_FAILED = "TRAINING_FAILED"
    PREDICTION_FAILED = "PREDICTION_FAILED"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    HTTP_ERROR = "HTTP_ERROR"


class ErrorDetails(BaseModel):
    """Детали ошибки."""

    error: str | None = Field(default=None, description="Описание ошибки")
    field: str | None = Field(default=None, description="Поле, вызвавшее ошибку")
    expected: str | None = Field(default=None, description="Ожидаемое значение")
    received: str | None = Field(default=None, description="Полученное значение")


class ErrorResponse(BaseModel):
    """Стандартный формат ответа с ошибкой."""

    code: ErrorCode = Field(description="Код ошибки")
    message: str = Field(description="Человекочитаемое сообщение об ошибке")
    details: ErrorDetails | None = Field(
        default=None, description="Дополнительные детали"
    )


class ChurnServiceError(Exception):
    """Базовое исключение для сервиса."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        details: ErrorDetails | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.details = details
        super().__init__(message)


class EmptyDatasetError(ChurnServiceError):
    """Датасет пуст."""

    def __init__(self) -> None:
        super().__init__(ErrorCode.EMPTY_DATASET, "Dataset is empty")


class DatasetNotFoundError(ChurnServiceError):
    """Файл датасета не найден."""

    def __init__(self, file_path: str) -> None:
        super().__init__(
            ErrorCode.DATASET_NOT_FOUND,
            f"Dataset file not found: {file_path}",
            ErrorDetails(expected="Existing file path", received=file_path),
        )


class DatasetEmptyFileError(ChurnServiceError):
    """Файл датасета пустой."""

    def __init__(self, file_path: str) -> None:
        super().__init__(
            ErrorCode.DATASET_EMPTY_FILE,
            f"Dataset file is empty: {file_path}",
            ErrorDetails(expected="Non-empty CSV file", received=file_path),
        )


class DatasetInvalidStructureError(ChurnServiceError):
    """Датасет имеет неверную структуру (отсутствуют обязательные колонки)."""

    def __init__(self, missing_columns: list[str]) -> None:
        super().__init__(
            ErrorCode.DATASET_INVALID_STRUCTURE,
            f"Dataset is missing required columns: {', '.join(missing_columns)}",
            ErrorDetails(
                expected="Columns: churn, monthly_fee, usage_hours, support_requests, account_age_months, failed_payments, region, device_type, payment_method, autopay_enabled",
                received=f"Missing: {', '.join(missing_columns)}",
            ),
        )


class ModelNotTrainedError(ChurnServiceError):
    """Модель не обучена."""

    def __init__(self) -> None:
        super().__init__(ErrorCode.MODEL_NOT_TRAINED, "Model has not been trained yet")


class TrainingFailedError(ChurnServiceError):
    """Ошибка при обучении модели."""

    def __init__(self, error: str) -> None:
        super().__init__(
            ErrorCode.TRAINING_FAILED,
            "Model training failed",
            ErrorDetails(error=error),
        )


class PredictionFailedError(ChurnServiceError):
    """Ошибка при предсказании."""

    def __init__(self, error: str) -> None:
        super().__init__(
            ErrorCode.PREDICTION_FAILED,
            "Prediction failed",
            ErrorDetails(error=error),
        )
