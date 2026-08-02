from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.routes.dataset import router as dataset_router
from src.api.routes.model import router as model_router
from src.api.routes.predict import router as predict_router
from src.exceptions import (
    ChurnServiceError,
    ErrorCode,
    ErrorDetails,
    ErrorResponse,
)
from src.model.history_repository import JsonTrainingHistoryRepository
from src.model.repository import ModelRepository
from src.schemas import HealthResponse

_model_path = Path(__file__).resolve().parent.parent / "models" / "churn_model.joblib"
_history_path = (
    Path(__file__).resolve().parent.parent / "models" / "training_history.json"
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    repo = ModelRepository(_model_path)
    repo.load()
    app.state.model_repo = repo

    history_repo = JsonTrainingHistoryRepository(_history_path)
    history_repo.load()
    app.state.training_history_repo = history_repo

    yield


app = FastAPI(lifespan=lifespan)


@app.exception_handler(ChurnServiceError)
async def churn_service_error_handler(
    request: Request, exc: ChurnServiceError
) -> JSONResponse:
    """Обработчик кастомных исключений сервиса."""
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            code=exc.code,
            message=exc.message,
            details=exc.details,
        ).model_dump(mode="json"),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Обработчик HTTP исключений."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            code=ErrorCode.HTTP_ERROR,
            message=str(exc.detail),
            details=None,
        ).model_dump(mode="json"),
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Обработчик ошибок валидации Pydantic."""
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            code=ErrorCode.VALIDATION_ERROR,
            message="Invalid request data",
            details=ErrorDetails(error=str(exc.errors())),
        ).model_dump(mode="json"),
    )


app.include_router(dataset_router)
app.include_router(model_router)
app.include_router(predict_router)


@app.get("/")
def root() -> HealthResponse:
    return HealthResponse(message="ml churn service is running")
