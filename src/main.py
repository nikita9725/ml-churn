from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.routes.dataset import router as dataset_router
from src.api.routes.model import router as model_router
from src.api.routes.predict import router as predict_router
from src.model.repository import ModelNotTrainedError, ModelRepository
from src.schemas import HealthResponse

_model_path = Path(__file__).resolve().parent.parent / "models" / "churn_model.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    repo = ModelRepository(_model_path)
    repo.load()
    app.state.model_repo = repo
    yield


app = FastAPI(lifespan=lifespan)


@app.exception_handler(ModelNotTrainedError)
async def model_not_trained_handler(
    request: Request, exc: ModelNotTrainedError
) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


app.include_router(dataset_router)
app.include_router(model_router)
app.include_router(predict_router)


@app.get("/")
def root() -> HealthResponse:
    return HealthResponse(message="ml churn service is running")
