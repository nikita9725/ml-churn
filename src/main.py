from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from src.dataset.repository import DatasetRepository

app = FastAPI()

_dataset_path = Path(__file__).resolve().parent.parent / "data" / "churn_dataset.csv"
dataset_repo = DatasetRepository(_dataset_path)


class FeatureVectorChurn(BaseModel):
    monthly_fee: float
    usage_hours: float
    support_requests: int
    account_age_months: int
    failed_payments: int
    region: str
    device_type: str
    payment_method: str
    autopay_enabled: int


class DatasetRowChurn(FeatureVectorChurn):
    churn: int


@app.get("/")
async def root():
    return {"message": "ml churn service is running"}


@app.post("/predict", response_model=FeatureVectorChurn)
async def predict(features: FeatureVectorChurn):
    return features


@app.get("/dataset/preview")
async def dataset_preview(n: int = 10):
    return dataset_repo.get_preview(n)


@app.get("/dataset/info")
async def dataset_info():
    return dataset_repo.get_info()
