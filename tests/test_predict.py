from collections.abc import Callable
from pathlib import Path

from fastapi.testclient import TestClient

from src.dataset.repository import DatasetRepository
from src.model.repository import ModelRepository

REAL_DATA_PATH = Path(__file__).parent.parent / "data" / "churn_dataset.csv"

SAMPLE_FEATURES = {
    "monthly_fee": 65.5,
    "usage_hours": 120.0,
    "support_requests": 3,
    "account_age_months": 24,
    "failed_payments": 1,
    "region": "EU",
    "device_type": "mobile",
    "payment_method": "credit_card",
    "autopay_enabled": 1,
}


def test_predict_returns_400_when_model_not_trained(client: TestClient) -> None:
    response = client.post("/predict", json=SAMPLE_FEATURES)
    assert response.status_code == 400
    assert "Model has not been trained yet" in response.json()["detail"]


def test_predict_validates_types(client: TestClient) -> None:
    payload = {**SAMPLE_FEATURES, "monthly_fee": "not_a_number"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_missing_field(client: TestClient) -> None:
    payload = {k: v for k, v in SAMPLE_FEATURES.items() if k != "region"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_single(
    client: TestClient,
    override_repo: Callable[[DatasetRepository | object], None],
    override_model_repo: Callable[[ModelRepository | object], None],
    model_path: Path,
) -> None:
    override_repo(DatasetRepository(REAL_DATA_PATH))
    override_model_repo(ModelRepository(model_path))

    client.post("/model/train")

    response = client.post("/predict", json=SAMPLE_FEATURES)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["probability_churn_0"] <= 1.0
    assert 0.0 <= data["probability_churn_1"] <= 1.0
    assert abs(data["probability_churn_0"] + data["probability_churn_1"] - 1.0) < 1e-6
    assert data["features"]["monthly_fee"] == 65.5


def test_predict_list(
    client: TestClient,
    override_repo: Callable[[DatasetRepository | object], None],
    override_model_repo: Callable[[ModelRepository | object], None],
    model_path: Path,
) -> None:
    override_repo(DatasetRepository(REAL_DATA_PATH))
    override_model_repo(ModelRepository(model_path))

    client.post("/model/train")

    features_list = [SAMPLE_FEATURES, SAMPLE_FEATURES]
    response = client.post("/predict", json=features_list)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2
    for item in data:
        assert item["prediction"] in [0, 1]
        assert 0.0 <= item["probability_churn_0"] <= 1.0
        assert 0.0 <= item["probability_churn_1"] <= 1.0
