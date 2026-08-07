from collections.abc import Callable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.dataset.repository import DatasetRepository
from src.exceptions import ErrorCode
from src.model.repository import ModelRepository

REAL_DATA_PATH = Path(__file__).parent.parent / "data" / "churn_dataset.csv"

SAMPLE_FEATURES = {
    "monthly_fee": 65.5,
    "usage_hours": 120.0,
    "support_requests": 3,
    "account_age_months": 24,
    "failed_payments": 1,
    "region": "europe",
    "device_type": "mobile",
    "payment_method": "card",
    "autopay_enabled": 1,
}


def test_predict_returns_400_when_model_not_trained(client: TestClient) -> None:
    response = client.post("/predict", json=SAMPLE_FEATURES)
    assert response.status_code == 400
    data = response.json()
    assert data["code"] == ErrorCode.MODEL_NOT_TRAINED.value
    assert data["message"] == "Model has not been trained yet"


def test_predict_validates_types(client: TestClient) -> None:
    payload = {**SAMPLE_FEATURES, "monthly_fee": "not_a_number"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_missing_field(client: TestClient) -> None:
    payload = {k: v for k, v in SAMPLE_FEATURES.items() if k != "region"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


@pytest.mark.parametrize(
    "field, invalid_value",
    [
        ("region", "mars"),
        ("device_type", "watch"),
        ("payment_method", "credit_card"),
        ("autopay_enabled", 2),
    ],
)
def test_predict_rejects_invalid_domain_values(
    client: TestClient, field: str, invalid_value: object
) -> None:
    payload = {**SAMPLE_FEATURES, field: invalid_value}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["code"] == ErrorCode.VALIDATION_ERROR.value
    assert field in data["details"]["error"]


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


def test_model_schema_endpoint(client: TestClient) -> None:
    response = client.get("/model/schema")
    assert response.status_code == 200
    data = response.json()
    assert "features" in data
    features = data["features"]
    assert len(features) == 9  # 6 numeric + 3 categorical

    # Проверяем, что все признаки имеют name, type и description
    for feature in features:
        assert "name" in feature
        assert "type" in feature
        assert "description" in feature
        assert feature["type"] in ["numeric", "categorical"]

    # Проверяем, что есть числовые и категориальные признаки
    numeric_features = [f for f in features if f["type"] == "numeric"]
    categorical_features = [f for f in features if f["type"] == "categorical"]
    assert len(numeric_features) == 6
    assert len(categorical_features) == 3
