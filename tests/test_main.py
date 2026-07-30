from fastapi.testclient import TestClient

from src.main import DatasetRowChurn, FeatureVectorChurn, app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "ml churn service is running"}


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


def test_predict_returns_input():
    response = client.post("/predict", json=SAMPLE_FEATURES)
    assert response.status_code == 200
    assert response.json() == SAMPLE_FEATURES


def test_predict_validates_types():
    payload = {**SAMPLE_FEATURES, "monthly_fee": "not_a_number"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_missing_field():
    payload = {k: v for k, v in SAMPLE_FEATURES.items() if k != "region"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_feature_vector_churn_model():
    obj = FeatureVectorChurn(**SAMPLE_FEATURES)
    assert obj.monthly_fee == 65.5
    assert obj.region == "EU"


def test_dataset_row_churn_model():
    row = DatasetRowChurn(**SAMPLE_FEATURES, churn=1)
    assert row.churn == 1
    assert row.monthly_fee == 65.5
