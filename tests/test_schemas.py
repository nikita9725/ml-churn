import pytest
from pydantic import ValidationError

from src.schemas import DatasetRowChurn, FeatureVectorChurn


def test_feature_vector_churn_model() -> None:
    obj = FeatureVectorChurn(
        monthly_fee=65.5,
        usage_hours=120.0,
        support_requests=3,
        account_age_months=24,
        failed_payments=1,
        region="europe",
        device_type="mobile",
        payment_method="card",
        autopay_enabled=1,
    )
    assert obj.monthly_fee == 65.5
    assert obj.region == "europe"


def test_dataset_row_churn_model() -> None:
    obj = DatasetRowChurn(
        monthly_fee=65.5,
        usage_hours=120.0,
        support_requests=3,
        account_age_months=24,
        failed_payments=1,
        region="europe",
        device_type="mobile",
        payment_method="card",
        autopay_enabled=1,
        churn=1,
    )
    assert obj.churn == 1
    assert obj.monthly_fee == 65.5


VALID_FEATURES = {
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


@pytest.mark.parametrize(
    "field, invalid_value",
    [
        ("region", "mars"),
        ("device_type", "watch"),
        ("payment_method", "credit_card"),
        ("autopay_enabled", 2),
    ],
)
def test_feature_vector_rejects_invalid_domain_values(
    field: str, invalid_value: object
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        FeatureVectorChurn(**{**VALID_FEATURES, field: invalid_value})
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == (field,)
    assert errors[0]["input"] == invalid_value


def test_dataset_row_rejects_invalid_churn() -> None:
    with pytest.raises(ValidationError) as exc_info:
        DatasetRowChurn(**VALID_FEATURES, churn=3)
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("churn",)
    assert errors[0]["input"] == 3
