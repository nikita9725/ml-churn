from src.schemas import DatasetRowChurn, FeatureVectorChurn


def test_feature_vector_churn_model() -> None:
    obj = FeatureVectorChurn(
        monthly_fee=65.5,
        usage_hours=120.0,
        support_requests=3,
        account_age_months=24,
        failed_payments=1,
        region="EU",
        device_type="mobile",
        payment_method="credit_card",
        autopay_enabled=1,
    )
    assert obj.monthly_fee == 65.5
    assert obj.region == "EU"


def test_dataset_row_churn_model() -> None:
    obj = DatasetRowChurn(
        monthly_fee=65.5,
        usage_hours=120.0,
        support_requests=3,
        account_age_months=24,
        failed_payments=1,
        region="EU",
        device_type="mobile",
        payment_method="credit_card",
        autopay_enabled=1,
        churn=1,
    )
    assert obj.churn == 1
    assert obj.monthly_fee == 65.5
