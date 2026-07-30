from dataclasses import dataclass
from typing import NamedTuple

import pandas as pd
from sklearn.model_selection import train_test_split  # type: ignore[import-untyped]

TARGET_COLUMN = "churn"


class PreparedData(NamedTuple):
    X: pd.DataFrame
    y: pd.Series


class SplitData(NamedTuple):
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


@dataclass
class ClassDistribution:
    train: dict[int, float]
    test: dict[int, float]


NUMERIC_COLUMNS = [
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
    "autopay_enabled",
]

CATEGORICAL_COLUMNS = [
    "region",
    "device_type",
    "payment_method",
]


def prepare_data(df: pd.DataFrame) -> PreparedData:
    df = df.copy()

    numeric_cols = [c for c in NUMERIC_COLUMNS if c in df.columns]
    categorical_cols = [c for c in CATEGORICAL_COLUMNS if c in df.columns]

    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return PreparedData(X=X, y=y)


def split_data(
    data: PreparedData,
    test_size: float = 0.2,
    random_state: int = 42,
) -> SplitData:
    X_train, X_test, y_train, y_test = train_test_split(
        data.X, data.y, test_size=test_size, stratify=data.y, random_state=random_state
    )
    return SplitData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


def check_class_distribution(
    y_train: pd.Series, y_test: pd.Series, tolerance: float = 0.01
) -> ClassDistribution:
    train_dist = y_train.value_counts(normalize=True).to_dict()
    test_dist = y_test.value_counts(normalize=True).to_dict()

    for label in train_dist:
        diff = abs(train_dist[label] - test_dist.get(label, 0.0))
        if diff > tolerance:
            raise ValueError(
                f"Class {label} distribution differs by {diff:.2%} "
                f"between train ({train_dist[label]:.2%}) and test ({test_dist.get(label, 0.0):.2%})"
            )

    return ClassDistribution(
        train={int(k): float(v) for k, v in train_dist.items() if isinstance(k, int)},
        test={int(k): float(v) for k, v in test_dist.items() if isinstance(k, int)},
    )
