import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.exceptions import (
    DatasetEmptyFileError,
    DatasetInvalidStructureError,
    DatasetNotFoundError,
)

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "churn",
    "monthly_fee",
    "usage_hours",
    "support_requests",
    "account_age_months",
    "failed_payments",
    "region",
    "device_type",
    "payment_method",
    "autopay_enabled",
]


@dataclass
class DatasetInfo:
    row_count: int
    column_count: int
    feature_names: list[str]
    churn_distribution: dict[int, int]


class DatasetRepository:
    def __init__(self, file_path: str | Path) -> None:
        self._file_path = Path(file_path)
        logger.info("Loading dataset from %s", self._file_path)

        if not self._file_path.exists():
            raise DatasetNotFoundError(str(self._file_path))

        try:
            self._df = pd.read_csv(self._file_path)
        except pd.errors.EmptyDataError:
            raise DatasetEmptyFileError(str(self._file_path))

        missing_columns = [
            col for col in REQUIRED_COLUMNS if col not in self._df.columns
        ]
        if missing_columns:
            raise DatasetInvalidStructureError(missing_columns)

        logger.info(
            "Dataset loaded: %d rows, %d columns", len(self._df), len(self._df.columns)
        )

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    def get_preview(self, n: int = 10) -> list[dict]:
        return self._df.head(n).to_dict(orient="records")

    def get_info(self) -> DatasetInfo:
        feature_names = [c for c in self._df.columns if c != "churn"]
        raw_counts = self._df["churn"].value_counts()
        churn_distribution = {int(k): int(raw_counts[k]) for k in raw_counts.index}
        return DatasetInfo(
            row_count=len(self._df),
            column_count=len(self._df.columns),
            feature_names=feature_names,
            churn_distribution=churn_distribution,
        )
