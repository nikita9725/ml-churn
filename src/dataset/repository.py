from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class DatasetInfo:
    row_count: int
    column_count: int
    feature_names: list[str]
    churn_distribution: dict[int, int]


class DatasetRepository:
    def __init__(self, file_path: str | Path) -> None:
        self._file_path = Path(file_path)
        self._df = pd.read_csv(self._file_path)

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
