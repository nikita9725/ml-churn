from fastapi import APIRouter, Depends

from src.api.dependencies import get_dataset_repo
from src.dataset.preprocessing import check_class_distribution, prepare_data, split_data
from src.dataset.repository import DatasetRepository
from src.schemas import (
    ClassDistributionResponse,
    DatasetInfoResponse,
    DatasetRowChurn,
    SplitInfoResponse,
)

router = APIRouter(prefix="/dataset", tags=["dataset"])


@router.get("/preview")
def dataset_preview(
    n: int = 10, repo: DatasetRepository = Depends(get_dataset_repo)
) -> list[DatasetRowChurn]:
    rows = repo.get_preview(n)
    return [DatasetRowChurn(**row) for row in rows]


@router.get("/info")
def dataset_info(
    repo: DatasetRepository = Depends(get_dataset_repo),
) -> DatasetInfoResponse:
    info = repo.get_info()
    return DatasetInfoResponse(
        row_count=info.row_count,
        column_count=info.column_count,
        feature_names=info.feature_names,
        churn_distribution={str(k): v for k, v in info.churn_distribution.items()},
    )


@router.get("/split-info")
def dataset_split_info(
    repo: DatasetRepository = Depends(get_dataset_repo),
) -> SplitInfoResponse:
    df = repo.df
    prepared = prepare_data(df)
    split = split_data(prepared)
    distribution = check_class_distribution(split.y_train, split.y_test)

    train_counts = split.y_train.value_counts()
    test_counts = split.y_test.value_counts()

    return SplitInfoResponse(
        train_size=len(split.X_train),
        test_size=len(split.X_test),
        train_distribution=ClassDistributionResponse(
            count_0=int(train_counts.get(0, 0)),
            count_1=int(train_counts.get(1, 0)),
            ratio_0=distribution.train.get(0, 0.0),
            ratio_1=distribution.train.get(1, 0.0),
        ),
        test_distribution=ClassDistributionResponse(
            count_0=int(test_counts.get(0, 0)),
            count_1=int(test_counts.get(1, 0)),
            ratio_0=distribution.test.get(0, 0.0),
            ratio_1=distribution.test.get(1, 0.0),
        ),
    )
