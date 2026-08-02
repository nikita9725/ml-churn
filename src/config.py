from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = _PROJECT_ROOT / "data" / "churn_dataset.csv"
MODEL_PATH = _PROJECT_ROOT / "models" / "churn_model.joblib"
HISTORY_PATH = _PROJECT_ROOT / "models" / "training_history.json"
