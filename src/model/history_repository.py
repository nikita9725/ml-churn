from __future__ import annotations

import json
from abc import ABC
from datetime import datetime
from pathlib import Path

from src.model.training import TrainingHistoryEntry


class TrainingHistoryRepository(ABC):
    def __init__(self) -> None:
        self._entries: list[TrainingHistoryEntry] = []

    def add(self, entry: TrainingHistoryEntry) -> None:
        self._entries.append(entry)

    def get_all(self, model_type: str | None = None) -> list[TrainingHistoryEntry]:
        if model_type is None:
            return list(self._entries)
        return [e for e in self._entries if e.model_type == model_type]

    def get_last(self, n: int = 10) -> list[TrainingHistoryEntry]:
        return self._entries[-n:] if self._entries else []

    def load(self) -> None:
        pass

    def save(self) -> None:
        pass


class InMemoryTrainingHistoryRepository(TrainingHistoryRepository):
    pass


class JsonTrainingHistoryRepository(TrainingHistoryRepository):
    def __init__(self, file_path: str | Path) -> None:
        super().__init__()
        self._file_path = Path(file_path)

    def add(self, entry: TrainingHistoryEntry) -> None:
        super().add(entry)
        self.save()

    def load(self) -> None:
        if not self._file_path.exists():
            return
        data = json.loads(self._file_path.read_text())
        self._entries = [
            TrainingHistoryEntry(
                timestamp=datetime.fromisoformat(e["timestamp"]),
                model_type=e["model_type"],
                hyperparameters=e["hyperparameters"],
                accuracy=e["accuracy"],
                f1=e["f1"],
                roc_auc=e["roc_auc"],
            )
            for e in data
        ]

    def save(self) -> None:
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "timestamp": e.timestamp.isoformat(),
                "model_type": e.model_type,
                "hyperparameters": e.hyperparameters,
                "accuracy": e.accuracy,
                "f1": e.f1,
                "roc_auc": e.roc_auc,
            }
            for e in self._entries
        ]
        self._file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
