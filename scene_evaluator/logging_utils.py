from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scene_evaluator.config import PROJECT_ROOT


EXPERIMENTS_PATH = PROJECT_ROOT / "logs" / "experiments.jsonl"
GENERATIONS_PATH = PROJECT_ROOT / "logs" / "generations.jsonl"


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def log_experiment(record: dict[str, Any]) -> None:
    append_jsonl(EXPERIMENTS_PATH, record)


def log_generation(record: dict[str, Any]) -> None:
    append_jsonl(GENERATIONS_PATH, record)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

