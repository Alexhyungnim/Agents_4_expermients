from __future__ import annotations
from pathlib import Path

DIRS = [
    "data/raw/metadata",
    "data/raw/pdf",
    "data/raw/parse_cache",
    "data/processed/papers",
    "data/processed/chunks",
    "data/processed/tasks",
    "data/processed/candidates",
    "data/processed/judged",
    "data/processed/datasets/sft",
    "data/processed/datasets/dpo",
    "data/processed/datasets/validator",
    "reports/data_stats",
    "reports/judge_stats",
    "reports/experiments",
    "training/checkpoints",
    "training/logs",
]

for d in DIRS:
    Path(d).mkdir(parents=True, exist_ok=True)
print("Project directories are ready.")
