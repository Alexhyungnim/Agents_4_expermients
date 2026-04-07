from __future__ import annotations
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    _ = parser.parse_args()

    out = Path("data/processed/chunks/chunks.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "paper_id": "demo_000001",
            "chunk_id": "demo_000001_c001",
            "section": "methods",
            "chunk_start": 0,
            "text": "This is a placeholder methods chunk mentioning equipment, process parameters, and measurements.",
            "matched_groups": ["methods_section", "equipment", "process", "outputs"],
            "group_hit_count": 4,
            "total_hits": 8,
        }
    ]

    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved placeholder chunks to {out}")
