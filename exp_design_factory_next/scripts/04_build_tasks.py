from __future__ import annotations
import argparse
import json
from pathlib import Path


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False)
    _ = parser.parse_args()

    chunks_path = Path("data/processed/chunks/chunks.jsonl")
    out = Path("data/processed/tasks/tasks.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    chunks = list(load_jsonl(chunks_path))
    by_paper: dict[str, list[dict]] = {}
    for row in chunks:
        by_paper.setdefault(row["paper_id"], []).append(row)

    with out.open("w", encoding="utf-8") as f:
        for paper_id, rows in by_paper.items():
            evidence_ids = [r["chunk_id"] for r in rows[:4]]
            tasks = [
                {
                    "task_id": f"task_{paper_id}_01",
                    "paper_id": paper_id,
                    "domain": "demo",
                    "task_type": "experiment_design",
                    "goal": "Design one feasible evidence-grounded experiment based on the paper.",
                    "available_resources": {
                        "equipment": [],
                        "materials": [],
                        "constraints": ["Use only evidence-supported resources."]
                    },
                    "evidence_chunk_ids": evidence_ids,
                    "evidence_summary": "Placeholder evidence summary."
                }
            ]
            for task in tasks:
                f.write(json.dumps(task, ensure_ascii=False) + "\n")
    print(f"Saved tasks to {out}")


if __name__ == "__main__":
    main()
