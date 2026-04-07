from __future__ import annotations
import json
from pathlib import Path


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    tasks_path = Path("data/processed/tasks/tasks.jsonl")
    out = Path("data/processed/candidates/weak_rag_candidates.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for task in load_jsonl(tasks_path):
            row = {
                "candidate_id": f"weak_{task['task_id']}_01",
                "task_id": task["task_id"],
                "generator_model": "weak_rag_baseline",
                "candidate_rank": 1,
                "reasoning_trace": {"note": "weak baseline"},
                "final_proposal": {
                    "goal": task["goal"],
                    "hypothesis": "Weak placeholder hypothesis",
                    "resources_used": [],
                    "independent_variables": [],
                    "dependent_variables": [],
                    "controls": [],
                    "design": {"conditions": [], "replicates": None, "procedure_outline": []},
                    "measurement_plan": [],
                    "analysis_plan": [],
                    "feasibility_checks": [],
                    "evidence_used": []
                },
                "raw_text": "weak rag placeholder"
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved weak baseline candidates to {out}")


if __name__ == "__main__":
    main()
