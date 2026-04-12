from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def strong_llm_call(prompt: str, n: int = 4) -> list[dict[str, Any]]:
    # Replace this stub with your model provider call.
    result = []
    for i in range(n):
        result.append(
            {
                "reasoning_trace": {
                    "resource_check": "Placeholder",
                    "variable_mapping": "Placeholder",
                    "control_strategy": "Placeholder",
                    "measurement_strategy": "Placeholder",
                    "risk_check": "Placeholder",
                },
                "final_proposal": {
                    "goal": "Placeholder goal",
                    "hypothesis": "Placeholder hypothesis",
                    "resources_used": ["Placeholder equipment"],
                    "independent_variables": ["var1"],
                    "dependent_variables": ["response1"],
                    "controls": ["control1"],
                    "design": {
                        "conditions": ["cond1", "cond2"],
                        "replicates": 3,
                        "procedure_outline": ["step1", "step2"]
                    },
                    "measurement_plan": ["measure1"],
                    "analysis_plan": ["analysis1"],
                    "feasibility_checks": ["check1"],
                    "evidence_used": []
                },
                "raw_text": prompt[:200]
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=False)
    parser.add_argument("--model", required=False)
    args = parser.parse_args()

    tasks_path = Path("data/processed/tasks/tasks.jsonl")
    out = Path("data/processed/candidates/candidates.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for task in load_jsonl(tasks_path):
            prompt = json.dumps(task, ensure_ascii=False, indent=2)
            cands = strong_llm_call(prompt, n=4)
            for i, cand in enumerate(cands, start=1):
                row = {
                    "candidate_id": f"cand_{task['task_id']}_{i:02d}",
                    "task_id": task["task_id"],
                    "generator_model": "strong_generator_A",
                    "candidate_rank": i,
                    **cand,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved candidates to {out}")
    print(f"Collection config: {args.collection}")
    print(f"Model config: {args.model}")


if __name__ == "__main__":
    main()
