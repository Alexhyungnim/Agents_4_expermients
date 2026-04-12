from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    tasks = {x["task_id"]: x for x in load_jsonl(Path("data/processed/tasks/tasks.jsonl"))}
    cands = {x["candidate_id"]: x for x in load_jsonl(Path("data/processed/candidates/candidates.jsonl"))}
    by_task = defaultdict(list)
    for row in load_jsonl(Path("data/processed/judged/judged_candidates.jsonl")):
        by_task[row["task_id"]].append(row)

    out = Path("data/processed/datasets/dpo/train.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for task_id, rows in by_task.items():
            accepted = [r for r in rows if r["overall_verdict"] == "accept_silver"]
            rejected = [r for r in rows if r["overall_verdict"] != "accept_silver"]
            if not accepted or not rejected:
                continue

            accepted = sorted(accepted, key=lambda x: x["total_score"], reverse=True)
            rejected = sorted(rejected, key=lambda x: x["total_score"])

            chosen = cands[accepted[0]["candidate_id"]]["final_proposal"]
            prompt = (
                "You are an engineering experiment design assistant. Return JSON only.\n\n"
                f"Task:\n{json.dumps(tasks[task_id], ensure_ascii=False, indent=2)}\n"
            )
            for rej in rejected[:3]:
                row = {
                    "prompt": prompt,
                    "chosen": json.dumps(chosen, ensure_ascii=False),
                    "rejected": json.dumps(cands[rej["candidate_id"]]["final_proposal"], ensure_ascii=False),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved DPO dataset to {out}")


if __name__ == "__main__":
    main()
