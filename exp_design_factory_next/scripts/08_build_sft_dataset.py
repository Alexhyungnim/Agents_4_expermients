from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SFT dataset rows from judged accepted_silver candidates."
    )
    parser.add_argument(
        "--tasks-path",
        type=Path,
        default=Path("data/processed/tasks/tasks.jsonl"),
        help="Input tasks JSONL file.",
    )
    parser.add_argument(
        "--candidates-path",
        type=Path,
        default=Path("data/processed/candidates/candidates.jsonl"),
        help="Input candidates JSONL file.",
    )
    parser.add_argument(
        "--judged-path",
        type=Path,
        default=Path("data/processed/judged/judged_candidates.jsonl"),
        help="Input judged JSONL file.",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("data/processed/datasets/sft/train.jsonl"),
        help="Output SFT dataset JSONL file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = {x["task_id"]: x for x in load_jsonl(args.tasks_path)}
    if not tasks:
        raise SystemExit(f"No task rows found in {args.tasks_path}")

    candidates = {x["candidate_id"]: x for x in load_jsonl(args.candidates_path)}
    if not candidates:
        raise SystemExit(f"No candidate rows found in {args.candidates_path}")

    selected_rows: list[dict[str, object]] = []
    for judged in load_jsonl(args.judged_path):
        if judged.get("storage_bucket") != "accepted_silver":
            continue
        if judged.get("overall_verdict") != "accept_silver":
            continue

        candidate_id = judged["candidate_id"]
        if candidate_id not in candidates:
            raise SystemExit(f"Judged row references missing candidate_id {candidate_id}")

        candidate = candidates[candidate_id]
        task_id = candidate["task_id"]
        if task_id not in tasks:
            raise SystemExit(f"Candidate {candidate_id} references missing task_id {task_id}")

        task = tasks[task_id]
        prompt = (
            "You are an engineering experiment design assistant. Return JSON only.\n\n"
            f"Task:\n{json.dumps(task, ensure_ascii=False, indent=2)}\n"
        )
        answer = json.dumps(candidate["final_proposal"], ensure_ascii=False)

        selected_rows.append(
            {
                "text": f"<|user|>\n{prompt}\n<|assistant|>\n{answer}",
                "task_id": task_id,
                "candidate_id": candidate_id,
                "candidate_source": candidate.get("candidate_source", "strong"),
                "storage_bucket": judged["storage_bucket"],
                "overall_verdict": judged["overall_verdict"],
                "compatibility_total_score_0to2": judged.get("compatibility_total_score_0to2", judged.get("total_score")),
                "raw_total_score_1to5": judged.get("raw_total_score_1to5"),
            }
        )

    if not selected_rows:
        raise SystemExit(
            f"No accepted_silver judged rows found in {args.judged_path}. "
            "SFT data is built only from accepted_silver rows."
        )

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with args.out_path.open("w", encoding="utf-8") as f:
        for row in selected_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(selected_rows)} SFT rows to {args.out_path}")


if __name__ == "__main__":
    main()
