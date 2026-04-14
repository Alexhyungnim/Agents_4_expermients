from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from common import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build DPO dataset rows from accepted_silver vs rejected judged candidates for the same task."
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
        default=Path("data/processed/datasets/dpo/train.jsonl"),
        help="Output DPO dataset JSONL file.",
    )
    parser.add_argument(
        "--max-rejected-per-task",
        type=int,
        default=3,
        help="Maximum number of rejected examples to pair with the chosen example per task.",
    )
    return parser.parse_args()


def _compatibility_score(row: dict[str, object]) -> int:
    score = row.get("compatibility_total_score_0to2", row.get("total_score"))
    if score is None:
        raise ValueError(f"Judged row for candidate {row.get('candidate_id')} is missing compatibility score")
    return int(score)


def main() -> None:
    args = parse_args()
    tasks = {x["task_id"]: x for x in load_jsonl(args.tasks_path)}
    if not tasks:
        raise SystemExit(f"No task rows found in {args.tasks_path}")

    candidates = {x["candidate_id"]: x for x in load_jsonl(args.candidates_path)}
    if not candidates:
        raise SystemExit(f"No candidate rows found in {args.candidates_path}")

    by_task: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in load_jsonl(args.judged_path):
        by_task[row["task_id"]].append(row)

    dpo_rows: list[dict[str, object]] = []
    skipped_tasks = 0

    for task_id, judged_rows in by_task.items():
        if task_id not in tasks:
            raise SystemExit(f"Judged rows reference missing task_id {task_id}")

        accepted = [
            row for row in judged_rows
            if row.get("storage_bucket") == "accepted_silver"
            and row.get("overall_verdict") == "accept_silver"
        ]
        rejected = [
            row for row in judged_rows
            if row.get("storage_bucket") == "rejected"
            and row.get("overall_verdict") == "reject"
        ]

        if not accepted or not rejected:
            skipped_tasks += 1
            continue

        accepted = sorted(accepted, key=_compatibility_score, reverse=True)
        rejected = sorted(rejected, key=_compatibility_score)

        chosen_judged = accepted[0]
        chosen_candidate_id = chosen_judged["candidate_id"]
        if chosen_candidate_id not in candidates:
            raise SystemExit(f"Accepted judged row references missing candidate_id {chosen_candidate_id}")

        chosen = candidates[chosen_candidate_id]["final_proposal"]
        prompt = (
            "You are an engineering experiment design assistant. Return JSON only.\n\n"
            f"Task:\n{json.dumps(tasks[task_id], ensure_ascii=False, indent=2)}\n"
        )

        for rejected_judged in rejected[: max(1, args.max_rejected_per_task)]:
            rejected_candidate_id = rejected_judged["candidate_id"]
            if rejected_candidate_id not in candidates:
                raise SystemExit(
                    f"Rejected judged row references missing candidate_id {rejected_candidate_id}"
                )

            dpo_rows.append(
                {
                    "prompt": prompt,
                    "chosen": json.dumps(chosen, ensure_ascii=False),
                    "rejected": json.dumps(
                        candidates[rejected_candidate_id]["final_proposal"],
                        ensure_ascii=False,
                    ),
                    "task_id": task_id,
                    "chosen_candidate_id": chosen_candidate_id,
                    "rejected_candidate_id": rejected_candidate_id,
                    "chosen_storage_bucket": chosen_judged["storage_bucket"],
                    "rejected_storage_bucket": rejected_judged["storage_bucket"],
                    "chosen_compatibility_total_score_0to2": _compatibility_score(chosen_judged),
                    "rejected_compatibility_total_score_0to2": _compatibility_score(rejected_judged),
                    "chosen_raw_total_score_1to5": chosen_judged.get("raw_total_score_1to5"),
                    "rejected_raw_total_score_1to5": rejected_judged.get("raw_total_score_1to5"),
                }
            )

    if not dpo_rows:
        raise SystemExit(
            f"No accepted_silver vs rejected pairs found in {args.judged_path}. "
            "DPO pairs require both buckets for the same task_id."
        )

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with args.out_path.open("w", encoding="utf-8") as f:
        for row in dpo_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(dpo_rows)} DPO rows to {args.out_path}")
    print(f"Skipped {skipped_tasks} task(s) without both accepted_silver and rejected examples.")


if __name__ == "__main__":
    main()
