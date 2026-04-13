from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    build_weak_baseline_candidate_record,
    default_candidates_out_path,
    load_jsonl,
    normalize_task_record,
    upsert_jsonl_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic weak-baseline candidate rows for the local-only workflow."
    )
    parser.add_argument(
        "--tasks-path",
        type=Path,
        default=Path("data/processed/tasks/tasks.jsonl"),
        help="Input tasks JSONL file.",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=default_candidates_out_path("weak_baseline"),
        help="Output weak-baseline candidates JSONL file.",
    )
    parser.add_argument(
        "--n-candidates-per-task",
        type=int,
        default=1,
        help="Number of weak baseline variants to write per task.",
    )
    parser.add_argument(
        "--generator-model",
        default="weak_rag_baseline",
        help="Generator model label to store in weak-baseline candidate rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = [normalize_task_record(row) for row in load_jsonl(args.tasks_path)]
    if not tasks:
        raise SystemExit(f"No task rows found in {args.tasks_path}")
    if args.n_candidates_per_task < 1:
        raise SystemExit("--n-candidates-per-task must be at least 1")

    rows = []
    for task in tasks:
        for candidate_rank in range(1, args.n_candidates_per_task + 1):
            rows.append(
                build_weak_baseline_candidate_record(
                    task,
                    candidate_rank=candidate_rank,
                    generator_model=args.generator_model,
                )
            )

    upsert_jsonl_rows(args.out_path, rows, key_field="candidate_id")
    print(f"Wrote {len(rows)} weak-baseline candidate row(s) to {args.out_path}")


if __name__ == "__main__":
    main()
