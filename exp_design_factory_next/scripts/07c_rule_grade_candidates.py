from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from common import (
    build_rule_first_row,
    default_candidates_out_path,
    dump_jsonl,
    infer_candidate_source_from_path,
    load_jsonl,
    normalize_candidate_source,
    normalize_candidate_record,
    normalize_task_record,
)


def default_rule_out_path(candidate_source: str) -> Path:
    if candidate_source == "weak_baseline":
        return Path("data/processed/judged/rule_first_weak_rag_candidates.jsonl")
    return Path("data/processed/judged/rule_first_candidates.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run rule-first grading on canonical candidate rows before local LLM judging."
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
        default=None,
        help="Input candidates JSONL file.",
    )
    parser.add_argument(
        "--candidate-source",
        default=None,
        help="Optional source shorthand. Allowed: strong, weak_baseline. Used when --candidates-path is omitted.",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=None,
        help="Rule-first output JSONL file. Defaults by candidate_source.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = {
        row["task_id"]: normalize_task_record(row)
        for row in load_jsonl(args.tasks_path)
    }
    if not tasks:
        raise SystemExit(f"No task rows found in {args.tasks_path}")

    cli_source = (
        normalize_candidate_source(args.candidate_source)
        if args.candidate_source is not None
        else None
    )
    candidates_path = args.candidates_path or default_candidates_out_path(cli_source or "strong")
    source_hint = cli_source or infer_candidate_source_from_path(candidates_path)
    candidates = [
        normalize_candidate_record(row, source_hint=source_hint)
        for row in load_jsonl(candidates_path)
    ]
    if not candidates:
        raise SystemExit(f"No candidate rows found in {candidates_path}")

    out_path = args.out_path or default_rule_out_path(source_hint)
    rows = []
    for candidate in candidates:
        task = tasks.get(candidate["task_id"])
        if task is None:
            raise SystemExit(
                f"Candidate {candidate['candidate_id']} references missing task_id {candidate['task_id']}"
            )
        rows.append(build_rule_first_row(candidate, task))

    dump_jsonl(out_path, rows)
    print(f"Saved {len(rows)} rule-first grading row(s) to {out_path}")
    print(
        f"candidate_source={source_hint} "
        f"tasks_covered={len({row['task_id'] for row in rows})} "
        f"candidates_path={candidates_path}"
    )
    print(f"rule_hard_fail_counts={dict(Counter(bool(row['rule_hard_fail']) for row in rows))}")


if __name__ == "__main__":
    main()
