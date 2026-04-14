from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    build_candidate_id_with_label,
    convert_main_rag_outputs_to_candidate_record,
    load_jsonl,
    maybe_read_json,
    normalize_task_record,
    resolve_task_for_candidate_source,
    upsert_jsonl_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert current main-side RAG outputs into canonical candidate rows."
    )
    parser.add_argument(
        "--tasks-path",
        type=Path,
        default=Path("data/processed/tasks/tasks.jsonl"),
        help="Input tasks JSONL file.",
    )
    parser.add_argument(
        "--task-id",
        default=None,
        help="Optional task_id to associate with the imported RAG candidate.",
    )
    parser.add_argument(
        "--rag1-path",
        type=Path,
        default=Path("../outputs/rag1_latest_output.json"),
        help="Path to main-side RAG1 output JSON.",
    )
    parser.add_argument(
        "--rag2-path",
        type=Path,
        default=Path("../outputs/rag2_advice.json"),
        help="Path to main-side RAG2 advice JSON.",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("data/processed/candidates/candidates.jsonl"),
        help="Canonical candidates JSONL output path.",
    )
    parser.add_argument(
        "--candidate-rank",
        type=int,
        default=1,
        help="Candidate rank to write for the imported RAG candidate.",
    )
    parser.add_argument(
        "--generator-model",
        default="main_rag_pipeline",
        help="Generator model label to store in the candidate row.",
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

    rag1_payload = maybe_read_json(args.rag1_path)
    if rag1_payload is None:
        raise SystemExit(f"RAG1 output not found: {args.rag1_path}")
    rag2_payload = maybe_read_json(args.rag2_path)

    task = resolve_task_for_candidate_source(
        tasks,
        task_id=args.task_id,
        goal_text=rag1_payload.get("available_bom", {}).get("goal"),
    )

    candidate_id = build_candidate_id_with_label(
        task["task_id"],
        "rag_main",
        args.candidate_rank,
        candidate_source="strong",
    )
    row = convert_main_rag_outputs_to_candidate_record(
        task=task,
        rag1_payload=rag1_payload,
        rag2_payload=rag2_payload,
        candidate_id=candidate_id,
        candidate_rank=args.candidate_rank,
        generator_model=args.generator_model,
        candidate_source="strong",
    )

    upsert_jsonl_rows(args.out_path, [row], key_field="candidate_id")
    print(f"Imported main-side RAG candidate to {args.out_path}")
    print(f"task_id={task['task_id']} candidate_id={candidate_id}")


if __name__ == "__main__":
    main()
