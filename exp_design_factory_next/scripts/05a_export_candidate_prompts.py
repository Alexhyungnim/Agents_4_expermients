from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    build_candidate_id,
    build_candidate_prompt,
    dump_jsonl,
    normalize_candidate_source,
    normalize_task_record,
    load_jsonl,
    write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one GPT-web prompt file per candidate request from tasks.jsonl."
    )
    parser.add_argument(
        "--tasks-path",
        type=Path,
        default=Path("data/processed/tasks/tasks.jsonl"),
        help="Input tasks JSONL file.",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=Path("data/manual/candidates"),
        help="Export root. Prompt files go under prompts/ and pasted outputs go under manual_raw/.",
    )
    parser.add_argument(
        "--candidate-source",
        default="strong",
        help="Candidate source label to assign. Allowed: strong, weak_baseline.",
    )
    parser.add_argument(
        "--n-candidates-per-task",
        type=int,
        default=4,
        help="Number of prompt variants to export per task.",
    )
    parser.add_argument(
        "--generator-model",
        default="manual_gpt_web",
        help="Generator model label to store in the manifest and imported candidate rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidate_source = normalize_candidate_source(args.candidate_source)
    tasks = [normalize_task_record(row) for row in load_jsonl(args.tasks_path)]

    if not tasks:
        raise SystemExit(f"No task rows found in {args.tasks_path}")
    if args.n_candidates_per_task < 1:
        raise SystemExit("--n-candidates-per-task must be at least 1")

    prompts_dir = args.export_dir / "prompts"
    manual_raw_dir = args.export_dir / "manual_raw"
    manifest_path = args.export_dir / "manifest.jsonl"

    prompts_dir.mkdir(parents=True, exist_ok=True)
    manual_raw_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []

    for task in tasks:
        for candidate_rank in range(1, args.n_candidates_per_task + 1):
            candidate_id = build_candidate_id(task["task_id"], candidate_rank, candidate_source)
            prompt_file = prompts_dir / f"{candidate_id}.txt"
            raw_file = manual_raw_dir / f"{candidate_id}.txt"

            prompt_text = build_candidate_prompt(
                task,
                candidate_rank=candidate_rank,
                n_candidates=args.n_candidates_per_task,
            )
            write_text(prompt_file, prompt_text)

            manifest_rows.append(
                {
                    "task_id": task["task_id"],
                    "candidate_id": candidate_id,
                    "candidate_source": candidate_source,
                    "candidate_rank": candidate_rank,
                    "generator_model": args.generator_model,
                    "prompt_file": str(prompt_file),
                    "raw_file": str(raw_file),
                }
            )

    dump_jsonl(manifest_path, manifest_rows)

    print(f"Exported {len(manifest_rows)} candidate prompt files.")
    print(f"Prompt directory: {prompts_dir}")
    print(f"Paste raw GPT-web outputs into: {manual_raw_dir}")
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
