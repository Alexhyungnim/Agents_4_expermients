from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    build_judge_prompt,
    dump_jsonl,
    infer_candidate_source_from_path,
    load_jsonl,
    normalize_candidate_record,
    normalize_task_record,
    write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export one GPT-web judge prompt file per candidate row."
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
        "--export-dir",
        type=Path,
        default=Path("data/manual/judgments"),
        help="Export root. Prompt files go under prompts/ and pasted outputs go under manual_raw/.",
    )
    parser.add_argument(
        "--judge-model",
        default="manual_gpt_web_judge",
        help="Judge model label to store in the manifest and imported judged rows.",
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

    source_hint = infer_candidate_source_from_path(args.candidates_path)
    candidates = [
        normalize_candidate_record(row, source_hint=source_hint)
        for row in load_jsonl(args.candidates_path)
    ]
    if not candidates:
        raise SystemExit(f"No candidate rows found in {args.candidates_path}")

    prompts_dir = args.export_dir / "prompts"
    manual_raw_dir = args.export_dir / "manual_raw"
    manifest_path = args.export_dir / "manifest.jsonl"

    prompts_dir.mkdir(parents=True, exist_ok=True)
    manual_raw_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []

    for candidate in candidates:
        task = tasks.get(candidate["task_id"])
        if task is None:
            raise SystemExit(
                f"Candidate {candidate['candidate_id']} references missing task_id {candidate['task_id']}"
            )

        prompt_file = prompts_dir / f"{candidate['candidate_id']}.txt"
        raw_file = manual_raw_dir / f"{candidate['candidate_id']}.txt"
        prompt_text = build_judge_prompt(task, candidate)
        write_text(prompt_file, prompt_text)

        manifest_rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "task_id": candidate["task_id"],
                "candidate_source": candidate["candidate_source"],
                "judge_model": args.judge_model,
                "prompt_file": str(prompt_file),
                "raw_file": str(raw_file),
            }
        )

    dump_jsonl(manifest_path, manifest_rows)
    print(f"Exported {len(manifest_rows)} judge prompt files.")
    print(f"Prompt directory: {prompts_dir}")
    print(f"Paste raw GPT-web outputs into: {manual_raw_dir}")
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
