from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    LocalTransformersLLM,
    build_candidate_id_with_label,
    build_baseline_safe_candidate_payload,
    build_local_candidate_outline_prompt,
    build_strong_contrast_variant_payload,
    candidate_payload_from_outline_text,
    load_jsonl,
    normalize_task_record,
    upsert_jsonl_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one baseline-safe strong candidate plus controlled contrast variants with a local Hugging Face causal LM."
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
        default=Path("data/processed/candidates/candidates.jsonl"),
        help="Canonical candidates JSONL output path.",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2-1.5B-Instruct",
        help="Local Hugging Face model name.",
    )
    parser.add_argument(
        "--generator-model",
        default="local_hf_qwen2_1p5b",
        help="Generator model label to store in candidate rows.",
    )
    parser.add_argument(
        "--n-candidates-per-task",
        type=int,
        default=4,
        help="Number of local candidates to generate per task.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Optional cap on the number of task rows to process from tasks.jsonl.",
    )
    parser.add_argument(
        "--replace-output",
        action="store_true",
        help="Replace the output JSONL instead of upserting into an existing file.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=220,
        help="Maximum new tokens for each candidate generation call.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0.0 for deterministic generation.",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Compatibility flag. Base-outline failures now fall back to a deterministic baseline-safe candidate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = [normalize_task_record(row) for row in load_jsonl(args.tasks_path)]
    if not tasks:
        raise SystemExit(f"No task rows found in {args.tasks_path}")
    if args.max_tasks is not None and args.max_tasks < 1:
        raise SystemExit("--max-tasks must be at least 1 when provided.")
    if args.n_candidates_per_task < 1:
        raise SystemExit("--n-candidates-per-task must be at least 1")
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]

    llm = LocalTransformersLLM(args.model_name)
    rows: list[dict[str, object]] = []
    warnings: list[str] = []

    for task in tasks:
        base_seed_payload: dict[str, object] | None = None
        base_raw_text = ""
        base_prompt = build_local_candidate_outline_prompt(
            task,
            candidate_rank=1,
            n_candidates=args.n_candidates_per_task,
        )
        try:
            base_raw_text = llm.complete(
                base_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            base_seed_payload = candidate_payload_from_outline_text(
                base_raw_text,
                task=task,
                candidate_rank=1,
            )
        except Exception as exc:
            warnings.append(
                f"{task['task_id']}: local outline generation fell back to deterministic baseline-safe candidate: {exc}"
            )
            base_raw_text = ""

        base_payload = build_baseline_safe_candidate_payload(
            task,
            seed_payload=base_seed_payload,
        )

        for candidate_rank in range(1, args.n_candidates_per_task + 1):
            candidate_id = build_candidate_id_with_label(
                task["task_id"],
                "local",
                candidate_rank,
                candidate_source="strong",
            )
            if candidate_rank == 1:
                candidate_payload = base_payload
                raw_text = (base_raw_text.strip() or "deterministic_local_baseline_safe_candidate") + (
                    "\n\n[baseline_safe_candidate]"
                )
            else:
                candidate_payload, variant_label = build_strong_contrast_variant_payload(
                    task,
                    base_payload,
                    candidate_rank=candidate_rank,
                )
                raw_text = (base_raw_text.strip() or "deterministic_local_baseline_safe_candidate") + (
                    f"\n\n[local_contrast_variant rank={candidate_rank} mode={variant_label}]"
                )

            rows.append(
                {
                    "candidate_id": candidate_id,
                    "task_id": task["task_id"],
                    "candidate_source": "strong",
                    "generator_model": args.generator_model,
                    "candidate_rank": candidate_rank,
                    "reasoning_trace": candidate_payload["reasoning_trace"],
                    "final_proposal": candidate_payload["final_proposal"],
                    "raw_text": raw_text,
                }
            )

    if not rows:
        raise SystemExit("No valid local candidates were generated.")

    if args.replace_output and args.out_path.exists():
        args.out_path.unlink()
    upsert_jsonl_rows(args.out_path, rows, key_field="candidate_id")
    print(f"Wrote {len(rows)} local candidate row(s) to {args.out_path}")
    print(
        f"tasks_processed={len(tasks)} "
        f"n_candidates_per_task={args.n_candidates_per_task}"
    )
    if warnings:
        print(f"Used deterministic baseline fallback for {len(warnings)} task(s).")
        for warning in warnings:
            print(f"[WARN] {warning}")


if __name__ == "__main__":
    main()
