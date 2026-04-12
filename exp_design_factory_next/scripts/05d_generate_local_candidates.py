from __future__ import annotations

import argparse
from copy import deepcopy

from common import (
    LocalTransformersLLM,
    build_candidate_id_with_label,
    build_local_candidate_outline_prompt,
    candidate_payload_from_outline_text,
    load_jsonl,
    normalize_task_record,
    upsert_jsonl_rows,
)

from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate canonical candidate rows with a local Hugging Face causal LM."
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
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Local Hugging Face model name.",
    )
    parser.add_argument(
        "--generator-model",
        default="local_hf_qwen",
        help="Generator model label to store in candidate rows.",
    )
    parser.add_argument(
        "--n-candidates-per-task",
        type=int,
        default=2,
        help="Number of local candidates to generate per task.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=900,
        help="Maximum new tokens for each candidate generation call.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Sampling temperature. Use 0.0 for deterministic generation.",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip parse failures instead of failing the whole run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = [normalize_task_record(row) for row in load_jsonl(args.tasks_path)]
    if not tasks:
        raise SystemExit(f"No task rows found in {args.tasks_path}")
    if args.n_candidates_per_task < 1:
        raise SystemExit("--n-candidates-per-task must be at least 1")

    llm = LocalTransformersLLM(args.model_name)
    rows: list[dict[str, object]] = []
    errors: list[str] = []

    for task in tasks:
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
            base_payload = candidate_payload_from_outline_text(
                base_raw_text,
                task=task,
                candidate_rank=1,
            )
        except Exception as exc:
            errors.append(f"{task['task_id']}: base local candidate generation failed: {exc}")
            continue

        for candidate_rank in range(1, args.n_candidates_per_task + 1):
            candidate_id = build_candidate_id_with_label(
                task["task_id"],
                "local",
                candidate_rank,
                candidate_source="strong",
            )
            candidate_payload = deepcopy(base_payload)
            raw_text = base_raw_text.strip()

            if candidate_rank > 1:
                # Keep a few deliberately weak strong-side contrast variants so local-only DPO has a chance
                # to form accepted vs rejected pairs for the same task without changing the schema.
                candidate_payload["reasoning_trace"]["risk_check"] = (
                    "Local contrast variant derived from the base candidate for local-only DPO."
                )
                candidate_payload["final_proposal"]["controls"] = []
                candidate_payload["final_proposal"]["design"]["replicates"] = None
                if candidate_rank > 2:
                    candidate_payload["final_proposal"]["analysis_plan"] = []
                    candidate_payload["final_proposal"]["evidence_used"] = []
                    candidate_payload["final_proposal"]["feasibility_checks"] = []
                raw_text = f"{raw_text}\n\n[local_contrast_variant rank={candidate_rank}]"

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

    if errors and not args.skip_invalid:
        raise SystemExit("Local candidate generation failed:\n- " + "\n- ".join(errors))
    if not rows:
        raise SystemExit("No valid local candidates were generated.")

    upsert_jsonl_rows(args.out_path, rows, key_field="candidate_id")
    print(f"Wrote {len(rows)} local candidate row(s) to {args.out_path}")
    if errors:
        print(f"Skipped {len(errors)} candidate(s) due to parse/validation errors.")
        for error in errors:
            print(f"[WARN] {error}")


if __name__ == "__main__":
    main()
