from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    LocalTransformersLLM,
    build_structural_fallback_judged_row,
    build_judge_prompt,
    build_rule_only_judged_row,
    compute_compatibility_total_score_0to2,
    compute_detailed_verdict,
    compute_overall_verdict,
    compute_raw_total_score_1to5,
    compute_storage_bucket,
    extract_json_payload,
    infer_candidate_source_from_path,
    load_merged_jsonl_rows,
    load_jsonl,
    looks_like_judge_payload,
    materialize_local_bucket_views,
    normalize_candidate_record,
    normalize_manual_judge_payload,
    normalize_task_record,
    upsert_jsonl_rows,
)


def default_rule_path(candidate_source: str) -> Path:
    if candidate_source == "weak_baseline":
        return Path("data/processed/judged/rule_first_weak_rag_candidates.jsonl")
    return Path("data/processed/judged/rule_first_candidates.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local LLM judging after rule-first grading and write canonical judged JSONL."
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
        "--rule-graded-path",
        type=Path,
        default=None,
        help="Rule-first grading JSONL file. Defaults by candidate_source.",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=None,
        help="Canonical judged JSONL output path. Defaults by candidate_source.",
    )
    parser.add_argument(
        "--strong-judged-path",
        type=Path,
        default=Path("data/processed/judged/judged_candidates.jsonl"),
        help="Canonical strong judged JSONL used when materializing local-only bucket views.",
    )
    parser.add_argument(
        "--weak-judged-path",
        type=Path,
        default=Path("data/processed/judged/judged_weak_rag_candidates.jsonl"),
        help="Canonical weak-baseline judged JSONL used when materializing local-only bucket views.",
    )
    parser.add_argument(
        "--bucket-dir",
        type=Path,
        default=Path("data/processed/judged/local_only_buckets"),
        help="Directory for materialized local-only bucket views.",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Local Hugging Face model name for judging.",
    )
    parser.add_argument(
        "--judge-model",
        default="local_hf_qwen_judge",
        help="Judge model label stored in judged rows.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1200,
        help="Maximum new tokens for each local judge call.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for local judging.",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip candidates whose local judge output cannot be parsed.",
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
    candidates = {
        row["candidate_id"]: normalize_candidate_record(row, source_hint=source_hint)
        for row in load_jsonl(args.candidates_path)
    }
    if not candidates:
        raise SystemExit(f"No candidate rows found in {args.candidates_path}")

    rule_path = args.rule_graded_path or default_rule_path(source_hint)
    rule_rows = {
        row["candidate_id"]: row
        for row in load_jsonl(rule_path)
    }
    if not rule_rows:
        raise SystemExit(f"No rule-first grading rows found in {rule_path}")

    llm = LocalTransformersLLM(args.model_name)
    judged_rows: list[dict[str, object]] = []
    errors: list[str] = []
    fallback_rows = 0

    for candidate_id, candidate in candidates.items():
        task = tasks.get(candidate["task_id"])
        if task is None:
            raise SystemExit(
                f"Candidate {candidate_id} references missing task_id {candidate['task_id']}"
            )

        rule_first_row = rule_rows.get(candidate_id)
        if rule_first_row is None:
            raise SystemExit(f"Candidate {candidate_id} is missing from rule-first grading file {rule_path}")

        if bool(rule_first_row.get("rule_hard_fail")):
            judged_rows.append(
                build_rule_only_judged_row(
                    candidate=candidate,
                    task=task,
                    rule_first_row=rule_first_row,
                    judge_model=f"{args.judge_model}_rule_only",
                    judge_contract_version="judge_v0_local",
                )
            )
            continue

        prompt = build_judge_prompt(task, candidate)
        try:
            raw_text = llm.complete(
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            payload = extract_json_payload(
                raw_text,
                context=f"local judgment for {candidate_id}",
                predicate=looks_like_judge_payload,
            )
            normalized = normalize_manual_judge_payload(
                payload,
                candidate=candidate,
                task=task,
            )

            raw_total_score_1to5 = compute_raw_total_score_1to5(normalized["rubric"])
            compatibility_total_score_0to2 = compute_compatibility_total_score_0to2(
                normalized["rubric_compatibility"]
            )
            detailed_verdict = compute_detailed_verdict(
                compatibility_total_score_0to2,
                normalized["hard_fail"],
            )
            overall_verdict = compute_overall_verdict(
                compatibility_total_score_0to2,
                normalized["hard_fail"],
            )
            storage_bucket = compute_storage_bucket(candidate["candidate_source"], overall_verdict)

            row: dict[str, object] = {
                "candidate_id": candidate["candidate_id"],
                "task_id": candidate["task_id"],
                "candidate_source": candidate["candidate_source"],
                "judge_model": args.judge_model,
                "judge_contract_version": "judge_v0_local",
                "rubric": normalized["rubric"],
                "rubric_compatibility": normalized["rubric_compatibility"],
                "rule_checks": normalized["rule_checks"],
                "hard_fail": normalized["hard_fail"],
                "hard_fail_reasons": normalized["hard_fail_reasons"],
                "failure_tags": normalized["failure_tags"],
                "raw_total_score_1to5": raw_total_score_1to5,
                "compatibility_total_score_0to2": compatibility_total_score_0to2,
                "total_score": compatibility_total_score_0to2,
                "detailed_verdict": detailed_verdict,
                "overall_verdict": overall_verdict,
                "storage_bucket": storage_bucket,
                "summary": normalized["summary"],
                "overall_reasoning": normalized["overall_reasoning"],
            }
            judged_rows.append(row)
        except Exception as exc:
            judged_rows.append(
                build_structural_fallback_judged_row(
                    candidate=candidate,
                    task=task,
                    judge_model=f"{args.judge_model}_fallback",
                    judge_contract_version="judge_v0_local",
                    summary_note=(
                        "Local judge fallback used because the local LLM judgment "
                        f"could not be parsed: {exc}"
                    ),
                )
            )
            fallback_rows += 1

    if errors and not args.skip_invalid:
        raise SystemExit("Local judging failed:\n- " + "\n- ".join(errors))
    if not judged_rows:
        raise SystemExit("No judged rows were produced.")

    strong_judged_path = args.strong_judged_path
    weak_judged_path = args.weak_judged_path
    out_path = args.out_path or (
        weak_judged_path if source_hint == "weak_baseline" else strong_judged_path
    )
    upsert_jsonl_rows(out_path, judged_rows, key_field="candidate_id")
    if source_hint == "weak_baseline":
        weak_judged_path = out_path
    else:
        strong_judged_path = out_path

    bucket_rows = load_merged_jsonl_rows(
        [strong_judged_path, weak_judged_path],
        key_field="candidate_id",
    )
    materialize_local_bucket_views(bucket_rows, out_dir=args.bucket_dir)

    print(f"Wrote {len(judged_rows)} judged row(s) to {out_path}")
    print(f"Materialized local-only bucket views under {args.bucket_dir}")
    if fallback_rows:
        print(f"Used structural fallback judging for {fallback_rows} candidate(s).")
    if errors:
        print(f"Skipped {len(errors)} candidate(s) due to local-judge parse/validation errors.")
        for error in errors:
            print(f"[WARN] {error}")


if __name__ == "__main__":
    main()
