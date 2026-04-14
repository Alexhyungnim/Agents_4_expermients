from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    canonicalize_local_judge_summary,
    LocalTransformersLLM,
    build_local_judge_prompt,
    build_local_judge_repair_prompt,
    build_structural_fallback_judged_row,
    build_rule_only_judged_row,
    compute_compatibility_total_score_0to2,
    compute_detailed_verdict,
    compute_overall_verdict,
    compute_raw_total_score_1to5,
    compute_storage_bucket,
    default_candidates_out_path,
    extract_json_payload,
    infer_candidate_source_from_path,
    load_merged_jsonl_rows,
    load_jsonl,
    looks_like_judge_payload,
    looks_like_collapsed_local_judge_rubric,
    materialize_local_bucket_views,
    normalize_candidate_record,
    normalize_candidate_source,
    normalize_manual_judge_payload,
    normalize_task_record,
    salvage_local_judge_payload,
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
        default=None,
        help="Input candidates JSONL file.",
    )
    parser.add_argument(
        "--candidate-source",
        default=None,
        help="Optional source shorthand. Allowed: strong, weak_baseline. Used when --candidates-path is omitted.",
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
        default="Qwen/Qwen2-1.5B-Instruct",
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
        default=320,
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
    parser.add_argument(
        "--repair-attempts",
        type=int,
        default=1,
        help="Number of local repair attempts after an unparseable judge output.",
    )
    parser.add_argument(
        "--repair-max-new-tokens",
        type=int,
        default=220,
        help="Maximum new tokens for each local repair attempt.",
    )
    return parser.parse_args()


def parse_or_salvage_local_judge_payload(raw_text: str, *, context: str) -> tuple[dict[str, object], str]:
    try:
        payload = extract_json_payload(
            raw_text,
            context=context,
            predicate=looks_like_judge_payload,
        )
        return payload, "direct"
    except Exception as direct_exc:
        salvaged = salvage_local_judge_payload(raw_text)
        if salvaged is not None:
            return salvaged, "salvaged"
        raise direct_exc


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
    candidates = {
        row["candidate_id"]: normalize_candidate_record(row, source_hint=source_hint)
        for row in load_jsonl(candidates_path)
    }
    if not candidates:
        raise SystemExit(f"No candidate rows found in {candidates_path}")

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
    initial_direct_rows = 0
    initial_salvaged_rows = 0
    repaired_rows = 0
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

        prompt = build_local_judge_prompt(
            task,
            candidate,
            rule_checks=rule_first_row.get("rule_checks"),
        )
        raw_text = ""
        try:
            raw_text = llm.complete(
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            payload, parse_mode = parse_or_salvage_local_judge_payload(
                raw_text,
                context=f"local judgment for {candidate_id}",
            )
            if parse_mode == "direct":
                initial_direct_rows += 1
            else:
                initial_salvaged_rows += 1
            normalized = normalize_manual_judge_payload(
                payload,
                candidate=candidate,
                task=task,
            )
            if looks_like_collapsed_local_judge_rubric(normalized):
                raise ValueError(
                    "Collapsed local judge rubric with score=1 across all dimensions."
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
            summary = canonicalize_local_judge_summary(
                normalized["summary"],
                candidate_source=candidate["candidate_source"],
                overall_verdict=overall_verdict,
                hard_fail=normalized["hard_fail"],
            )

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
                "summary": summary,
                "overall_reasoning": normalized["overall_reasoning"],
            }
            judged_rows.append(row)
        except Exception as exc:
            repaired_payload: dict[str, object] | None = None
            last_exc = exc
            for attempt_idx in range(1, max(0, args.repair_attempts) + 1):
                repair_prompt = build_local_judge_repair_prompt(raw_text)
                repair_text = llm.complete(
                    repair_prompt,
                    max_new_tokens=args.repair_max_new_tokens,
                    temperature=0.0,
                )
                try:
                    repaired_payload, parse_mode = parse_or_salvage_local_judge_payload(
                        repair_text,
                        context=f"local judgment repair {attempt_idx} for {candidate_id}",
                    )
                    normalized = normalize_manual_judge_payload(
                        repaired_payload,
                        candidate=candidate,
                        task=task,
                    )
                    if looks_like_collapsed_local_judge_rubric(normalized):
                        raise ValueError(
                            "Collapsed local judge rubric with score=1 across all dimensions."
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
                    summary = canonicalize_local_judge_summary(
                        normalized["summary"],
                        candidate_source=candidate["candidate_source"],
                        overall_verdict=overall_verdict,
                        hard_fail=normalized["hard_fail"],
                    )

                    row = {
                        "candidate_id": candidate["candidate_id"],
                        "task_id": candidate["task_id"],
                        "candidate_source": candidate["candidate_source"],
                        "judge_model": f"{args.judge_model}_repair",
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
                        "summary": summary,
                        "overall_reasoning": normalized["overall_reasoning"],
                    }
                    judged_rows.append(row)
                    repaired_rows += 1
                    break
                except Exception as repair_exc:
                    last_exc = repair_exc
                    repaired_payload = None

            if repaired_payload is not None:
                continue

            judged_rows.append(
                build_structural_fallback_judged_row(
                    candidate=candidate,
                    task=task,
                    judge_model=f"{args.judge_model}_fallback",
                    judge_contract_version="judge_v0_local",
                    summary_note=(
                        "Local judge fallback used because the local LLM judgment "
                        f"could not be parsed after repair attempts: {last_exc}"
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
    bucket_counts = materialize_local_bucket_views(bucket_rows, out_dir=args.bucket_dir)

    print(f"Wrote {len(judged_rows)} judged row(s) to {out_path}")
    print(f"Materialized local-only bucket views under {args.bucket_dir}")
    print(
        "Bucket view counts: "
        f"accepted_silver_lite={bucket_counts['accepted_silver_lite']} "
        f"rejected={bucket_counts['rejected']} "
        f"weak_baseline={bucket_counts['weak_baseline']}"
    )
    print(f"candidate_source={source_hint} candidates_path={candidates_path}")
    print(f"First-pass local judge JSON extractions: {initial_direct_rows}")
    print(f"First-pass malformed-output salvages: {initial_salvaged_rows}")
    if repaired_rows:
        print(f"Recovered local judge rows via repair attempts: {repaired_rows}")
    if fallback_rows:
        print(f"Used structural fallback judging for {fallback_rows} candidate(s).")
    if errors:
        print(f"Skipped {len(errors)} candidate(s) due to local-judge parse/validation errors.")
        for error in errors:
            print(f"[WARN] {error}")


if __name__ == "__main__":
    main()
