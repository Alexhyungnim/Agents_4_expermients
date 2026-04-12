from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    compute_detailed_verdict,
    compute_overall_verdict,
    compute_storage_bucket,
    compute_total_score,
    default_judged_out_path,
    dump_jsonl,
    extract_json_payload,
    infer_candidate_source_from_path,
    load_jsonl,
    looks_like_judge_payload,
    normalize_candidate_record,
    normalize_manual_judge_payload,
    normalize_task_record,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import pasted GPT-web judgment outputs from manual_raw/ into canonical judged_candidates JSONL."
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data/manual/judgments/manifest.jsonl"),
        help="Manifest created by 07a_export_judge_prompts.py.",
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
        help="Input candidates JSONL file that was judged.",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=None,
        help="Output judged JSONL. Defaults by candidate_source.",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip bad raw files instead of failing the whole import.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_rows = list(load_jsonl(args.manifest_path))
    if not manifest_rows:
        raise SystemExit(f"No manifest rows found in {args.manifest_path}")

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

    inferred_source = manifest_rows[0].get("candidate_source", source_hint)
    out_path = args.out_path or default_judged_out_path(str(inferred_source))

    judged_rows: list[dict[str, object]] = []
    errors: list[str] = []

    for manifest_row in manifest_rows:
        candidate_id = str(manifest_row.get("candidate_id", "<missing_candidate_id>"))
        raw_file = Path(str(manifest_row.get("raw_file", "")))

        candidate = candidates.get(candidate_id)
        if candidate is None:
            errors.append(f"{candidate_id}: candidate row not found in {args.candidates_path}")
            continue

        task = tasks.get(candidate["task_id"])
        if task is None:
            errors.append(f"{candidate_id}: task row {candidate['task_id']} not found in {args.tasks_path}")
            continue

        if not raw_file.exists():
            errors.append(f"{candidate_id}: raw file does not exist: {raw_file}")
            continue

        try:
            raw_text = raw_file.read_text(encoding="utf-8")
            payload = extract_json_payload(
                raw_text,
                context=str(raw_file),
                predicate=looks_like_judge_payload,
            )
            normalized = normalize_manual_judge_payload(
                payload,
                candidate=candidate,
                task=task,
            )

            total_score = compute_total_score(normalized["rubric"])
            detailed_verdict = compute_detailed_verdict(total_score, normalized["hard_fail"])
            overall_verdict = compute_overall_verdict(total_score, normalized["hard_fail"])
            storage_bucket = compute_storage_bucket(candidate["candidate_source"], overall_verdict)

            row: dict[str, object] = {
                "candidate_id": candidate["candidate_id"],
                "task_id": candidate["task_id"],
                "candidate_source": candidate["candidate_source"],
                "judge_model": str(manifest_row.get("judge_model", "manual_gpt_web_judge")),
                "judge_contract_version": "judge_v0",
                "rubric": normalized["rubric"],
                "rule_checks": normalized["rule_checks"],
                "hard_fail": normalized["hard_fail"],
                "hard_fail_reasons": normalized["hard_fail_reasons"],
                "failure_tags": normalized["failure_tags"],
                "total_score": total_score,
                "detailed_verdict": detailed_verdict,
                "overall_verdict": overall_verdict,
                "storage_bucket": storage_bucket,
                "summary": normalized["summary"],
            }
            if normalized["overall_reasoning"] is not None:
                row["overall_reasoning"] = normalized["overall_reasoning"]

            judged_rows.append(row)
        except Exception as exc:
            errors.append(f"{candidate_id}: {exc}")

    if errors and not args.skip_invalid:
        raise SystemExit("Judgment import failed:\n- " + "\n- ".join(errors))

    if not judged_rows:
        raise SystemExit("No valid judged rows were imported.")

    dump_jsonl(out_path, judged_rows)
    print(f"Imported {len(judged_rows)} judged rows to {out_path}")
    if errors:
        print(f"Skipped {len(errors)} invalid raw files.")
        for error in errors:
            print(f"[WARN] {error}")


if __name__ == "__main__":
    main()
