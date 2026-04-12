from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    default_candidates_out_path,
    dump_jsonl,
    extract_json_payload,
    load_jsonl,
    looks_like_candidate_payload,
    normalize_candidate_payload,
    normalize_candidate_source,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import pasted GPT-web candidate outputs from manual_raw/ into canonical candidates JSONL."
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("data/manual/candidates/manifest.jsonl"),
        help="Manifest created by 05a_export_candidate_prompts.py.",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=None,
        help="Output candidates JSONL. Defaults by candidate_source.",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip bad raw files instead of failing the whole import.",
    )
    parser.add_argument(
        "--allow-plain-final-proposal",
        action="store_true",
        help="Allow a raw JSON object that is only a final_proposal. The importer wraps it with a fallback reasoning_trace.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_rows = list(load_jsonl(args.manifest_path))
    if not manifest_rows:
        raise SystemExit(f"No manifest rows found in {args.manifest_path}")

    imported_rows: list[dict[str, object]] = []
    errors: list[str] = []

    inferred_source = normalize_candidate_source(
        manifest_rows[0].get("candidate_source"),
        default="strong",
    )
    out_path = args.out_path or default_candidates_out_path(inferred_source)

    for row in manifest_rows:
        candidate_id = str(row.get("candidate_id", "<missing_candidate_id>"))
        raw_file = Path(str(row.get("raw_file", "")))

        if not raw_file.exists():
            errors.append(f"{candidate_id}: raw file does not exist: {raw_file}")
            continue

        try:
            raw_text = raw_file.read_text(encoding="utf-8")
            payload = extract_json_payload(
                raw_text,
                context=str(raw_file),
                predicate=looks_like_candidate_payload,
            )
            candidate_payload = normalize_candidate_payload(
                payload,
                allow_plain_final_proposal=args.allow_plain_final_proposal,
            )
            candidate_source = normalize_candidate_source(
                row.get("candidate_source"),
                default=inferred_source,
            )

            imported_rows.append(
                {
                    "candidate_id": candidate_id,
                    "task_id": str(row["task_id"]),
                    "candidate_source": candidate_source,
                    "generator_model": str(row.get("generator_model", "manual_gpt_web")),
                    "candidate_rank": int(row["candidate_rank"]),
                    "reasoning_trace": candidate_payload["reasoning_trace"],
                    "final_proposal": candidate_payload["final_proposal"],
                    "raw_text": raw_text.strip(),
                }
            )
        except Exception as exc:
            errors.append(f"{candidate_id}: {exc}")

    if errors and not args.skip_invalid:
        raise SystemExit("Candidate import failed:\n- " + "\n- ".join(errors))

    if not imported_rows:
        raise SystemExit("No valid candidate rows were imported.")

    dump_jsonl(out_path, imported_rows)
    print(f"Imported {len(imported_rows)} candidate rows to {out_path}")
    if errors:
        print(f"Skipped {len(errors)} invalid raw files.")
        for error in errors:
            print(f"[WARN] {error}")


if __name__ == "__main__":
    main()
