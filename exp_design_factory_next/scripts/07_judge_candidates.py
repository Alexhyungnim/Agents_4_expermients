from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def llm_judge_call(candidate: dict[str, Any]) -> dict[str, Any]:
    # Replace with your judge provider call.
    return {
        "resource_consistency": {"score": 2, "reason": "Placeholder"},
        "variable_quality": {"score": 1, "reason": "Placeholder"},
        "control_strategy": {"score": 1, "reason": "Placeholder"},
        "measurement_plan": {"score": 1, "reason": "Placeholder"},
        "analysis_plan": {"score": 1, "reason": "Placeholder"},
        "evidence_grounding": {"score": 2, "reason": "Placeholder"},
        "novelty_or_usefulness": {"score": 1, "reason": "Placeholder"},
    }


def rule_checks(candidate: dict[str, Any]) -> dict[str, bool]:
    proposal = candidate["final_proposal"]
    return {
        "unsupported_equipment": False,
        "missing_controls": len(proposal.get("controls", [])) == 0,
        "empty_replicates": proposal.get("design", {}).get("replicates") in [None, 0, ""],
        "evidence_mismatch": False,
    }


def summarize_verdict(rubric: dict[str, Any], rules: dict[str, bool]) -> tuple[int, bool, str]:
    total = sum(v["score"] for v in rubric.values())
    hard_fail = any(rules.values())
    verdict = "accept_silver" if (total >= 11 and not hard_fail) else "reject"
    return total, hard_fail, verdict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", required=False)
    parser.add_argument("--rubric", required=False)
    args = parser.parse_args()

    cands = Path("data/processed/candidates/candidates.jsonl")
    out = Path("data/processed/judged/judged_candidates.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for cand in load_jsonl(cands):
            rubric = llm_judge_call(cand)
            rules = rule_checks(cand)
            total, hard_fail, verdict = summarize_verdict(rubric, rules)
            row = {
                "candidate_id": cand["candidate_id"],
                "task_id": cand["task_id"],
                "judge_model": "strong_judge_B",
                "rubric": rubric,
                "rule_checks": rules,
                "total_score": total,
                "hard_fail": hard_fail,
                "overall_verdict": verdict,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved judged candidates to {out}")
    print(f"Judge config: {args.judge}")
    print(f"Rubric config: {args.rubric}")


if __name__ == "__main__":
    main()
