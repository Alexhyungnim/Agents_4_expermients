from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Callable, Iterator


FENCED_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
NON_ALNUM_RE = re.compile(r"[^a-zA-Z0-9._-]+")

TASK_REQUIRED_FIELDS = [
    "task_id",
    "paper_id",
    "domain",
    "task_type",
    "goal",
    "available_resources",
    "evidence_chunk_ids",
    "evidence_summary",
]

FINAL_PROPOSAL_REQUIRED_FIELDS = [
    "goal",
    "hypothesis",
    "resources_used",
    "independent_variables",
    "dependent_variables",
    "controls",
    "design",
    "measurement_plan",
    "analysis_plan",
    "feasibility_checks",
    "evidence_used",
]

RUBRIC_DIMENSIONS_V0 = [
    "objective_clarity",
    "factor_response_levels",
    "design_choice_appropriateness",
    "interaction_curvature_awareness",
    "resource_aware_design",
    "execution_feasibility",
    "documentation_rigor",
    "blocking_awareness",
    "iterative_experimentation",
    "bom_compliance",
    "claim_discipline",
]

RULE_CHECK_FIELDS_V0 = [
    "unsupported_equipment",
    "missing_controls",
    "empty_replicates",
    "evidence_mismatch",
]


def load_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return iter(())
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def slugify(value: str) -> str:
    value = value.strip()
    if not value:
        return "empty"
    return NON_ALNUM_RE.sub("_", value).strip("_") or "empty"


def pretty_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def ensure_keys(data: dict[str, Any], required_keys: list[str], *, context: str) -> None:
    missing = [key for key in required_keys if key not in data]
    if missing:
        raise ValueError(f"{context} is missing required keys: {', '.join(missing)}")


def normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple, set)):
        out = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return out
    text = str(value).strip()
    return [text] if text else []


def normalize_candidate_source(value: Any, *, default: str | None = None) -> str:
    raw = str(value).strip().lower() if value is not None else ""
    if not raw and default is not None:
        raw = default

    if raw in {"", "strong", "candidate", "manual_strong"}:
        return "strong"
    if raw in {"weak", "weak_baseline", "weak_rag", "weak_rag_baseline", "baseline"}:
        return "weak_baseline"

    raise ValueError(f"Unsupported candidate_source: {value}")


def infer_candidate_source_from_path(path: Path) -> str:
    return "weak_baseline" if "weak" in path.name.lower() else "strong"


def default_candidates_out_path(candidate_source: str) -> Path:
    if candidate_source == "weak_baseline":
        return Path("data/processed/candidates/weak_rag_candidates.jsonl")
    return Path("data/processed/candidates/candidates.jsonl")


def default_judged_out_path(candidate_source: str) -> Path:
    if candidate_source == "weak_baseline":
        return Path("data/processed/judged/judged_weak_rag_candidates.jsonl")
    return Path("data/processed/judged/judged_candidates.jsonl")


def build_candidate_id(task_id: str, candidate_rank: int, candidate_source: str) -> str:
    if candidate_source == "weak_baseline":
        return f"weak_{task_id}_{candidate_rank:02d}"
    return f"cand_{task_id}_{candidate_rank:02d}"


def normalize_task_record(task: dict[str, Any]) -> dict[str, Any]:
    ensure_keys(task, TASK_REQUIRED_FIELDS, context=f"task {task.get('task_id', '<unknown>')}")
    out = dict(task)

    if not isinstance(out["available_resources"], dict):
        raise ValueError(f"task {out['task_id']} has non-object available_resources")

    resources = dict(out["available_resources"])
    resources["equipment"] = normalize_string_list(resources.get("equipment", []))
    resources["materials"] = normalize_string_list(resources.get("materials", []))
    resources["constraints"] = normalize_string_list(resources.get("constraints", []))
    resources["forbidden_items"] = normalize_string_list(resources.get("forbidden_items", []))

    out["available_resources"] = resources
    out["evidence_chunk_ids"] = [str(x) for x in out.get("evidence_chunk_ids", [])]
    out["task_id"] = str(out["task_id"])
    out["paper_id"] = str(out["paper_id"])
    out["domain"] = str(out["domain"])
    out["task_type"] = str(out["task_type"])
    out["goal"] = str(out["goal"])
    out["evidence_summary"] = str(out["evidence_summary"])
    return out


def _normalize_replicates(value: Any) -> int | None:
    if value in {None, ""}:
        return None
    if isinstance(value, bool):
        raise ValueError("replicates cannot be boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise ValueError("replicates must be an integer or null")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.isdigit():
            return int(text)
    raise ValueError("replicates must be an integer or null")


def looks_like_final_proposal(payload: Any) -> bool:
    return isinstance(payload, dict) and all(key in payload for key in FINAL_PROPOSAL_REQUIRED_FIELDS)


def looks_like_candidate_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if "final_proposal" in payload:
        return True
    return looks_like_final_proposal(payload)


def normalize_candidate_payload(
    payload: Any,
    *,
    allow_plain_final_proposal: bool = True,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("candidate payload must be a JSON object")

    if "final_proposal" not in payload:
        if allow_plain_final_proposal and looks_like_final_proposal(payload):
            payload = {
                "reasoning_trace": {"note": "fallback_from_plain_final_proposal"},
                "final_proposal": payload,
            }
        else:
            raise ValueError("candidate payload must contain final_proposal")

    reasoning_trace = payload.get("reasoning_trace", {})
    if isinstance(reasoning_trace, str):
        reasoning_trace = {"note": reasoning_trace.strip()}
    if not isinstance(reasoning_trace, dict):
        raise ValueError("reasoning_trace must be an object or string")

    final_proposal = payload["final_proposal"]
    if not isinstance(final_proposal, dict):
        raise ValueError("final_proposal must be an object")

    ensure_keys(final_proposal, FINAL_PROPOSAL_REQUIRED_FIELDS, context="final_proposal")

    out = dict(final_proposal)
    out["goal"] = str(out["goal"])
    out["hypothesis"] = str(out["hypothesis"])
    out["resources_used"] = normalize_string_list(out.get("resources_used"))
    out["independent_variables"] = normalize_string_list(out.get("independent_variables"))
    out["dependent_variables"] = normalize_string_list(out.get("dependent_variables"))
    out["controls"] = normalize_string_list(out.get("controls"))
    out["measurement_plan"] = normalize_string_list(out.get("measurement_plan"))
    out["analysis_plan"] = normalize_string_list(out.get("analysis_plan"))
    out["feasibility_checks"] = normalize_string_list(out.get("feasibility_checks"))
    out["evidence_used"] = [str(x) for x in out.get("evidence_used", [])]

    design = out["design"]
    if not isinstance(design, dict):
        raise ValueError("final_proposal.design must be an object")
    if "conditions" not in design or "replicates" not in design or "procedure_outline" not in design:
        raise ValueError("final_proposal.design must contain conditions, replicates, and procedure_outline")

    out["design"] = {
        "conditions": normalize_string_list(design.get("conditions")),
        "replicates": _normalize_replicates(design.get("replicates")),
        "procedure_outline": normalize_string_list(design.get("procedure_outline")),
    }

    return {
        "reasoning_trace": reasoning_trace,
        "final_proposal": out,
    }


def normalize_candidate_record(
    row: dict[str, Any],
    *,
    source_hint: str | None = None,
) -> dict[str, Any]:
    ensure_keys(
        row,
        ["candidate_id", "task_id", "generator_model", "candidate_rank", "final_proposal", "raw_text"],
        context=f"candidate {row.get('candidate_id', '<unknown>')}",
    )
    payload = normalize_candidate_payload(row, allow_plain_final_proposal=False)
    out = {
        "candidate_id": str(row["candidate_id"]),
        "task_id": str(row["task_id"]),
        "candidate_source": normalize_candidate_source(row.get("candidate_source"), default=source_hint or "strong"),
        "generator_model": str(row["generator_model"]),
        "candidate_rank": int(row["candidate_rank"]),
        "reasoning_trace": payload["reasoning_trace"],
        "final_proposal": payload["final_proposal"],
        "raw_text": str(row["raw_text"]).strip(),
    }
    return out


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(v) for v in value]
    return value


def _normalize_quotes(text: str) -> str:
    return (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )


def sanitize_manual_raw_text(raw_text: str) -> str:
    text = raw_text
    if text.startswith("\ufeff"):
        text = text[1:]
    return text.replace("\u2028", "\n").replace("\u2029", "\n")


def _extract_balanced_region(text: str, start_idx: int) -> str | None:
    opening = text[start_idx]
    if opening not in "{[":
        return None
    closing = "}" if opening == "{" else "]"

    depth = 0
    in_string = False
    escape = False

    for idx in range(start_idx, len(text)):
        ch = text[idx]

        if escape:
            escape = False
            continue

        if ch == "\\":
            escape = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == opening:
            depth += 1
        elif ch == closing:
            depth -= 1
            if depth == 0:
                return text[start_idx:idx + 1]
    return None


def iter_json_snippets(raw_text: str) -> list[str]:
    text = raw_text.strip()
    normalized = _normalize_quotes(text)
    snippets: list[str] = []

    for candidate in [text, normalized]:
        if candidate:
            snippets.append(candidate)
        for match in FENCED_BLOCK_RE.finditer(candidate):
            block = match.group(1).strip()
            if block:
                snippets.append(block)
        for idx, ch in enumerate(candidate):
            if ch in "{[":
                block = _extract_balanced_region(candidate, idx)
                if block:
                    snippets.append(block.strip())

    deduped: list[str] = []
    seen: set[str] = set()
    for snippet in snippets:
        if snippet and snippet not in seen:
            deduped.append(snippet)
            seen.add(snippet)
    return deduped


def parse_json_loose(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        value = ast.literal_eval(text)
        return _to_json_safe(value)


def extract_json_payload(
    raw_text: str,
    *,
    context: str,
    predicate: Callable[[Any], bool] | None = None,
) -> Any:
    sanitized_text = sanitize_manual_raw_text(raw_text)
    snippets = iter_json_snippets(sanitized_text)
    parse_errors: list[str] = []

    for snippet in snippets:
        try:
            payload = parse_json_loose(snippet)
        except Exception as exc:
            parse_errors.append(str(exc))
            continue

        candidates = [payload]
        if isinstance(payload, list):
            candidates = payload

        for candidate in candidates:
            if predicate is None or predicate(candidate):
                return candidate

    details = "; ".join(parse_errors[:3]) if parse_errors else "no parseable JSON blocks found"
    raise ValueError(f"Could not extract a matching JSON payload from {context}: {details}")


def build_candidate_prompt(task: dict[str, Any], *, candidate_rank: int, n_candidates: int) -> str:
    task_json = pretty_json(task)
    return f"""You are an engineering experiment design assistant.

Generate exactly one experiment proposal candidate.

Candidate variant target: {candidate_rank} of {n_candidates}
If you are generating several variants for this same task across repeated runs, make this candidate meaningfully distinct while staying feasible.

Task record:
{task_json}

Return JSON only. No markdown fences. No explanation before or after the JSON object.

Use exactly this schema:
{{
  "reasoning_trace": {{
    "resource_check": "",
    "variable_mapping": "",
    "control_strategy": "",
    "measurement_strategy": "",
    "risk_check": ""
  }},
  "final_proposal": {{
    "goal": "",
    "hypothesis": "",
    "resources_used": [],
    "independent_variables": [],
    "dependent_variables": [],
    "controls": [],
    "design": {{
      "conditions": [],
      "replicates": 3,
      "procedure_outline": []
    }},
    "measurement_plan": [],
    "analysis_plan": [],
    "feasibility_checks": [],
    "evidence_used": []
  }}
}}

Rules:
- Use only listed resources and evidence-supported resources.
- Do not use forbidden items.
- Keep the first experiment narrow and executable.
- `final_proposal.evidence_used` should reference task `evidence_chunk_ids` when possible.
- `design.replicates` must be an integer or null.
""".strip()


def build_judge_prompt(task: dict[str, Any], candidate: dict[str, Any]) -> str:
    task_json = pretty_json(task)
    candidate_json = pretty_json(candidate["final_proposal"])

    rubric_schema_lines = []
    for key in RUBRIC_DIMENSIONS_V0:
        rubric_schema_lines.append(f'    "{key}": {{"score": 0, "reason": ""}}')
    rubric_schema = ",\n".join(rubric_schema_lines)

    return f"""You are a strict evaluator for engineering experiment proposals.

Evaluate the proposal only against the provided task resources, constraints, and evidence context.
Do not assume missing capabilities.
If an important part is not stated, treat it as missing.

Task record:
{task_json}

Candidate proposal:
{candidate_json}

Score the proposal on these rubric dimensions:
- objective_clarity
- factor_response_levels
- design_choice_appropriateness
- interaction_curvature_awareness
- resource_aware_design
- execution_feasibility
- documentation_rigor
- blocking_awareness
- iterative_experimentation
- bom_compliance
- claim_discipline

Rule checks:
- unsupported_equipment
- missing_controls
- empty_replicates
- evidence_mismatch

Hard-fail if one or more of these are true:
- the objective is unclear
- factors and responses are not clearly specified
- the design does not match the stated goal
- the experiment is not feasible under the provided setup
- the proposal depends on forbidden items or missing capabilities
- the proposal is only a loose parameter sweep with no meaningful design logic

Return JSON only. No markdown fences.
Do not compute total_score or overall_verdict. The importer will compute canonical verdict fields.

Use exactly this schema:
{{
  "hard_fail": false,
  "hard_fail_reasons": [],
  "rubric": {{
{rubric_schema}
  }},
  "rule_checks": {{
    "unsupported_equipment": false,
    "missing_controls": false,
    "empty_replicates": false,
    "evidence_mismatch": false
  }},
  "failure_tags": [],
  "overall_reasoning": {{
    "strengths": [],
    "weaknesses": [],
    "main_concerns": []
  }},
  "summary": ""
}}
""".strip()


def looks_like_judge_payload(payload: Any) -> bool:
    return isinstance(payload, dict) and ("rubric" in payload or "scores" in payload)


def _normalize_rubric_entry(value: Any, *, key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"rubric[{key}] must be an object")
    if "score" not in value:
        raise ValueError(f"rubric[{key}] is missing score")
    score = value["score"]
    if isinstance(score, str) and score.strip().isdigit():
        score = int(score.strip())
    if not isinstance(score, int) or score not in {0, 1, 2}:
        raise ValueError(f"rubric[{key}].score must be 0, 1, or 2")
    reason = str(value.get("reason", "")).strip()
    return {"score": score, "reason": reason}


def normalize_rubric_v0(raw_payload: dict[str, Any]) -> dict[str, Any]:
    rubric = raw_payload.get("rubric", raw_payload.get("scores"))
    if not isinstance(rubric, dict):
        raise ValueError("judge payload must contain rubric or scores object")

    out: dict[str, Any] = {}
    missing = [key for key in RUBRIC_DIMENSIONS_V0 if key not in rubric]
    if missing:
        raise ValueError(f"judge rubric is missing required dimensions: {', '.join(missing)}")

    for key in RUBRIC_DIMENSIONS_V0:
        out[key] = _normalize_rubric_entry(rubric[key], key=key)
    return out


def compute_rule_checks(candidate: dict[str, Any], task: dict[str, Any]) -> dict[str, bool]:
    proposal = candidate["final_proposal"]
    resources = task.get("available_resources", {})

    allowed_resources = {
        item.strip().lower()
        for item in normalize_string_list(resources.get("equipment", [])) + normalize_string_list(resources.get("materials", []))
        if item.strip()
    }
    forbidden_resources = {
        item.strip().lower()
        for item in normalize_string_list(resources.get("forbidden_items", []))
        if item.strip()
    }
    used_resources = {
        item.strip().lower()
        for item in normalize_string_list(proposal.get("resources_used", []))
        if item.strip()
    }
    evidence_used = {str(x) for x in proposal.get("evidence_used", [])}
    known_evidence = {str(x) for x in task.get("evidence_chunk_ids", [])}

    unsupported_equipment = False
    if allowed_resources and used_resources:
        unsupported_equipment = any(
            item not in allowed_resources or item in forbidden_resources
            for item in used_resources
        )

    return {
        "unsupported_equipment": unsupported_equipment,
        "missing_controls": len(normalize_string_list(proposal.get("controls", []))) == 0,
        "empty_replicates": proposal.get("design", {}).get("replicates") in {None, 0, ""},
        "evidence_mismatch": bool(evidence_used and known_evidence and not evidence_used.issubset(known_evidence)),
    }


def normalize_rule_checks(
    raw_rule_checks: Any,
    *,
    candidate: dict[str, Any],
    task: dict[str, Any],
) -> dict[str, bool]:
    fallback = compute_rule_checks(candidate, task)
    if raw_rule_checks is None:
        return fallback
    if not isinstance(raw_rule_checks, dict):
        raise ValueError("rule_checks must be an object")

    out: dict[str, bool] = {}
    for key in RULE_CHECK_FIELDS_V0:
        if key in raw_rule_checks:
            out[key] = bool(raw_rule_checks[key])
        else:
            out[key] = fallback[key]
    return out


def normalize_reasoning_summary(raw_payload: dict[str, Any]) -> tuple[list[str], list[str], dict[str, Any], str]:
    hard_fail_reasons = normalize_string_list(raw_payload.get("hard_fail_reasons", []))
    failure_tags = normalize_string_list(raw_payload.get("failure_tags", []))

    overall_reasoning = raw_payload.get("overall_reasoning")
    if overall_reasoning is None:
        normalized_reasoning = {
            "strengths": [],
            "weaknesses": [],
            "main_concerns": [],
        }
    else:
        if not isinstance(overall_reasoning, dict):
            raise ValueError("overall_reasoning must be an object")
        normalized_reasoning = {
            "strengths": normalize_string_list(overall_reasoning.get("strengths", [])),
            "weaknesses": normalize_string_list(overall_reasoning.get("weaknesses", [])),
            "main_concerns": normalize_string_list(overall_reasoning.get("main_concerns", [])),
        }

    summary = str(raw_payload.get("summary", "")).strip()
    if not summary:
        summary = "Manual judgment imported without summary."

    return hard_fail_reasons, failure_tags, normalized_reasoning, summary


def _merge_unique_strings(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


def derive_hard_fail(
    rubric: dict[str, Any],
    rule_checks: dict[str, bool],
    *,
    hard_fail: Any = None,
    hard_fail_reasons: list[str] | None = None,
) -> tuple[bool, list[str]]:
    reasons = list(hard_fail_reasons or [])

    if rubric["objective_clarity"]["score"] == 0:
        reasons.append("unclear_objective")
    if rubric["factor_response_levels"]["score"] == 0:
        reasons.append("missing_factor_response_structure")
    if rubric["design_choice_appropriateness"]["score"] == 0:
        reasons.append("poor_design_match")
    if rubric["execution_feasibility"]["score"] == 0:
        reasons.append("infeasible_execution")
    if rubric["bom_compliance"]["score"] == 0:
        reasons.append("bom_violation")
    if rule_checks["unsupported_equipment"]:
        reasons.append("infeasible_execution")
    if rule_checks["evidence_mismatch"]:
        reasons.append("evidence_mismatch")

    merged = _merge_unique_strings(reasons)
    hard_fail_bool = bool(hard_fail) or bool(merged)
    return hard_fail_bool, merged


def compute_total_score(rubric: dict[str, Any]) -> int:
    return sum(int(rubric[key]["score"]) for key in RUBRIC_DIMENSIONS_V0)


def compute_detailed_verdict(total_score: int, hard_fail: bool) -> str:
    if hard_fail:
        return "fail"
    if total_score <= 8:
        return "fail"
    if total_score <= 13:
        return "plausible_idea_only"
    if total_score <= 18:
        return "weak_proposal"
    if total_score <= 22:
        return "usable_with_revisions"
    if total_score <= 26:
        return "strong_proposal"
    return "rigorous"


def compute_overall_verdict(total_score: int, hard_fail: bool) -> str:
    return "accept_silver" if total_score >= 11 and not hard_fail else "reject"


def compute_storage_bucket(candidate_source: str, overall_verdict: str) -> str:
    if candidate_source == "weak_baseline":
        return "weak_baseline"
    return "accepted_silver" if overall_verdict == "accept_silver" else "rejected"


def normalize_manual_judge_payload(
    payload: Any,
    *,
    candidate: dict[str, Any],
    task: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("judge payload must be a JSON object")

    rubric = normalize_rubric_v0(payload)
    rule_checks = normalize_rule_checks(payload.get("rule_checks"), candidate=candidate, task=task)
    hard_fail_reasons, failure_tags, overall_reasoning, summary = normalize_reasoning_summary(payload)
    hard_fail, hard_fail_reasons = derive_hard_fail(
        rubric,
        rule_checks,
        hard_fail=payload.get("hard_fail"),
        hard_fail_reasons=hard_fail_reasons,
    )

    return {
        "rubric": rubric,
        "rule_checks": rule_checks,
        "hard_fail": hard_fail,
        "hard_fail_reasons": hard_fail_reasons,
        "failure_tags": _merge_unique_strings(failure_tags),
        "overall_reasoning": overall_reasoning,
        "summary": summary,
    }
