from __future__ import annotations

import ast
import json
import re
from copy import deepcopy
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


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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


def build_candidate_id_with_label(
    task_id: str,
    origin_label: str,
    candidate_rank: int,
    *,
    candidate_source: str,
) -> str:
    prefix = "weak" if candidate_source == "weak_baseline" else "cand"
    return f"{prefix}_{task_id}_{slugify(origin_label)}_{candidate_rank:02d}"


def upsert_jsonl_rows(path: Path, rows: list[dict[str, Any]], *, key_field: str) -> None:
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for existing_row in load_jsonl(path):
        key = str(existing_row[key_field])
        merged[key] = existing_row
        order.append(key)

    for row in rows:
        key = str(row[key_field])
        if key not in merged:
            order.append(key)
        merged[key] = row

    dump_jsonl(path, [merged[key] for key in order])


def load_merged_jsonl_rows(paths: list[Path], *, key_field: str) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for path in paths:
        for row in load_jsonl(path):
            key = str(row[key_field])
            if key not in merged:
                order.append(key)
            merged[key] = row

    return [merged[key] for key in order]


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


def parse_labeled_sections(text: str, labels: list[str]) -> dict[str, str]:
    pattern = re.compile(
        r"^(%s):\s*(.*)$" % "|".join(re.escape(label) for label in labels),
        re.MULTILINE,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return {}

    sections: dict[str, str] = {}
    for idx, match in enumerate(matches):
        label = match.group(1)
        inline_value = match.group(2).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        content = inline_value
        if block:
            content = f"{inline_value}\n{block}".strip() if inline_value else block
        sections[label] = content.strip()
    return sections


def text_block_to_items(text: str) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []

    bullet_lines = []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("- "):
            bullet_lines.append(line[2:].strip())
        elif line.startswith("* "):
            bullet_lines.append(line[2:].strip())

    if bullet_lines:
        return [x for x in bullet_lines if x]

    parts = re.split(r"[\n;,]+", raw)
    items = [part.strip(" -") for part in parts if part.strip(" -")]
    return items


def maybe_read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_local_hf_model_path(model_name: str) -> str:
    direct_path = Path(model_name).expanduser()
    if direct_path.exists():
        return str(direct_path)

    repo_cache_dir = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"models--{model_name.replace('/', '--')}"
    )
    if not repo_cache_dir.exists():
        return model_name

    refs_main = repo_cache_dir / "refs" / "main"
    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        if revision:
            snapshot_dir = repo_cache_dir / "snapshots" / revision
            if snapshot_dir.exists():
                return str(snapshot_dir)

    snapshot_roots = sorted(
        path for path in (repo_cache_dir / "snapshots").iterdir()
        if path.is_dir()
    ) if (repo_cache_dir / "snapshots").exists() else []
    for snapshot_dir in snapshot_roots:
        if (snapshot_dir / "config.json").exists():
            return str(snapshot_dir)

    return model_name


def resolve_task_for_candidate_source(
    tasks: dict[str, dict[str, Any]],
    *,
    task_id: str | None = None,
    goal_text: str | None = None,
) -> dict[str, Any]:
    if task_id is not None:
        if task_id not in tasks:
            raise ValueError(f"Requested task_id not found: {task_id}")
        return tasks[task_id]

    if len(tasks) == 1:
        return next(iter(tasks.values()))

    if goal_text:
        goal_text_norm = str(goal_text).strip().lower()
        matches = [
            task for task in tasks.values()
            if str(task.get("goal", "")).strip().lower() == goal_text_norm
        ]
        if len(matches) == 1:
            return matches[0]

    raise ValueError(
        "Could not resolve a unique task. Pass --task-id explicitly or ensure tasks.jsonl has one row."
    )


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


class LocalTransformersLLM:
    def __init__(self, model_name: str, *, trust_remote_code: bool = True):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local LLM generation requires transformers and torch to be installed."
            ) from exc

        self._torch = torch
        self.model_name = model_name
        resolved_model_path = resolve_local_hf_model_path(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            resolved_model_path,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            resolved_model_path,
            local_files_only=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def _build_prompt_text(self, prompt: str) -> str:
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                return prompt
        return prompt

    def complete(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 900,
        temperature: float = 0.2,
    ) -> str:
        prompt_text = self._build_prompt_text(prompt)
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        do_sample = temperature > 0
        generate_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": 1.05,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = 0.95

        with self._torch.no_grad():
            outputs = self.model.generate(**generate_kwargs)

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()


def build_judge_prompt(task: dict[str, Any], candidate: dict[str, Any]) -> str:
    task_json = pretty_json(task)
    candidate_json = pretty_json(candidate["final_proposal"])

    rubric_schema_lines = []
    for key in RUBRIC_DIMENSIONS_V0:
        rubric_schema_lines.append(f'    "{key}": {{"score": 1, "reason": ""}}')
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

Use a 1 to 5 scale for each rubric item:
- 1 or 2 = weak / missing / not acceptable
- 3 = partial / mixed
- 4 or 5 = strong / clearly present

The importer will automatically derive the compatibility 0 to 2 scale used by the current accept/reject pipeline.

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


def build_local_judge_prompt(
    task: dict[str, Any],
    candidate: dict[str, Any],
    *,
    rule_checks: dict[str, bool] | None = None,
) -> str:
    resources = task.get("available_resources", {})
    proposal = candidate["final_proposal"]
    rule_checks = rule_checks or {}

    task_snapshot = {
        "goal": str(task.get("goal", "")).strip(),
        "equipment": normalize_string_list(resources.get("equipment", [])),
        "materials": normalize_string_list(resources.get("materials", [])),
        "constraints": normalize_string_list(resources.get("constraints", [])),
        "forbidden_items": normalize_string_list(resources.get("forbidden_items", [])),
        "evidence_chunk_ids": [str(x) for x in task.get("evidence_chunk_ids", [])],
    }
    candidate_snapshot = {
        "goal": str(proposal.get("goal", "")).strip(),
        "hypothesis": str(proposal.get("hypothesis", "")).strip(),
        "resources_used": normalize_string_list(proposal.get("resources_used", [])),
        "independent_variables": normalize_string_list(proposal.get("independent_variables", [])),
        "dependent_variables": normalize_string_list(proposal.get("dependent_variables", [])),
        "controls": normalize_string_list(proposal.get("controls", [])),
        "conditions": normalize_string_list(proposal.get("design", {}).get("conditions", [])),
        "replicates": proposal.get("design", {}).get("replicates"),
        "measurement_plan": normalize_string_list(proposal.get("measurement_plan", [])),
        "analysis_plan": normalize_string_list(proposal.get("analysis_plan", [])),
        "feasibility_checks": normalize_string_list(proposal.get("feasibility_checks", [])),
        "evidence_used": [str(x) for x in proposal.get("evidence_used", [])],
    }
    compact_rubric = ", ".join(f'"{key}": 1' for key in RUBRIC_DIMENSIONS_V0)

    return f"""Return one valid JSON object only. No markdown. No explanation.

Judge this experiment proposal against the task resources and constraints only.
Use integer rubric scores only.
Do not include per-dimension reasons.
Do not output rule_checks.
Keep summary under 18 words.

Scoring scale:
- 1 or 2 = weak or missing
- 3 = partial
- 4 or 5 = strong

Rubric meaning:
- objective_clarity = clear goal and hypothesis
- factor_response_levels = IV, DV, conditions, replicates
- design_choice_appropriateness = design matches the goal
- interaction_curvature_awareness = enough conditions for trend or curvature learning
- resource_aware_design = narrow and realistic for listed resources
- execution_feasibility = executable with listed setup
- documentation_rigor = procedure, measurements, and analysis are specified
- blocking_awareness = controls, fixed settings, and risks are acknowledged
- iterative_experimentation = good first-pass experiment
- bom_compliance = no unsupported or forbidden items
- claim_discipline = evidence-grounded and restrained claims

Task:
{json.dumps(task_snapshot, ensure_ascii=False)}

Candidate:
{json.dumps(candidate_snapshot, ensure_ascii=False)}

Rule hints:
{json.dumps(rule_checks, ensure_ascii=False)}

Return exactly this shape:
{{"hard_fail": false, "hard_fail_reasons": [], "failure_tags": [], "rubric": {{{compact_rubric}}}, "summary": ""}}
""".strip()


def build_local_judge_repair_prompt(raw_text: str) -> str:
    compact_rubric = ", ".join(f'"{key}": 1' for key in RUBRIC_DIMENSIONS_V0)
    clipped = sanitize_manual_raw_text(raw_text).strip()[:2400]
    return f"""Rewrite the malformed output below into one valid JSON object only.

Rules:
- keep the same judgment meaning if possible
- rubric values must be integers 1 to 5
- do not add explanation outside the JSON
- keep summary under 18 words

Required shape:
{{"hard_fail": false, "hard_fail_reasons": [], "failure_tags": [], "rubric": {{{compact_rubric}}}, "summary": ""}}

Malformed output:
{clipped}
""".strip()


RAG1_SECTION_LABELS = [
    "Candidate Experiment",
    "Why This Is Feasible With Current BOM",
    "Borrowed Literature Precedents",
    "New Adaptation / Novel Twist",
    "Needed Equipment",
    "Needed Materials / Consumables",
    "Key Process Parameters To Sweep",
    "Measurements / Outputs",
    "Main Risk / Failure Mode",
    "Missing Capability / Assumption",
    "Source Papers Used",
]

LOCAL_CANDIDATE_OUTLINE_LABELS = [
    "Goal",
    "Hypothesis",
    "Resources Used",
    "Independent Variables",
    "Dependent Variables",
    "Controls",
    "Conditions",
    "Replicates",
    "Procedure Outline",
    "Measurement Plan",
    "Analysis Plan",
    "Feasibility Checks",
    "Evidence Used",
]


def _parse_first_int(text: str) -> int | None:
    match = re.search(r"\d+", str(text or ""))
    if not match:
        return None
    return int(match.group(0))


def _split_variable_phrase(text: str) -> list[str]:
    cleaned = str(text or "").strip(" .")
    if not cleaned:
        return []

    parts = [
        part.strip(" .")
        for part in re.split(r"\s*(?:,| and )\s*", cleaned)
        if part.strip(" .")
    ]
    return parts or [cleaned]


def _infer_task_variables(goal: str) -> tuple[list[str], list[str]]:
    text = str(goal or "").strip()
    if not text:
        return [], []

    patterns = [
        re.compile(r"study how\s+(?P<iv>.+?)\s+affects\s+(?P<dv>.+?)(?:\s+in\s+|\s+for\s+|$)", re.IGNORECASE),
        re.compile(r"how\s+(?P<iv>.+?)\s+affects\s+(?P<dv>.+?)(?:\s+in\s+|\s+for\s+|$)", re.IGNORECASE),
        re.compile(r"effect of\s+(?P<iv>.+?)\s+on\s+(?P<dv>.+?)(?:\s+in\s+|\s+for\s+|$)", re.IGNORECASE),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return _split_variable_phrase(match.group("iv")), _split_variable_phrase(match.group("dv"))
    return [], []


def _match_allowed_items(items: list[str], allowed_items: list[str]) -> list[str]:
    allowed_map = {item.strip().lower(): item for item in allowed_items if item.strip()}
    matched: list[str] = []
    for item in items:
        normalized = item.strip().lower()
        if normalized in allowed_map:
            matched.append(allowed_map[normalized])
    return _merge_unique_strings(matched)


def _conditions_have_explicit_numbers(conditions: list[str]) -> bool:
    return bool(conditions) and all(re.search(r"\d", condition) for condition in conditions)


def detect_measurement_mode(
    task: dict[str, Any],
    dependent_variables: list[str],
) -> str:
    resources = task.get("available_resources", {})
    equipment = normalize_string_list(resources.get("equipment", []))
    dv_text = " ; ".join(dependent_variables).lower()
    equipment_lower = {item.lower() for item in equipment}

    if "roughness" in dv_text or "profilometer" in " ".join(equipment_lower):
        return "profilometry"
    if "density" in dv_text or ("archimedes density kit" in equipment_lower and "analytical balance" in equipment_lower):
        return "density"
    if "hardness" in dv_text or "microhardness" in dv_text or "microhardness tester" in equipment_lower:
        return "microhardness"
    if any(keyword in dv_text for keyword in ["phase", "austenite", "martensite"]) or "xrd" in equipment_lower:
        return "xrd_phase"
    if any(keyword in dv_text for keyword in ["width", "height", "track", "bead"]) and (
        "optical microscope" in equipment_lower or "digital calipers" in equipment_lower
    ):
        return "geometry"
    if any(keyword in dv_text for keyword in ["porosity", "crack", "lack-of-fusion", "fusion"]) or "sem" in equipment_lower:
        return "sem_imaging"
    return "generic"


def build_explicit_numeric_condition_levels(
    task: dict[str, Any],
    independent_variables: list[str],
    *,
    wider: bool = False,
) -> list[str]:
    default_ivs, _ = _infer_task_variables(str(task.get("goal", "")))
    iv = (independent_variables or default_ivs or ["main process setting"])[0]
    iv_lower = iv.lower()

    if "power" in iv_lower:
        percentages = [88, 96, 104, 112] if wider else [90, 100, 110]
        return [
            f"Laser power = {pct}% of the validated midpoint setpoint"
            for pct in percentages
        ]

    if "temperature" in iv_lower:
        percentages = [95, 100, 105, 110] if wider else [97, 100, 103]
        return [
            f"{iv} = {pct}% of the validated midpoint value"
            for pct in percentages
        ]

    percentages = [85, 95, 105, 115] if wider else [90, 100, 110]
    return [
        f"{iv} = {pct}% of the validated baseline value"
        for pct in percentages
    ]


def _pick_first_matching(items: list[str], needles: list[str]) -> list[str]:
    matches: list[str] = []
    lowered = [(item, item.lower()) for item in items]
    for needle in needles:
        for original, lowered_item in lowered:
            if needle in lowered_item:
                matches.append(original)
                break
    return _merge_unique_strings(matches)


def _build_safe_baseline_resources(task: dict[str, Any], *, measurement_mode: str) -> list[str]:
    resources = task.get("available_resources", {})
    equipment = normalize_string_list(resources.get("equipment", []))
    materials = normalize_string_list(resources.get("materials", []))
    mode_equipment_map = {
        "sem_imaging": ["ded machine", "sem", "precision saw", "mounting press", "polishing"],
        "geometry": ["ded machine", "optical microscope", "digital calipers"],
        "xrd_phase": ["ded machine", "vacuum furnace", "xrd"],
        "microhardness": ["ded machine", "microhardness tester", "precision saw", "mounting press", "polishing"],
        "density": ["ded machine", "analytical balance", "archimedes density kit"],
        "profilometry": ["ded machine", "optical profilometer"],
        "generic": ["ded machine"],
    }
    preferred_equipment = _pick_first_matching(
        equipment,
        mode_equipment_map.get(measurement_mode, mode_equipment_map["generic"]),
    )
    if not preferred_equipment:
        preferred_equipment = equipment[:4]

    material_needles = ["powder", "argon", "resin"]
    if measurement_mode == "density":
        material_needles.append("ethanol")
    preferred_materials = _pick_first_matching(materials, material_needles)
    if not preferred_materials:
        preferred_materials = materials[:3]

    return _merge_unique_strings(preferred_equipment + preferred_materials)


def build_task_specific_baseline_plan(
    task: dict[str, Any],
    *,
    independent_variables: list[str],
    dependent_variables: list[str],
) -> dict[str, Any]:
    resources = task.get("available_resources", {})
    equipment = normalize_string_list(resources.get("equipment", []))
    equipment_lower = {item.lower() for item in equipment}
    iv = (independent_variables or ["main process setting"])[0]
    dv_text = " and ".join(dependent_variables or ["main response"])
    iv_lower = iv.lower()
    measurement_mode = detect_measurement_mode(task, dependent_variables)
    relevant_resources = _build_safe_baseline_resources(task, measurement_mode=measurement_mode)

    if "vacuum furnace" in equipment_lower and "temperature" in iv_lower:
        first_step = (
            f"Produce one fixed baseline set of DED NiTi coupons, then apply each explicit numeric {iv_lower} condition in the vacuum furnace."
        )
        fixed_step = "Keep the as-built coupon geometry and baseline DED settings unchanged before heat treatment."
    else:
        first_step = (
            f"Run one DED coupon at each explicit numeric {iv_lower} condition within the validated stable operating window."
        )
        fixed_step = f"Keep all non-{iv_lower} process settings fixed across the full screening run."

    controls = [fixed_step]
    procedure_outline = [first_step]
    measurement_plan: list[str] = [f"Record the exact numeric {iv_lower} level and replicate identifier for every sample."]
    analysis_plan: list[str] = []
    feasibility_checks = ["Uses only listed equipment and materials."]

    if measurement_mode == "sem_imaging":
        controls.extend(
            [
                "Use the same powder lot, argon shielding setup, and sample geometry for every run.",
                "Prepare every sample with the same sectioning, mounting, grinding, and polishing workflow before SEM imaging.",
            ]
        )
        procedure_outline.extend(
            [
                "Section each build with the precision saw, then mount, grind, and polish for cross-section imaging.",
                f"Acquire SEM cross-sections for every replicate and measure {dv_text.lower()} from the prepared sections.",
            ]
        )
        measurement_plan.extend(
            [
                f"Quantify {dv_text.lower()} from SEM cross-sections using one consistent image-analysis workflow.",
                f"Summarize {dv_text.lower()} by condition before comparing the screened settings.",
            ]
        )
        analysis_plan.extend(
            [
                f"Compute the mean and spread of {dv_text.lower()} for each numeric {iv_lower} level.",
                f"Plot {dv_text.lower()} versus {iv_lower} to check for a monotonic or non-linear trend.",
            ]
        )
        feasibility_checks.append("Uses SEM-based imaging with the listed metallography tools.")
    elif measurement_mode == "geometry":
        controls.extend(
            [
                "Use the same powder lot, argon shielding setup, and coupon geometry for every run.",
                "Measure every coupon with the same optical-microscopy setup and caliper workflow.",
            ]
        )
        procedure_outline.extend(
            [
                "Measure track width from optical micrographs after each run.",
                "Measure build height from the same coupons with digital calipers.",
            ]
        )
        measurement_plan.extend(
            [
                "Capture optical images for track-width measurement at the same magnification for every coupon.",
                "Record build height with digital calipers for every replicate.",
                f"Summarize {dv_text.lower()} by condition before comparing the screened settings.",
            ]
        )
        analysis_plan.extend(
            [
                f"Compare the paired geometric responses across the numeric {iv_lower} levels.",
                "Identify whether the same setting range improves both responses or creates a tradeoff.",
            ]
        )
        feasibility_checks.append("Uses optical microscopy and caliper measurements only.")
    elif measurement_mode == "xrd_phase":
        controls.extend(
            [
                "Use coupons built with one fixed baseline DED setting set before heat treatment.",
                "Use the same XRD scan setup and specimen preparation workflow for every condition.",
            ]
        )
        procedure_outline.extend(
            [
                "Heat treat coupons at each explicit numeric temperature condition in the vacuum furnace.",
                f"Run XRD on every condition and quantify {dv_text.lower()} from the diffraction data.",
            ]
        )
        measurement_plan.extend(
            [
                "Collect XRD scans with one fixed scan range and step condition for every sample.",
                f"Estimate {dv_text.lower()} with one consistent peak-based analysis workflow.",
            ]
        )
        analysis_plan.extend(
            [
                f"Compare the estimated {dv_text.lower()} across the temperature conditions.",
                "Use the first-pass result to identify whether a narrower temperature window is worth testing next.",
            ]
        )
        feasibility_checks.append("Uses only the listed vacuum furnace and XRD workflow.")
    elif measurement_mode == "microhardness":
        controls.extend(
            [
                "Use the same powder lot, shielding setup, and coupon geometry for every run.",
                "Prepare every cross-section with the same metallography workflow before hardness testing.",
            ]
        )
        procedure_outline.extend(
            [
                "Section, mount, grind, and polish each coupon after deposition.",
                f"Measure {dv_text.lower()} on every prepared sample with the microhardness tester.",
            ]
        )
        measurement_plan.extend(
            [
                "Use the same indentation load and spacing for every hardness map.",
                f"Summarize {dv_text.lower()} by condition before comparing the screened settings.",
            ]
        )
        analysis_plan.extend(
            [
                f"Compute the mean and spread of {dv_text.lower()} for each numeric {iv_lower} level.",
                "Look for the smallest setting change that produces a stable directional shift in hardness.",
            ]
        )
        feasibility_checks.append("Uses only the listed hardness and sample-preparation tools.")
    elif measurement_mode == "density":
        controls.extend(
            [
                "Use the same powder lot, shielding setup, and coupon geometry for every run.",
                "Use one Archimedes setup and fluid handling workflow for every density measurement.",
            ]
        )
        procedure_outline.extend(
            [
                "Produce coupons at each explicit numeric condition inside the stable deposition window.",
                f"Measure {dv_text.lower()} for every coupon with the analytical balance and Archimedes density kit.",
            ]
        )
        measurement_plan.extend(
            [
                "Measure dry and immersed mass for every replicate with the same balance and density-kit setup.",
                f"Convert the raw measurements into {dv_text.lower()} with one consistent calculation sheet.",
            ]
        )
        analysis_plan.extend(
            [
                f"Compare the mean {dv_text.lower()} across the screened settings.",
                "Flag any condition that improves density without immediately widening the experimental scope.",
            ]
        )
        feasibility_checks.append("Uses only the listed analytical balance and Archimedes density kit.")
    elif measurement_mode == "profilometry":
        controls.extend(
            [
                "Use the same powder lot, shielding setup, and coupon geometry for every run.",
                "Measure every coupon with the same optical-profilometer scan settings.",
            ]
        )
        procedure_outline.extend(
            [
                "Produce coupons at each explicit numeric condition inside the stable deposition window.",
                f"Measure {dv_text.lower()} for every coupon with the optical profilometer.",
            ]
        )
        measurement_plan.extend(
            [
                "Use the same scan length and filtering settings for every profilometer trace.",
                f"Summarize {dv_text.lower()} by condition before comparing the screened settings.",
            ]
        )
        analysis_plan.extend(
            [
                f"Compare the mean and spread of {dv_text.lower()} across the screened settings.",
                "Use the first-pass result to decide whether a narrower surface-finish study is justified.",
            ]
        )
        feasibility_checks.append("Uses only the listed optical profilometer workflow.")
    else:
        controls.extend(
            [
                f"Keep all non-{iv_lower} settings fixed across all conditions.",
                "Use one consistent measurement workflow across the full screening set.",
            ]
        )
        procedure_outline.append("Measure the target response with the listed equipment after each run.")
        measurement_plan.append(f"Summarize {dv_text.lower()} by condition before comparing the screened settings.")
        analysis_plan.append(f"Compare {dv_text.lower()} across the numeric {iv_lower} levels.")
        feasibility_checks.append("Keeps the experiment narrow and executable with listed resources.")

    feasibility_checks.append(
        f"Uses explicit numeric {iv_lower} levels with a narrow first-pass experimental scope."
    )
    analysis_plan.append("Use the first-pass result to decide whether a narrower follow-up sweep is justified.")

    return {
        "measurement_mode": measurement_mode,
        "resources_used": relevant_resources,
        "controls": _merge_unique_strings(controls),
        "procedure_outline": _merge_unique_strings(procedure_outline),
        "measurement_plan": _merge_unique_strings(measurement_plan),
        "analysis_plan": _merge_unique_strings(analysis_plan),
        "feasibility_checks": _merge_unique_strings(feasibility_checks),
    }


def build_baseline_safe_candidate_payload(
    task: dict[str, Any],
    *,
    seed_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    existing = deepcopy(seed_payload) if seed_payload is not None else {
        "reasoning_trace": {},
        "final_proposal": {},
    }
    existing_proposal = existing.get("final_proposal", {})
    resources = task.get("available_resources", {})
    equipment = normalize_string_list(resources.get("equipment", []))
    default_ivs, default_dvs = _infer_task_variables(str(task.get("goal", "")))

    independent_variables = normalize_string_list(existing_proposal.get("independent_variables", [])) or default_ivs
    if not independent_variables:
        independent_variables = ["main process setting"]
    dependent_variables = normalize_string_list(existing_proposal.get("dependent_variables", [])) or default_dvs
    if not dependent_variables:
        dependent_variables = ["main response"]

    iv = independent_variables[0]
    numeric_conditions = build_explicit_numeric_condition_levels(task, independent_variables)
    task_plan = build_task_specific_baseline_plan(
        task,
        independent_variables=independent_variables,
        dependent_variables=dependent_variables,
    )
    relevant_resources = task_plan["resources_used"] or equipment[:4]

    evidence_ids = [str(x) for x in task.get("evidence_chunk_ids", []) if str(x).strip()]
    dv_text = " and ".join(dependent_variables)
    controls = task_plan["controls"]
    procedure_outline = task_plan["procedure_outline"]
    measurement_plan = task_plan["measurement_plan"]
    analysis_plan = task_plan["analysis_plan"]
    feasibility_checks = task_plan["feasibility_checks"]

    payload = {
        "reasoning_trace": {
            "resource_check": "; ".join(relevant_resources) or "Uses only listed equipment and materials.",
            "variable_mapping": f"{'; '.join(independent_variables)}; {dv_text}",
            "control_strategy": "; ".join(controls),
            "measurement_strategy": "; ".join(measurement_plan),
            "risk_check": "; ".join(feasibility_checks),
        },
        "final_proposal": {
            "goal": str(task.get("goal", "")).strip() or str(existing_proposal.get("goal", "")).strip(),
            "hypothesis": (
                f"Within the evidence-supported operating window, changing {iv.lower()} will change {dv_text.lower()}."
            ),
            "resources_used": relevant_resources,
            "independent_variables": independent_variables,
            "dependent_variables": dependent_variables,
            "controls": controls,
            "design": {
                "conditions": numeric_conditions,
                "replicates": 3,
                "procedure_outline": procedure_outline,
            },
            "measurement_plan": measurement_plan,
            "analysis_plan": analysis_plan,
            "feasibility_checks": feasibility_checks,
            "evidence_used": evidence_ids,
        },
    }
    return normalize_candidate_payload(payload, allow_plain_final_proposal=False)


def build_strong_contrast_variant_payload(
    task: dict[str, Any],
    baseline_payload: dict[str, Any],
    *,
    candidate_rank: int,
) -> tuple[dict[str, Any], str]:
    payload = deepcopy(baseline_payload)
    proposal = payload["final_proposal"]
    independent_variables = normalize_string_list(proposal.get("independent_variables", []))
    dependent_variables = normalize_string_list(proposal.get("dependent_variables", []))
    iv = (independent_variables or _infer_task_variables(str(task.get("goal", "")))[0] or ["main process setting"])[0]
    dv = " and ".join(
        dependent_variables or _infer_task_variables(str(task.get("goal", "")))[1] or ["main response"]
    )
    measurement_mode = detect_measurement_mode(task, dependent_variables)

    if candidate_rank == 2:
        payload["reasoning_trace"]["risk_check"] = (
            "Feasible contrast variant that widens the numeric screening span while trading off some replication."
        )
        payload["final_proposal"]["controls"] = payload["final_proposal"]["controls"][:2]
        payload["final_proposal"]["design"]["conditions"] = build_explicit_numeric_condition_levels(
            task,
            independent_variables,
            wider=True,
        )
        payload["final_proposal"]["design"]["replicates"] = (
            3 if measurement_mode in {"xrd_phase", "density", "microhardness"} else 2
        )
        payload["final_proposal"]["analysis_plan"] = [
            f"Screen mean {dv.lower()} across the wider numeric {iv.lower()} range.",
            "Use the result to choose a narrower follow-up window.",
        ]
        payload["final_proposal"]["feasibility_checks"] = [
            "Uses only listed equipment and materials.",
            f"Expands the numeric {iv.lower()} span while keeping the plan executable as a first-pass screen.",
            "Retains evidence alignment and core controls.",
        ]
        return normalize_candidate_payload(payload, allow_plain_final_proposal=False), "feasible_span_variant"

    if candidate_rank == 3:
        payload["reasoning_trace"]["control_strategy"] = (
            "Controls intentionally reduced in this contrast variant to preserve diversity against the baseline-safe design."
        )
        payload["reasoning_trace"]["risk_check"] = (
            "Deliberately weaker contrast variant with limited controls and no replication."
        )
        payload["final_proposal"]["controls"] = []
        payload["final_proposal"]["design"]["replicates"] = None
        if measurement_mode in {"geometry", "profilometry"}:
            payload["final_proposal"]["design"]["conditions"] = payload["final_proposal"]["design"]["conditions"][:2]
        payload["final_proposal"]["analysis_plan"] = [
            f"Compare {dv.lower()} qualitatively across the numeric {iv.lower()} conditions."
        ]
        payload["final_proposal"]["feasibility_checks"] = [
            "Fast contrast variant with intentionally weaker controls and replication.",
        ]
        if measurement_mode in {"density", "xrd_phase"}:
            payload["final_proposal"]["evidence_used"] = []
        return normalize_candidate_payload(payload, allow_plain_final_proposal=False), "undercontrolled_variant"

    payload["reasoning_trace"]["control_strategy"] = (
        "Only one control retained in this contrast variant, with reduced justification and no replication."
    )
    payload["reasoning_trace"]["risk_check"] = (
        "Deliberately weak contrast variant with thin evidence and minimal analysis."
    )
    payload["final_proposal"]["controls"] = payload["final_proposal"]["controls"][:1]
    payload["final_proposal"]["design"]["replicates"] = None
    payload["final_proposal"]["design"]["conditions"] = payload["final_proposal"]["design"]["conditions"][:2]
    payload["final_proposal"]["analysis_plan"] = []
    payload["final_proposal"]["feasibility_checks"] = [
        "Minimal contrast variant retained for rejected-pool diversity.",
    ]
    payload["final_proposal"]["evidence_used"] = []
    return normalize_candidate_payload(payload, allow_plain_final_proposal=False), "thin_justification_variant"


def build_local_candidate_outline_prompt(
    task: dict[str, Any],
    *,
    candidate_rank: int,
    n_candidates: int,
) -> str:
    variant_note = (
        "Make one conservative baseline-safe first-pass experiment with explicit numeric factor levels, strong controls, clear measurements, and only listed resources."
        if candidate_rank == 1
        else "Make a contrast variant that stays resource-feasible but is meaningfully different from the conservative baseline."
    )
    task_json = pretty_json(task)
    return f"""You are drafting one experiment proposal outline for local post-processing.

Candidate variant target: {candidate_rank} of {n_candidates}
{variant_note}

Task record:
{task_json}

Return only the labeled outline below. Keep each field short.
Use bullet items for list fields when helpful.
Do not use markdown fences.

Goal:
Hypothesis:
Resources Used:
Independent Variables:
Dependent Variables:
Controls:
Conditions:
Replicates:
Procedure Outline:
Measurement Plan:
Analysis Plan:
Feasibility Checks:
Evidence Used:
""".strip()


def candidate_payload_from_outline_text(
    raw_text: str,
    *,
    task: dict[str, Any],
    candidate_rank: int,
) -> dict[str, Any]:
    sections = parse_labeled_sections(raw_text, LOCAL_CANDIDATE_OUTLINE_LABELS)
    if not sections:
        raise ValueError("Could not parse labeled outline sections from local candidate output")

    available_resources = task.get("available_resources", {})
    allowed_resources = normalize_string_list(available_resources.get("equipment", [])) + normalize_string_list(
        available_resources.get("materials", [])
    )
    default_ivs, default_dvs = _infer_task_variables(str(task.get("goal", "")))

    resources_used = _match_allowed_items(
        text_block_to_items(sections.get("Resources Used", "")),
        allowed_resources,
    )
    if candidate_rank == 1 and not resources_used:
        resources_used = _merge_unique_strings(allowed_resources[:4])

    independent_variables = text_block_to_items(sections.get("Independent Variables", "")) or default_ivs
    dependent_variables = text_block_to_items(sections.get("Dependent Variables", "")) or default_dvs

    controls = [
        item for item in text_block_to_items(sections.get("Controls", ""))
        if item.strip().lower() not in {"none", "n/a", "na", "no control", "no controls"}
    ]
    if candidate_rank == 1 and not controls:
        controls = [
            "Keep all non-target process settings fixed.",
            "Use the same powder lot and shielding gas setup.",
        ]

    conditions = text_block_to_items(sections.get("Conditions", ""))
    if len(conditions) == 1:
        condition_text = conditions[0]
        split_source = condition_text.split(":", 1)[1].strip() if ":" in condition_text else condition_text
        split_conditions = [part.strip(" .") for part in split_source.split(" and ") if part.strip(" .")]
        if len(split_conditions) >= 2:
            conditions = split_conditions
    if candidate_rank == 1 and (not conditions or not _conditions_have_explicit_numbers(conditions)):
        conditions = build_explicit_numeric_condition_levels(task, independent_variables)

    replicates = _parse_first_int(sections.get("Replicates", ""))
    if replicates is None:
        replicates = 3 if candidate_rank == 1 else None

    procedure_outline = text_block_to_items(sections.get("Procedure Outline", ""))
    if candidate_rank == 1 and not procedure_outline:
        procedure_outline = [
            "Prepare samples using the listed process setup.",
            "Run the planned conditions in the stable operating window.",
            "Section and inspect samples with the listed characterization tools.",
        ]

    measurement_plan = text_block_to_items(sections.get("Measurement Plan", ""))
    if candidate_rank == 1 and not measurement_plan:
        measurement_plan = dependent_variables[:]

    analysis_plan = text_block_to_items(sections.get("Analysis Plan", ""))
    if candidate_rank == 1 and not analysis_plan:
        analysis_plan = ["Compare the measured response across conditions."]

    feasibility_checks = text_block_to_items(sections.get("Feasibility Checks", ""))
    if candidate_rank == 1 and not feasibility_checks:
        feasibility_checks = [
            "Uses only listed equipment and materials.",
            "Keeps the first experiment narrow and executable.",
        ]

    known_evidence = [str(x) for x in task.get("evidence_chunk_ids", [])]
    requested_evidence = [item for item in text_block_to_items(sections.get("Evidence Used", "")) if item in known_evidence]
    evidence_used = requested_evidence or (known_evidence[:1] if candidate_rank == 1 else [])

    goal = str(sections.get("Goal", "")).strip() or str(task.get("goal", "")).strip()
    hypothesis = str(sections.get("Hypothesis", "")).strip()
    if not hypothesis:
        if independent_variables and dependent_variables:
            hypothesis = f"Changing {independent_variables[0]} will affect {dependent_variables[0]}."
        else:
            hypothesis = f"A narrower first-pass experiment can test the task goal: {goal}"

    return normalize_candidate_payload(
        {
            "reasoning_trace": {
                "resource_check": "; ".join(resources_used) or "Resource list was sparse in the local outline.",
                "variable_mapping": "; ".join(independent_variables + dependent_variables) or goal,
                "control_strategy": "; ".join(controls) or "Controls were left under-specified in the local outline.",
                "measurement_strategy": "; ".join(measurement_plan) or "Measurement plan was sparse in the local outline.",
                "risk_check": "; ".join(feasibility_checks) or "Feasibility and risk notes were sparse in the local outline.",
            },
            "final_proposal": {
                "goal": goal,
                "hypothesis": hypothesis,
                "resources_used": resources_used,
                "independent_variables": independent_variables,
                "dependent_variables": dependent_variables,
                "controls": controls,
                "design": {
                    "conditions": conditions,
                    "replicates": replicates,
                    "procedure_outline": procedure_outline,
                },
                "measurement_plan": measurement_plan,
                "analysis_plan": analysis_plan,
                "feasibility_checks": feasibility_checks,
                "evidence_used": evidence_used,
            },
        },
        allow_plain_final_proposal=False,
    )


def build_weak_baseline_candidate_record(
    task: dict[str, Any],
    *,
    candidate_rank: int = 1,
    generator_model: str = "weak_rag_baseline",
) -> dict[str, Any]:
    available_resources = task.get("available_resources", {})
    equipment = normalize_string_list(available_resources.get("equipment", []))
    materials = normalize_string_list(available_resources.get("materials", []))
    default_ivs, default_dvs = _infer_task_variables(str(task.get("goal", "")))

    iv = default_ivs[0] if default_ivs else "main process setting"
    dv = default_dvs[0] if default_dvs else "main response"
    resources_used = _merge_unique_strings(equipment[:1] + materials[:1])

    payload = normalize_candidate_payload(
        {
            "reasoning_trace": {
                "resource_check": "Weak baseline uses only a minimal subset of listed resources.",
                "variable_mapping": f"Probe {iv} against {dv} with intentionally weak structure.",
                "control_strategy": "Controls intentionally omitted for the weak baseline.",
                "measurement_strategy": f"Measure only {dv} in a minimal first pass.",
                "risk_check": "This weak baseline is intentionally under-specified and should stay in the weak_baseline bucket.",
            },
            "final_proposal": {
                "goal": str(task.get("goal", "")).strip(),
                "hypothesis": f"{iv} may affect {dv}.",
                "resources_used": resources_used,
                "independent_variables": [iv],
                "dependent_variables": [dv],
                "controls": [],
                "design": {
                    "conditions": [f"one broad setting change in {iv}"],
                    "replicates": None,
                    "procedure_outline": [
                        "Run one minimal exploratory setting.",
                        f"Check {dv} with the listed lab tools if available.",
                    ],
                },
                "measurement_plan": [dv],
                "analysis_plan": [],
                "feasibility_checks": ["Weak baseline intentionally keeps the plan minimal and under-specified."],
                "evidence_used": [],
            },
        },
        allow_plain_final_proposal=False,
    )

    return {
        "candidate_id": build_candidate_id(str(task["task_id"]), int(candidate_rank), "weak_baseline"),
        "task_id": str(task["task_id"]),
        "candidate_source": "weak_baseline",
        "generator_model": generator_model,
        "candidate_rank": int(candidate_rank),
        "reasoning_trace": payload["reasoning_trace"],
        "final_proposal": payload["final_proposal"],
        "raw_text": "deterministic_weak_baseline_generated_from_task",
    }


def convert_main_rag_outputs_to_candidate_record(
    *,
    task: dict[str, Any],
    rag1_payload: dict[str, Any],
    rag2_payload: dict[str, Any] | None,
    candidate_id: str,
    candidate_rank: int = 1,
    generator_model: str = "main_rag_pipeline",
    candidate_source: str = "strong",
) -> dict[str, Any]:
    available_bom = rag1_payload.get("available_bom", {})
    rag1_proposal = str(rag1_payload.get("rag1_proposal", "")).strip()
    if not rag1_proposal:
        raise ValueError("RAG1 payload is missing rag1_proposal")

    sections = parse_labeled_sections(rag1_proposal, RAG1_SECTION_LABELS)
    source_papers = text_block_to_items(sections.get("Source Papers Used", ""))

    rag2_bom_check = rag2_payload.get("bom_check", {}) if isinstance(rag2_payload, dict) else {}
    rag2_missing = rag2_payload.get("missing_or_limited_items", []) if isinstance(rag2_payload, dict) else []
    rag2_narrowing = rag2_payload.get("narrowing_advice", []) if isinstance(rag2_payload, dict) else []
    rag2_message = str(rag2_payload.get("message_to_rag1", "")).strip() if isinstance(rag2_payload, dict) else ""

    reference_ids = []
    for item in rag2_missing + rag2_narrowing:
        for support in item.get("literature_support", []) or []:
            ref_id = str(support.get("reference_id", "")).strip()
            if ref_id:
                reference_ids.append(ref_id)

    task_evidence_ids = _merge_unique_strings(
        [str(x).strip() for x in task.get("evidence_chunk_ids", []) if str(x).strip()]
    )
    imported_evidence_hints = _merge_unique_strings(source_papers + reference_ids)

    resources_used = _merge_unique_strings(
        text_block_to_items(sections.get("Needed Equipment", ""))
        + text_block_to_items(sections.get("Needed Materials / Consumables", ""))
    )
    independent_variables = _merge_unique_strings(
        text_block_to_items(sections.get("Key Process Parameters To Sweep", ""))
    )
    dependent_variables = _merge_unique_strings(
        text_block_to_items(sections.get("Measurements / Outputs", ""))
    )
    feasibility_checks = _merge_unique_strings(
        [
            sections.get("Why This Is Feasible With Current BOM", ""),
            sections.get("Main Risk / Failure Mode", ""),
            sections.get("Missing Capability / Assumption", ""),
            str(rag2_bom_check.get("reason", "")).strip(),
            rag2_message,
        ]
        + [str(item.get("why_it_matters", "")).strip() for item in rag2_missing]
    )
    procedure_outline = _merge_unique_strings(
        [
            sections.get("Candidate Experiment", ""),
            sections.get("Borrowed Literature Precedents", ""),
            sections.get("New Adaptation / Novel Twist", ""),
        ]
    )

    fallback_analysis_plan = []
    if dependent_variables:
        fallback_analysis_plan.append("Compare measured outputs across proposed conditions.")
    fallback_analysis_plan.extend(
        str(item.get("advice", "")).strip() for item in rag2_narrowing
    )

    payload = normalize_candidate_payload(
        {
            "reasoning_trace": {
                "resource_check": (
                    str(rag2_bom_check.get("reason", "")).strip()
                    or sections.get("Why This Is Feasible With Current BOM", "")
                    or "Imported from main-side RAG outputs."
                ),
                "variable_mapping": "; ".join(independent_variables) or sections.get("Candidate Experiment", ""),
                "control_strategy": "Imported from main-side RAG outputs; explicit controls were not structured.",
                "measurement_strategy": "; ".join(dependent_variables) or sections.get("Measurements / Outputs", ""),
                "risk_check": " | ".join(
                    _merge_unique_strings(
                        [
                            sections.get("Main Risk / Failure Mode", ""),
                            sections.get("Missing Capability / Assumption", ""),
                            rag2_message,
                        ]
                    )
                ),
            },
            "final_proposal": {
                "goal": str(task.get("goal") or available_bom.get("goal") or sections.get("Candidate Experiment", "")).strip(),
                "hypothesis": (
                    sections.get("New Adaptation / Novel Twist", "")
                    or sections.get("Candidate Experiment", "")
                ),
                "resources_used": resources_used,
                "independent_variables": independent_variables,
                "dependent_variables": dependent_variables,
                "controls": [],
                "design": {
                    "conditions": [],
                    "replicates": None,
                    "procedure_outline": procedure_outline,
                },
                "measurement_plan": dependent_variables,
                "analysis_plan": _merge_unique_strings(fallback_analysis_plan) or ["Compare outputs across tested conditions."],
                "feasibility_checks": feasibility_checks,
                "evidence_used": task_evidence_ids or imported_evidence_hints,
            },
        },
        allow_plain_final_proposal=False,
    )

    raw_text_parts = [
        pretty_json(rag1_payload),
    ]
    if rag2_payload is not None:
        raw_text_parts.append(pretty_json(rag2_payload))

    return {
        "candidate_id": candidate_id,
        "task_id": str(task["task_id"]),
        "candidate_source": normalize_candidate_source(candidate_source, default="strong"),
        "generator_model": generator_model,
        "candidate_rank": int(candidate_rank),
        "reasoning_trace": payload["reasoning_trace"],
        "final_proposal": payload["final_proposal"],
        "raw_text": "\n\n".join(raw_text_parts),
    }


def looks_like_judge_payload(payload: Any) -> bool:
    return isinstance(payload, dict) and ("rubric" in payload or "scores" in payload)


def _normalize_rubric_entry(value: Any, *, key: str) -> dict[str, Any]:
    if isinstance(value, int):
        score = value
        if score not in {1, 2, 3, 4, 5}:
            raise ValueError(f"rubric[{key}] score must be 1, 2, 3, 4, or 5")
        return {"score": score, "reason": ""}
    if isinstance(value, str) and value.strip().isdigit():
        score = int(value.strip())
        if score not in {1, 2, 3, 4, 5}:
            raise ValueError(f"rubric[{key}] score must be 1, 2, 3, 4, or 5")
        return {"score": score, "reason": ""}
    if not isinstance(value, dict):
        raise ValueError(f"rubric[{key}] must be an object")
    if "score" not in value:
        raise ValueError(f"rubric[{key}] is missing score")
    score = value["score"]
    if isinstance(score, str) and score.strip().isdigit():
        score = int(score.strip())
    if not isinstance(score, int) or score not in {1, 2, 3, 4, 5}:
        raise ValueError(f"rubric[{key}].score must be 1, 2, 3, 4, or 5")
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


def salvage_local_judge_payload(raw_text: str) -> dict[str, Any] | None:
    text = _normalize_quotes(sanitize_manual_raw_text(raw_text))
    rubric: dict[str, int] = {}

    for key in RUBRIC_DIMENSIONS_V0:
        pattern = re.compile(
            rf'["\']?{re.escape(key)}["\']?\s*:\s*(?:\{{[^{{}}]*?["\']score["\']\s*:\s*)?([1-5])',
            re.IGNORECASE | re.DOTALL,
        )
        match = pattern.search(text)
        if not match:
            return None
        rubric[key] = int(match.group(1))

    payload: dict[str, Any] = {"rubric": rubric}

    hard_fail_match = re.search(r'["\']?hard_fail["\']?\s*:\s*(true|false)', text, re.IGNORECASE)
    if hard_fail_match:
        payload["hard_fail"] = hard_fail_match.group(1).lower() == "true"

    for field in ["hard_fail_reasons", "failure_tags"]:
        array_match = re.search(
            rf'["\']?{field}["\']?\s*:\s*(\[[^\]]*\])',
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if not array_match:
            continue
        try:
            parsed_array = parse_json_loose(array_match.group(1))
        except Exception:
            continue
        if isinstance(parsed_array, list):
            payload[field] = [str(item).strip() for item in parsed_array if str(item).strip()]

    summary_match = re.search(
        r'["\']?summary["\']?\s*:\s*"([^"\n]{0,240})"',
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if summary_match:
        payload["summary"] = summary_match.group(1).strip()

    return payload


def map_raw_rubric_score_1to5_to_compatibility(score: int) -> int:
    if score in {1, 2}:
        return 0
    if score == 3:
        return 1
    if score in {4, 5}:
        return 2
    raise ValueError(f"Unsupported raw rubric score: {score}")


def derive_compatibility_rubric(raw_rubric: dict[str, Any]) -> dict[str, Any]:
    compatibility_rubric: dict[str, Any] = {}
    for key in RUBRIC_DIMENSIONS_V0:
        entry = raw_rubric[key]
        compatibility_rubric[key] = {
            "score": map_raw_rubric_score_1to5_to_compatibility(int(entry["score"])),
            "reason": str(entry.get("reason", "")).strip(),
        }
    return compatibility_rubric


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


def compute_rule_failure_tags(rule_checks: dict[str, bool]) -> list[str]:
    tags: list[str] = []
    if rule_checks.get("unsupported_equipment"):
        tags.append("infeasible_execution")
        tags.append("bom_violation")
    if rule_checks.get("missing_controls"):
        tags.append("weak_documentation")
    if rule_checks.get("empty_replicates"):
        tags.append("weak_resource_logic")
    if rule_checks.get("evidence_mismatch"):
        tags.append("evidence_mismatch")
    return _merge_unique_strings(tags)


def compute_rule_hard_fail(rule_checks: dict[str, bool]) -> bool:
    return bool(rule_checks.get("unsupported_equipment") or rule_checks.get("evidence_mismatch"))


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
    rubric_compatibility: dict[str, Any],
    rule_checks: dict[str, bool],
    *,
    hard_fail: Any = None,
    hard_fail_reasons: list[str] | None = None,
) -> tuple[bool, list[str]]:
    reasons = list(hard_fail_reasons or [])

    if rubric_compatibility["objective_clarity"]["score"] == 0:
        reasons.append("unclear_objective")
    if rubric_compatibility["factor_response_levels"]["score"] == 0:
        reasons.append("missing_factor_response_structure")
    if rubric_compatibility["design_choice_appropriateness"]["score"] == 0:
        reasons.append("poor_design_match")
    if rubric_compatibility["execution_feasibility"]["score"] == 0:
        reasons.append("infeasible_execution")
    if rubric_compatibility["bom_compliance"]["score"] == 0:
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


def compute_raw_total_score_1to5(rubric: dict[str, Any]) -> int:
    return compute_total_score(rubric)


def compute_compatibility_total_score_0to2(rubric_compatibility: dict[str, Any]) -> int:
    return compute_total_score(rubric_compatibility)


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


def looks_like_collapsed_local_judge_rubric(normalized_payload: dict[str, Any]) -> bool:
    rubric = normalized_payload.get("rubric", {})
    if not isinstance(rubric, dict):
        return False

    try:
        raw_scores = [int(rubric[key]["score"]) for key in RUBRIC_DIMENSIONS_V0]
    except Exception:
        return False

    if not raw_scores:
        return False

    unique_scores = set(raw_scores)
    if len(unique_scores) != 1:
        return False

    only_score = next(iter(unique_scores))
    # Small local judges sometimes echo the schema template back with the same score everywhere.
    # When that happens, trust the deterministic structural fallback more than the collapsed rubric.
    return only_score in {1, 5}


def build_rule_first_row(candidate: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    rule_checks = compute_rule_checks(candidate, task)
    rule_failure_tags = compute_rule_failure_tags(rule_checks)
    rule_hard_fail = compute_rule_hard_fail(rule_checks)
    suggested_storage_bucket = (
        "weak_baseline"
        if candidate["candidate_source"] == "weak_baseline"
        else ("rejected" if rule_hard_fail else "pending_local_judge")
    )

    return {
        "candidate_id": candidate["candidate_id"],
        "task_id": candidate["task_id"],
        "candidate_source": candidate["candidate_source"],
        "rule_checks": rule_checks,
        "rule_failure_tags": rule_failure_tags,
        "rule_hard_fail": rule_hard_fail,
        "suggested_storage_bucket": suggested_storage_bucket,
    }


def build_rule_only_judged_row(
    *,
    candidate: dict[str, Any],
    task: dict[str, Any],
    rule_first_row: dict[str, Any],
    judge_model: str,
    judge_contract_version: str,
) -> dict[str, Any]:
    rubric: dict[str, Any] = {}
    for key in RUBRIC_DIMENSIONS_V0:
        score = 2
        reason = "Skipped local LLM judge because the candidate hit a rule-based hard fail."
        if key in {"execution_feasibility", "bom_compliance"} and rule_first_row["rule_checks"].get("unsupported_equipment"):
            score = 1
            reason = "Rule-based hard fail: unsupported equipment or forbidden resource use."
        elif key == "factor_response_levels" and rule_first_row["rule_checks"].get("evidence_mismatch"):
            score = 1
            reason = "Rule-based hard fail: evidence references do not match the task evidence."
        rubric[key] = {"score": score, "reason": reason}

    rubric_compatibility = derive_compatibility_rubric(rubric)
    hard_fail = True
    hard_fail_reasons = _merge_unique_strings(
        compute_rule_failure_tags(rule_first_row["rule_checks"]) + ["rule_based_hard_fail"]
    )
    raw_total_score_1to5 = compute_raw_total_score_1to5(rubric)
    compatibility_total_score_0to2 = compute_compatibility_total_score_0to2(rubric_compatibility)
    overall_verdict = compute_overall_verdict(compatibility_total_score_0to2, hard_fail)
    storage_bucket = compute_storage_bucket(candidate["candidate_source"], overall_verdict)

    return {
        "candidate_id": candidate["candidate_id"],
        "task_id": candidate["task_id"],
        "candidate_source": candidate["candidate_source"],
        "judge_model": judge_model,
        "judge_contract_version": judge_contract_version,
        "rubric": rubric,
        "rubric_compatibility": rubric_compatibility,
        "rule_checks": rule_first_row["rule_checks"],
        "hard_fail": hard_fail,
        "hard_fail_reasons": hard_fail_reasons,
        "failure_tags": _merge_unique_strings(rule_first_row["rule_failure_tags"]),
        "raw_total_score_1to5": raw_total_score_1to5,
        "compatibility_total_score_0to2": compatibility_total_score_0to2,
        "total_score": compatibility_total_score_0to2,
        "detailed_verdict": compute_detailed_verdict(compatibility_total_score_0to2, hard_fail),
        "overall_verdict": overall_verdict,
        "storage_bucket": storage_bucket,
        "summary": "Rejected by rule-first grading before local LLM judging.",
        "overall_reasoning": {
            "strengths": [],
            "weaknesses": rule_first_row["rule_failure_tags"],
            "main_concerns": hard_fail_reasons,
        },
    }


def build_structural_fallback_judged_row(
    *,
    candidate: dict[str, Any],
    task: dict[str, Any],
    judge_model: str,
    judge_contract_version: str,
    summary_note: str,
) -> dict[str, Any]:
    proposal = candidate["final_proposal"]
    rule_checks = compute_rule_checks(candidate, task)

    n_iv = len(normalize_string_list(proposal.get("independent_variables", [])))
    n_dv = len(normalize_string_list(proposal.get("dependent_variables", [])))
    n_controls = len(normalize_string_list(proposal.get("controls", [])))
    n_conditions = len(normalize_string_list(proposal.get("design", {}).get("conditions", [])))
    replicates = proposal.get("design", {}).get("replicates")
    n_steps = len(normalize_string_list(proposal.get("design", {}).get("procedure_outline", [])))
    n_measurements = len(normalize_string_list(proposal.get("measurement_plan", [])))
    n_analysis = len(normalize_string_list(proposal.get("analysis_plan", [])))
    n_feasibility = len(normalize_string_list(proposal.get("feasibility_checks", [])))
    n_evidence = len([str(x) for x in proposal.get("evidence_used", []) if str(x).strip()])
    has_goal = bool(str(proposal.get("goal", "")).strip())
    has_hypothesis = bool(str(proposal.get("hypothesis", "")).strip())

    rubric = {
        "objective_clarity": {
            "score": 5 if has_goal and has_hypothesis else 2,
            "reason": "Fallback structural scoring based on explicit goal and hypothesis fields.",
        },
        "factor_response_levels": {
            "score": 5 if n_iv >= 1 and n_dv >= 1 and n_conditions >= 3 and replicates and replicates >= 3 else (3 if n_iv >= 1 and n_dv >= 1 else 2),
            "reason": "Fallback structural scoring based on variables, conditions, and replicates.",
        },
        "design_choice_appropriateness": {
            "score": 4 if n_conditions >= 3 and n_analysis >= 1 else (3 if n_conditions >= 2 else 2),
            "reason": "Fallback structural scoring based on design completeness and analysis plan.",
        },
        "interaction_curvature_awareness": {
            "score": 4 if n_conditions >= 3 else (3 if n_conditions >= 2 else 2),
            "reason": "Fallback structural scoring based on the number of planned conditions.",
        },
        "resource_aware_design": {
            "score": 5 if not rule_checks["unsupported_equipment"] and normalize_string_list(proposal.get("resources_used", [])) else 2,
            "reason": "Fallback structural scoring based on listed resources and BOM fit.",
        },
        "execution_feasibility": {
            "score": 5 if not rule_checks["unsupported_equipment"] and n_steps >= 2 else 2,
            "reason": "Fallback structural scoring based on procedure detail and BOM fit.",
        },
        "documentation_rigor": {
            "score": 4 if n_steps >= 2 and n_measurements >= 1 and n_analysis >= 1 else 2,
            "reason": "Fallback structural scoring based on procedure, measurements, and analysis.",
        },
        "blocking_awareness": {
            "score": 4 if n_controls >= 1 and replicates and n_feasibility >= 1 else (2 if n_feasibility >= 1 else 1),
            "reason": "Fallback structural scoring based on controls, replicates, and feasibility notes.",
        },
        "iterative_experimentation": {
            "score": 4 if n_conditions and n_conditions <= 4 and n_analysis >= 1 else 2,
            "reason": "Fallback structural scoring based on whether the plan looks like a narrow first pass.",
        },
        "bom_compliance": {
            "score": 5 if not rule_checks["unsupported_equipment"] else 1,
            "reason": "Fallback structural scoring based on rule-based BOM compliance.",
        },
        "claim_discipline": {
            "score": 4 if n_evidence >= 1 and n_feasibility >= 1 else (3 if n_evidence >= 1 or n_feasibility >= 1 else 2),
            "reason": "Fallback structural scoring based on evidence and feasibility support.",
        },
    }

    if rule_checks["missing_controls"]:
        rubric["factor_response_levels"]["score"] = min(int(rubric["factor_response_levels"]["score"]), 2)
        rubric["design_choice_appropriateness"]["score"] = min(int(rubric["design_choice_appropriateness"]["score"]), 2)
        rubric["documentation_rigor"]["score"] = min(int(rubric["documentation_rigor"]["score"]), 2)
        rubric["blocking_awareness"]["score"] = min(int(rubric["blocking_awareness"]["score"]), 1)
    if rule_checks["empty_replicates"]:
        rubric["factor_response_levels"]["score"] = min(int(rubric["factor_response_levels"]["score"]), 2)
        rubric["interaction_curvature_awareness"]["score"] = min(int(rubric["interaction_curvature_awareness"]["score"]), 2)
        rubric["blocking_awareness"]["score"] = min(int(rubric["blocking_awareness"]["score"]), 1)
        rubric["iterative_experimentation"]["score"] = min(int(rubric["iterative_experimentation"]["score"]), 2)
    if rule_checks["evidence_mismatch"]:
        rubric["claim_discipline"]["score"] = 1
    if int(candidate.get("candidate_rank", 1)) > 1:
        rubric["documentation_rigor"]["score"] = min(int(rubric["documentation_rigor"]["score"]), 2)
        rubric["blocking_awareness"]["score"] = min(int(rubric["blocking_awareness"]["score"]), 2)
        rubric["iterative_experimentation"]["score"] = min(int(rubric["iterative_experimentation"]["score"]), 2)

    rubric_compatibility = derive_compatibility_rubric(rubric)
    hard_fail, hard_fail_reasons = derive_hard_fail(
        rubric_compatibility,
        rule_checks,
        hard_fail=False,
        hard_fail_reasons=[],
    )
    failure_tags = _merge_unique_strings(compute_rule_failure_tags(rule_checks) + hard_fail_reasons)
    raw_total_score_1to5 = compute_raw_total_score_1to5(rubric)
    compatibility_total_score_0to2 = compute_compatibility_total_score_0to2(rubric_compatibility)
    overall_verdict = compute_overall_verdict(compatibility_total_score_0to2, hard_fail)
    storage_bucket = compute_storage_bucket(candidate["candidate_source"], overall_verdict)

    return {
        "candidate_id": candidate["candidate_id"],
        "task_id": candidate["task_id"],
        "candidate_source": candidate["candidate_source"],
        "judge_model": judge_model,
        "judge_contract_version": judge_contract_version,
        "rubric": rubric,
        "rubric_compatibility": rubric_compatibility,
        "rule_checks": rule_checks,
        "hard_fail": hard_fail,
        "hard_fail_reasons": hard_fail_reasons,
        "failure_tags": failure_tags,
        "raw_total_score_1to5": raw_total_score_1to5,
        "compatibility_total_score_0to2": compatibility_total_score_0to2,
        "total_score": compatibility_total_score_0to2,
        "detailed_verdict": compute_detailed_verdict(compatibility_total_score_0to2, hard_fail),
        "overall_verdict": overall_verdict,
        "storage_bucket": storage_bucket,
        "summary": summary_note,
        "overall_reasoning": {
            "strengths": ["Fallback structural judge used the canonical candidate fields."],
            "weaknesses": failure_tags,
            "main_concerns": hard_fail_reasons,
        },
    }


def canonicalize_local_judge_summary(
    summary: str,
    *,
    candidate_source: str,
    overall_verdict: str,
    hard_fail: bool,
) -> str:
    text = str(summary or "").strip()
    lowered = text.lower()

    contradictory_positive = (
        overall_verdict == "reject"
        and any(
            phrase in lowered
            for phrase in [
                "meets all criteria",
                "strong candidate",
                "accepted",
                "accept_silver",
            ]
        )
    )
    contradictory_negative = (
        overall_verdict == "accept_silver"
        and any(
            phrase in lowered
            for phrase in [
                "reject",
                "not valid",
                "does not meet",
                "failure",
            ]
        )
    )

    if candidate_source == "weak_baseline":
        if not text or contradictory_positive:
            if hard_fail:
                return "Weak baseline judged as under-specified and retained in the weak_baseline bucket."
            return "Weak baseline judged and retained in the weak_baseline provenance bucket."

    if not text or contradictory_positive or contradictory_negative:
        if overall_verdict == "accept_silver":
            return "Locally judged as acceptable silver for the current task."
        return "Locally judged as reject for the current task."

    return text


def materialize_local_bucket_views(
    rows: list[dict[str, Any]],
    *,
    out_dir: Path,
) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    accepted_silver_lite = [
        row for row in rows
        if row.get("candidate_source") != "weak_baseline"
        and row.get("storage_bucket") == "accepted_silver"
    ]
    rejected = [
        row for row in rows
        if row.get("candidate_source") != "weak_baseline"
        and row.get("storage_bucket") == "rejected"
    ]
    weak_baseline = [
        row for row in rows
        if row.get("candidate_source") == "weak_baseline"
        or row.get("storage_bucket") == "weak_baseline"
    ]

    dump_jsonl(out_dir / "accepted_silver_lite.jsonl", accepted_silver_lite)
    dump_jsonl(out_dir / "rejected.jsonl", rejected)
    dump_jsonl(out_dir / "weak_baseline.jsonl", weak_baseline)
    return {
        "accepted_silver_lite": len(accepted_silver_lite),
        "rejected": len(rejected),
        "weak_baseline": len(weak_baseline),
    }


def normalize_manual_judge_payload(
    payload: Any,
    *,
    candidate: dict[str, Any],
    task: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("judge payload must be a JSON object")

    rubric = normalize_rubric_v0(payload)
    rubric_compatibility = derive_compatibility_rubric(rubric)
    rule_checks = normalize_rule_checks(payload.get("rule_checks"), candidate=candidate, task=task)
    hard_fail_reasons, failure_tags, overall_reasoning, summary = normalize_reasoning_summary(payload)
    hard_fail, hard_fail_reasons = derive_hard_fail(
        rubric_compatibility,
        rule_checks,
        hard_fail=payload.get("hard_fail"),
        hard_fail_reasons=hard_fail_reasons,
    )

    return {
        "rubric": rubric,
        "rubric_compatibility": rubric_compatibility,
        "rule_checks": rule_checks,
        "hard_fail": hard_fail,
        "hard_fail_reasons": hard_fail_reasons,
        "failure_tags": _merge_unique_strings(failure_tags),
        "overall_reasoning": overall_reasoning,
        "summary": summary,
    }
