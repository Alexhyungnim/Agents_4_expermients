import re
import json
import ast
from pathlib import Path
from typing import List, Dict, Any


JUDGE_PROMPT = """You are a strict evaluator for experiment proposals in engineering research.

Your task is to evaluate whether a proposed experiment is a rigorous and feasible experiment proposal.

The proposal must be evaluated using the rubric below.
The proposal must also be checked against the provided available BOM, equipment list, materials list, forbidden items, and other setup constraints.

Scoring scale for each rubric item:
- 0 = missing, very weak, or clearly inappropriate
- 1 = partly present, but incomplete or weak
- 2 = clear, appropriate, and sufficiently rigorous

Rubric:

1. objective_clarity
Check whether the goal of the experiment is clear.
Look for whether the proposal clearly states if it is trying to compare conditions, screen factors, test a hypothesis, or find an optimum.

2. factor_response_levels
Check whether the experiment structure is clear.
Look for clearly stated input factors, their levels or ranges, and the output responses or measurements.

3. design_choice_appropriateness
Check whether the proposed design matches the stated goal.
For example, if the goal is screening, the design should reflect screening logic.
If the goal is optimization, the design should show optimization or response-surface-type reasoning.

4. interaction_curvature_awareness
Check whether the proposal shows awareness that factors may interact.
If optimization is claimed, check whether the proposal considers curvature, center points, staged refinement, or follow-up design.

5. resource_aware_design
Check whether the proposal uses runs and resources efficiently.
Look for whether the design is too large, too vague, or wasteful when a smaller but useful design could be used.

6. execution_feasibility
Check whether the experiment can actually be done using the provided setup.
Only count equipment, materials, and capabilities that are explicitly available in the BOM/setup.
Penalize the proposal if it requires missing tools, unrealistic capabilities, or forbidden items.

7. documentation_rigor
Check whether the proposal includes enough detail to support recording, tracking, and reviewing the experiment.
Look for raw data collection, key outputs, or useful documentation of the process.

8. blocking_awareness
Check whether the proposal considers known nuisance variation.
Look for awareness of possible variation from machine condition, operator, day, batch, or other known sources.

9. iterative_experimentation
Check whether the proposal allows step-by-step learning.
Look for follow-up experiments, refinement, staged testing, or a plan that does not assume one single experiment will answer everything.

10. bom_compliance
Check whether the proposal stays within the provided BOM and constraints.
Look for:
- only using listed equipment and materials
- not depending on forbidden items
- not assuming hidden capabilities
- making claims that are actually supported by the listed setup

11. claim_discipline
Check whether the proposal avoids overclaiming.
Penalize vague claims of novelty, optimization, mechanism discovery, or feasibility if the proposal does not provide enough support.

Major failure tags to use when relevant:
- unclear_objective
- missing_factor_response_structure
- poor_design_match
- no_interaction_awareness
- weak_resource_logic
- infeasible_execution
- weak_documentation
- no_blocking_awareness
- no_iterative_plan
- bom_violation
- forbidden_item_dependency
- unsupported_claims

Hard-fail rules:
Set hard_fail = true if one or more of the following is true:
- The objective is unclear.
- Factors and responses are not clearly specified.
- The design does not match the stated goal.
- The experiment is not feasible under the provided BOM/setup.
- The proposal depends on forbidden items or missing capabilities.
- The proposal is only a loose parameter sweep with no meaningful design logic.

Evaluation instructions:
- Be strict.
- Do not reward proposals just because they sound plausible.
- Judge only from the provided setup, constraints, and proposal text.
- Do not assume missing information.
- If an important part is not stated, treat it as missing.
- After scoring, explain the reason for each score briefly.
- Also provide an overall summary explaining the main strengths and weaknesses.

Return JSON only.
Use exactly this schema:

{
  "hard_fail": false,
  "hard_fail_reasons": [],
  "scores": {
    "objective_clarity": {"score": 0, "reason": ""},
    "factor_response_levels": {"score": 0, "reason": ""},
    "design_choice_appropriateness": {"score": 0, "reason": ""},
    "interaction_curvature_awareness": {"score": 0, "reason": ""},
    "resource_aware_design": {"score": 0, "reason": ""},
    "execution_feasibility": {"score": 0, "reason": ""},
    "documentation_rigor": {"score": 0, "reason": ""},
    "blocking_awareness": {"score": 0, "reason": ""},
    "iterative_experimentation": {"score": 0, "reason": ""},
    "bom_compliance": {"score": 0, "reason": ""},
    "claim_discipline": {"score": 0, "reason": ""}
  },
  "failure_tags": [],
  "total_score": 0,
  "overall_verdict": "",
  "overall_reasoning": {
    "strengths": [],
    "weaknesses": [],
    "main_concerns": []
  },
  "summary": ""
}

Verdict rules:
- If hard_fail is true, overall_verdict must be "fail".
- Otherwise:
  - 0 to 8 = fail
  - 9 to 13 = plausible_idea_only
  - 14 to 18 = weak_proposal
  - 19 to 22 = usable_with_revisions
- Under the current 22-point rubric, do not use "strong_proposal" or "rigorous".
"""


def find_bom_blocks(text: str) -> List[Dict[str, Any]]:
    """
    Find all available_bom = { ... } blocks.
    Returns list of dicts with:
      - bom_text
      - bom_dict
      - start
      - end
    """
    pattern = re.compile(r'available_bom\s*=\s*\{', re.MULTILINE)
    matches = list(pattern.finditer(text))
    results = []

    for m in matches:
        start = m.start()
        brace_start = text.find("{", m.start())
        if brace_start == -1:
            continue

        depth = 0
        end = None
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        if end is None:
            continue

        bom_text = text[m.start():end]
        raw_dict_text = text[brace_start:end]

        bom_dict = None
        try:
            bom_dict = ast.literal_eval(raw_dict_text)
        except Exception:
            bom_dict = {"raw_parse_failed": True, "raw_text": raw_dict_text}

        results.append({
            "bom_text": bom_text,
            "bom_dict": bom_dict,
            "start": start,
            "end": end,
        })

    return results


def split_case_region(region_text: str) -> List[str]:
    """
    Split a text region into proposal chunks.
    Uses common markers from your pasted logs.
    """
    markers = [
        r'\nShortlisted paper cards:',
        r'\nResponse:',
        r'\nResponse\n',
        r'\nProposed EXperiment:',
        r'\nProposed Experiment:',
    ]

    split_pattern = re.compile("|".join(markers), re.IGNORECASE)
    parts = split_pattern.split(region_text)
    found_markers = split_pattern.findall(region_text)

    chunks = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue

        # Restore marker meaning except for leading unrelated text
        if i == 0 and not found_markers:
            chunks.append(part)
        elif i == 0 and found_markers:
            # first chunk is preamble before first marker; usually skip if tiny
            if len(part) > 80:
                chunks.append(part)
        else:
            marker = found_markers[i - 1] if i - 1 < len(found_markers) else ""
            rebuilt = f"{marker.strip()}\n{part}".strip()
            chunks.append(rebuilt)

    # Filter out obvious BOM-only leftovers
    cleaned = []
    for c in chunks:
        lower = c.lower()
        if "available_bom" in lower and len(c) < 500:
            continue
        cleaned.append(c)

    return cleaned


def extract_cases(text: str) -> List[Dict[str, Any]]:
    """
    Pair each BOM block with the proposal chunks that follow it until the next BOM block.
    """
    boms = find_bom_blocks(text)
    if not boms:
        return []

    cases = []
    for idx, bom in enumerate(boms):
        region_start = bom["end"]
        region_end = boms[idx + 1]["start"] if idx + 1 < len(boms) else len(text)
        region_text = text[region_start:region_end].strip()

        proposal_chunks = split_case_region(region_text)
        for j, proposal in enumerate(proposal_chunks, start=1):
            # Skip tiny junk chunks
            if len(proposal.strip()) < 120:
                continue

            cases.append({
                "case_id": f"{idx+1:03d}_{j:02d}",
                "bom_text": bom["bom_text"],
                "bom_dict": bom["bom_dict"],
                "proposal_text": proposal.strip(),
            })

    return cases


def build_prompt(case: Dict[str, Any]) -> str:
    bom_dict = case["bom_dict"]
    bom_json = json.dumps(bom_dict, indent=2, ensure_ascii=False)

    forbidden = bom_dict.get("forbidden_items", [])
    forbidden_json = json.dumps(forbidden, indent=2, ensure_ascii=False)

    prompt = f"""{JUDGE_PROMPT}

Evaluate the following experiment proposal.

Available BOM:
{bom_json}

Forbidden items:
{forbidden_json}

Experiment proposal:
{case["proposal_text"]}

Return JSON only.
"""
    return prompt


def save_outputs(cases: List[Dict[str, Any]], output_dir: str = "generated_prompts") -> None:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    jsonl_path = outdir / "prompts.jsonl"
    manifest_path = outdir / "manifest.csv"

    with jsonl_path.open("w", encoding="utf-8") as jf, manifest_path.open("w", encoding="utf-8") as mf:
        mf.write("case_id,prompt_file\n")

        for idx, case in enumerate(cases, start=1):
            prompt_text = build_prompt(case)
            prompt_file = outdir / f"prompt_{idx:03d}_{case['case_id']}.txt"
            prompt_file.write_text(prompt_text, encoding="utf-8")

            row = {
                "case_id": case["case_id"],
                "bom": case["bom_dict"],
                "proposal_text": case["proposal_text"],
                "prompt_text": prompt_text,
            }
            jf.write(json.dumps(row, ensure_ascii=False) + "\n")
            mf.write(f"{case['case_id']},{prompt_file.name}\n")


def main():
    input_path = "proposed_experiments.txt"
    output_dir = "generated_prompts"

    text = Path(input_path).read_text(encoding="utf-8")
    cases = extract_cases(text)

    if not cases:
        print("No cases found.")
        return

    save_outputs(cases, output_dir=output_dir)
    print(f"Done. Extracted {len(cases)} cases.")
    print(f"Saved to: {output_dir}/")


if __name__ == "__main__":
    main()
