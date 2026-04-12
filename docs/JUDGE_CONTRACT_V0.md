# Judge Contract V0

## Purpose

This document defines the provisional `v0` judging contract for experiment proposals in this repo.

It aligns two existing judging surfaces:

- the detailed rubric logic in `rubricsetup.py`
- the current FT-pipeline judge output shape in `exp_design_factory_next/scripts/07_judge_candidates.py`

Goals:

- preserve the current repo's `accept_silver` / `reject` workflow
- elevate the detailed rubric in `rubricsetup.py` to the canonical scoring contract
- keep the JSON record shape close to the current `judged_candidates.jsonl` flow

## Canonical Sources

The `v0` judging contract is based on:

- `rubricsetup.py`
  - canonical rubric dimensions
  - canonical hard-fail conditions
  - canonical failure tags
  - canonical detailed verdict vocabulary
- `exp_design_factory_next/scripts/07_judge_candidates.py`
  - canonical judged-record location and current `accept_silver` / `reject` gate
  - canonical lightweight `rule_checks` concept

## Judge Input Contract

The judge evaluates one candidate against one task and one available setup.

Required input payload components:

- task metadata
- candidate metadata
- candidate proposal content
- available BOM or equivalent resource description

Accepted input sources in the current repo:

1. task-driven FT path
   - task record from `data/processed/tasks/tasks.jsonl`
   - candidate record from `data/processed/candidates/*.jsonl`

2. proposal-text path
   - BOM block plus free-form proposal text, as used by `rubricsetup.py`

For `v0`, the judge contract does not require a single upstream producer. It requires a normalized candidate payload at judgment time.

## Canonical Rubric Dimensions

The canonical `v0` rubric uses the detailed dimensions from `rubricsetup.py`.

Required rubric keys:

- `objective_clarity`
- `factor_response_levels`
- `design_choice_appropriateness`
- `interaction_curvature_awareness`
- `resource_aware_design`
- `execution_feasibility`
- `documentation_rigor`
- `blocking_awareness`
- `iterative_experimentation`
- `bom_compliance`
- `claim_discipline`

Each rubric item must have:

- `score`: integer in `{0, 1, 2}`
- `reason`: string

Scale:

- `0` = missing, very weak, or clearly inappropriate
- `1` = partly present, but incomplete or weak
- `2` = clear, appropriate, and sufficiently rigorous

## Canonical Hard-Fail Logic

The canonical `v0` hard-fail reasons are derived from `rubricsetup.py`.

Hard-fail when one or more of these are true:

- the objective is unclear
- factors and responses are not clearly specified
- the design does not match the stated goal
- the experiment is not feasible under the provided BOM/setup
- the proposal depends on forbidden items or missing capabilities
- the proposal is only a loose parameter sweep with no meaningful design logic

Required fields:

- `hard_fail`: boolean
- `hard_fail_reasons`: string[]

Recommended `hard_fail_reasons` vocabulary:

- `unclear_objective`
- `missing_factor_response_structure`
- `poor_design_match`
- `infeasible_execution`
- `bom_violation`
- `forbidden_item_dependency`
- `loose_parameter_sweep`

## Failure Tags

The canonical `failure_tags` list follows `rubricsetup.py`.

Recommended tag vocabulary:

- `unclear_objective`
- `missing_factor_response_structure`
- `poor_design_match`
- `no_interaction_awareness`
- `weak_resource_logic`
- `infeasible_execution`
- `weak_documentation`
- `no_blocking_awareness`
- `no_iterative_plan`
- `bom_violation`
- `forbidden_item_dependency`
- `unsupported_claims`

Not every tag is a hard fail. `failure_tags` is a broader diagnostic field.

## Compatibility Rule Checks

To stay close to the current FT pipeline, `v0` keeps a lightweight `rule_checks` object.

Required `rule_checks` keys in `v0`:

- `unsupported_equipment`: boolean
- `missing_controls`: boolean
- `empty_replicates`: boolean
- `evidence_mismatch`: boolean

Meaning:

- these are fast structural checks, not the full rubric
- they are retained because `07_judge_candidates.py` already emits them

`v0` hard-fail interpretation:

- `unsupported_equipment = true` should force `hard_fail = true`
- `evidence_mismatch = true` should usually force `hard_fail = true`
- `missing_controls = true` and `empty_replicates = true` are strong rejection signals and may force `hard_fail` if the task requires them for basic feasibility

## Verdict Contract

### 1. `total_score`

`total_score` is the sum of the 11 canonical rubric scores.

Current mathematical range:

- minimum = `0`
- maximum = `22`

This matters because `rubricsetup.py` currently contains detailed verdict bands that extend to `30`, but the present rubric only has 11 items scored 0 to 2.

### 2. `detailed_verdict`

The canonical detailed verdict vocabulary is:

- `fail`
- `plausible_idea_only`
- `weak_proposal`
- `usable_with_revisions`
- `strong_proposal`
- `rigorous`

For `v0`, the detailed verdict bands are interpreted as:

- if `hard_fail = true`, `detailed_verdict = "fail"`
- else `0` to `8` -> `fail`
- else `9` to `13` -> `plausible_idea_only`
- else `14` to `18` -> `weak_proposal`
- else `19` to `22` -> `usable_with_revisions`

Reserved but currently unreachable under the existing 11-dimension rubric:

- `strong_proposal`
- `rigorous`

### 3. `overall_verdict`

To preserve compatibility with the current FT pipeline, `v0` keeps:

- `overall_verdict = "accept_silver" | "reject"`

Compatibility rule for `v0`:

- `accept_silver` if `total_score >= 11` and `hard_fail = false`
- otherwise `reject`

This matches the current gate in `exp_design_factory_next/scripts/07_judge_candidates.py`.

### 4. `storage_bucket`

`storage_bucket` is the dataset-facing bucket name used by `DATASET_PLAN_V0`.

Allowed values:

- `accepted_silver`
- `rejected`
- `weak_baseline`

Bucket rule:

- if `candidate_source = "weak_baseline"`, set `storage_bucket = "weak_baseline"`
- else if `overall_verdict = "accept_silver"`, set `storage_bucket = "accepted_silver"`
- else set `storage_bucket = "rejected"`

## Judged Candidate Record

The `v0` judged candidate record should stay close to the current `judged_candidates.jsonl` rows.

Required top-level fields:

- `candidate_id`: string
- `task_id`: string
- `candidate_source`: `strong | weak_baseline`
- `judge_model`: string
- `judge_contract_version`: string
- `rubric`: object
- `rule_checks`: object
- `hard_fail`: boolean
- `hard_fail_reasons`: string[]
- `failure_tags`: string[]
- `total_score`: integer
- `detailed_verdict`: string
- `overall_verdict`: `accept_silver | reject`
- `storage_bucket`: `accepted_silver | rejected | weak_baseline`
- `summary`: string

Optional but recommended:

- `overall_reasoning`
- `judged_at`
- `rubric_legacy`

### `rubric_legacy`

Because the current FT scaffold uses a reduced 7-field rubric, `v0` may optionally carry:

- `rubric_legacy`

Recommended keys:

- `resource_consistency`
- `variable_quality`
- `control_strategy`
- `measurement_plan`
- `analysis_plan`
- `evidence_grounding`
- `novelty_or_usefulness`

This is optional in `v0`. The canonical scoring object is `rubric`.

## Example Judged Candidate Record

```json
{
  "candidate_id": "cand_task_paper_001_01_01",
  "task_id": "task_paper_001_01",
  "candidate_source": "strong",
  "judge_model": "strong_judge_B",
  "judge_contract_version": "judge_v0",
  "rubric": {
    "objective_clarity": {"score": 2, "reason": "The proposal clearly states a comparative welding objective."},
    "factor_response_levels": {"score": 2, "reason": "Two process factors, levels, and measured responses are stated."},
    "design_choice_appropriateness": {"score": 1, "reason": "The design is plausible but not yet fully justified as a screening design."},
    "interaction_curvature_awareness": {"score": 1, "reason": "The proposal notes staged follow-up, but interaction handling is limited."},
    "resource_aware_design": {"score": 2, "reason": "The first-pass run plan is small and practical for the listed lab resources."},
    "execution_feasibility": {"score": 2, "reason": "The required equipment is listed in the BOM."},
    "documentation_rigor": {"score": 1, "reason": "Measurements are named, but raw-data logging detail is still thin."},
    "blocking_awareness": {"score": 1, "reason": "Replicates are present, but nuisance variation is only partly addressed."},
    "iterative_experimentation": {"score": 2, "reason": "The proposal explicitly frames the study as a first-step experiment."},
    "bom_compliance": {"score": 2, "reason": "No forbidden tools or hidden capabilities are assumed."},
    "claim_discipline": {"score": 1, "reason": "Mechanistic claims are restrained but still somewhat broad."}
  },
  "rule_checks": {
    "unsupported_equipment": false,
    "missing_controls": false,
    "empty_replicates": false,
    "evidence_mismatch": false
  },
  "hard_fail": false,
  "hard_fail_reasons": [],
  "failure_tags": [],
  "total_score": 17,
  "detailed_verdict": "weak_proposal",
  "overall_verdict": "accept_silver",
  "storage_bucket": "accepted_silver",
  "summary": "Feasible first-pass proposal with clear factors and BOM compliance, but still needs stronger experimental-design justification and better documentation detail."
}
```

## V0 Summary

The `v0` judge contract makes one main alignment choice:

- the detailed 11-dimension rubric from `rubricsetup.py` is canonical

It also keeps one important compatibility choice:

- `overall_verdict` stays `accept_silver | reject` so the current FT dataset scripts remain conceptually compatible

This preserves the current repo shape while making the judging semantics explicit enough for later code alignment.
