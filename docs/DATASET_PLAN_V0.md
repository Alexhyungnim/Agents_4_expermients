# Dataset Plan V0

## Purpose

This document defines the provisional `v0` storage plan for fine-tuning data creation.

Goals:

- keep the current `exp_design_factory_next` JSONL layout intact
- define how `accepted_silver`, `rejected`, and `weak_baseline` should be stored and interpreted
- make the task, candidate, and judged-candidate record shapes explicit

This is a documentation-only pass. No data files or scripts are changed here.

## Current Producer-Compatible Layout

The current repo already writes these raw files:

| Stage | Current path | Producer |
| --- | --- | --- |
| Tasks | `data/processed/tasks/tasks.jsonl` | `04_build_tasks.py` |
| Strong candidates | `data/processed/candidates/candidates.jsonl` | `05_generate_candidates.py` |
| Weak baseline candidates | `data/processed/candidates/weak_rag_candidates.jsonl` | `06_generate_weak_rag.py` |
| Judged strong candidates | `data/processed/judged/judged_candidates.jsonl` | `07_judge_candidates.py` |
| SFT dataset | `data/processed/datasets/sft/train.jsonl` | `08_build_sft_dataset.py` |
| DPO dataset | `data/processed/datasets/dpo/train.jsonl` | `09_build_dpo_dataset.py` |
| Validator dataset | `data/processed/datasets/validator/train.jsonl` | `10_build_validator_dataset.py` |

## Provisional V0 Storage Rules

### 1. Raw Producer Files Stay As They Are

For `v0`, the raw JSONL producer files above remain the canonical storage locations.

No path rename is introduced here.

### 2. Dataset Buckets Are Logical Buckets First

For `v0`, the three dataset-facing buckets are defined as logical buckets over the raw files:

- `accepted_silver`
- `rejected`
- `weak_baseline`

This avoids forcing a code refactor while still defining clean bucket semantics.

### 3. Bucket Definitions

#### `accepted_silver`

Source:

- `data/processed/judged/judged_candidates.jsonl`

Selection rule:

- `candidate_source = "strong"`
- `overall_verdict = "accept_silver"`

Use:

- default positive pool for SFT
- `chosen` side for DPO

#### `rejected`

Source:

- `data/processed/judged/judged_candidates.jsonl`

Selection rule:

- `candidate_source = "strong"`
- `overall_verdict = "reject"`

Use:

- default negative pool for DPO
- optional error analysis and validator data

#### `weak_baseline`

Source in current repo:

- `data/processed/candidates/weak_rag_candidates.jsonl`

Recommended future judged mirror:

- `data/processed/judged/judged_weak_rag_candidates.jsonl`

Selection rule:

- `candidate_source = "weak_baseline"`

Use:

- baseline comparison set
- optional extra rejected examples for DPO later
- optional evaluation-only split

Important rule:

- `weak_baseline` is a provenance bucket, not a quality bucket
- a weak-baseline candidate stays in the `weak_baseline` storage bucket even if it is later judged

## Required Cross-Record Fields

To align the raw producer files with the dataset buckets, `v0` treats these fields as the minimum cross-record identifiers:

- `task_id`
- `candidate_id`
- `candidate_source`

Allowed `candidate_source` values:

- `strong`
- `weak_baseline`

Compatibility note:

- current strong-candidate rows do not yet carry `candidate_source`
- for `v0`, strong candidates should be interpreted as `candidate_source = "strong"` by default
- current weak baseline rows should be interpreted as `candidate_source = "weak_baseline"` by file of origin

## Record Contracts

### 1. Task Record

Canonical location:

- `data/processed/tasks/tasks.jsonl`

Required fields:

- `task_id`: string
- `paper_id`: string
- `domain`: string
- `task_type`: string
- `goal`: string
- `available_resources`: object
- `evidence_chunk_ids`: string[] or integer[]
- `evidence_summary`: string

Recommended `available_resources` fields:

- `equipment`: string[]
- `materials`: string[]
- `constraints`: string[]
- `forbidden_items`: string[]

Compatibility note:

- `forbidden_items` is recommended for `v0` even though the current scaffold task builder does not write it yet
- this keeps the task contract closer to the BOM logic used by the main-side RAG pipeline and `rubricsetup.py`

Example task record:

```json
{
  "task_id": "task_paper_001_01",
  "paper_id": "paper_001",
  "domain": "welding",
  "task_type": "experiment_design",
  "goal": "Design one feasible evidence-grounded friction welding experiment for carbon steel joints.",
  "available_resources": {
    "equipment": [
      "friction welding machine",
      "thermocouple",
      "microhardness tester",
      "tensile testing machine",
      "optical microscope",
      "SEM"
    ],
    "materials": ["carbon steel", "stainless steel", "thermocouple"],
    "constraints": [
      "Use only evidence-supported resources.",
      "Keep the first experiment narrow and executable."
    ],
    "forbidden_items": ["EBSD", "synchrotron", "CFD simulation"]
  },
  "evidence_chunk_ids": ["chunk_001", "chunk_004", "chunk_007"],
  "evidence_summary": "Prior papers vary friction pressure and upset pressure and measure hardness and tensile strength."
}
```

### 2. Candidate Record

Canonical locations:

- strong: `data/processed/candidates/candidates.jsonl`
- weak baseline: `data/processed/candidates/weak_rag_candidates.jsonl`

Required fields:

- `candidate_id`: string
- `task_id`: string
- `generator_model`: string
- `candidate_rank`: integer
- `reasoning_trace`: object
- `final_proposal`: object
- `raw_text`: string

Recommended `v0` field:

- `candidate_source`: `strong | weak_baseline`

Compatibility rule:

- if missing and read from `candidates.jsonl`, interpret as `strong`
- if missing and read from `weak_rag_candidates.jsonl`, interpret as `weak_baseline`

Required `final_proposal` fields in `v0`:

- `goal`: string
- `hypothesis`: string
- `resources_used`: string[]
- `independent_variables`: string[]
- `dependent_variables`: string[]
- `controls`: string[]
- `design`: object
- `measurement_plan`: string[]
- `analysis_plan`: string[]
- `feasibility_checks`: string[]
- `evidence_used`: string[]

Required `design` fields:

- `conditions`: string[]
- `replicates`: integer or null
- `procedure_outline`: string[]

Example candidate record:

```json
{
  "candidate_id": "cand_task_paper_001_01_01",
  "task_id": "task_paper_001_01",
  "candidate_source": "strong",
  "generator_model": "strong_generator_A",
  "candidate_rank": 1,
  "reasoning_trace": {
    "resource_check": "All core equipment is listed in the task resources.",
    "variable_mapping": "Use friction pressure and upset pressure as the first-pass factors.",
    "control_strategy": "Keep material grade and specimen geometry fixed.",
    "measurement_strategy": "Measure hardness and tensile strength for each condition.",
    "risk_check": "Avoid too many factors in the first run."
  },
  "final_proposal": {
    "goal": "Compare a small pressure matrix for carbon steel friction welds.",
    "hypothesis": "A moderate increase in upset pressure will improve joint strength without requiring additional equipment.",
    "resources_used": [
      "friction welding machine",
      "thermocouple",
      "microhardness tester",
      "tensile testing machine",
      "optical microscope"
    ],
    "independent_variables": ["friction pressure", "upset pressure"],
    "dependent_variables": ["tensile strength", "hardness"],
    "controls": ["same material grade", "same specimen geometry", "same operator"],
    "design": {
      "conditions": [
        "low friction pressure / low upset pressure",
        "low friction pressure / high upset pressure",
        "high friction pressure / low upset pressure",
        "high friction pressure / high upset pressure"
      ],
      "replicates": 3,
      "procedure_outline": [
        "Prepare carbon steel specimens with fixed geometry.",
        "Run four friction welding conditions.",
        "Measure hardness across the interface.",
        "Perform tensile tests on all joints."
      ]
    },
    "measurement_plan": ["hardness profile", "ultimate tensile strength"],
    "analysis_plan": ["compare condition means", "inspect tradeoff between strength and hardness"],
    "feasibility_checks": ["all required equipment is in the BOM", "no forbidden characterization is assumed"],
    "evidence_used": ["chunk_001", "chunk_004", "chunk_007"]
  },
  "raw_text": "Task: friction welding study for carbon steel..."
}
```

### 3. Judged Candidate Record

Canonical location in current repo:

- `data/processed/judged/judged_candidates.jsonl`

Recommended future weak-baseline judged location:

- `data/processed/judged/judged_weak_rag_candidates.jsonl`

Required fields:

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

Selection meaning:

- `storage_bucket = accepted_silver` -> eligible for SFT positive pool
- `storage_bucket = rejected` -> eligible for DPO negative pool
- `storage_bucket = weak_baseline` -> baseline pool, stored separately by provenance

Example judged candidate record:

```json
{
  "candidate_id": "cand_task_paper_001_01_01",
  "task_id": "task_paper_001_01",
  "candidate_source": "strong",
  "judge_model": "strong_judge_B",
  "judge_contract_version": "judge_v0",
  "rubric": {
    "objective_clarity": {"score": 2, "reason": "The comparative objective is explicit."},
    "factor_response_levels": {"score": 2, "reason": "Factors, levels, and responses are stated."},
    "design_choice_appropriateness": {"score": 1, "reason": "The design is sensible but still lightly justified."},
    "interaction_curvature_awareness": {"score": 1, "reason": "There is some staged logic but limited interaction detail."},
    "resource_aware_design": {"score": 2, "reason": "The study is appropriately narrow for a first pass."},
    "execution_feasibility": {"score": 2, "reason": "The listed setup is enough to run the experiment."},
    "documentation_rigor": {"score": 1, "reason": "Measurement detail is present but record-keeping detail is thin."},
    "blocking_awareness": {"score": 1, "reason": "Some nuisance variation is controlled, but not deeply."},
    "iterative_experimentation": {"score": 2, "reason": "The proposal is framed as a first-step experiment with follow-up potential."},
    "bom_compliance": {"score": 2, "reason": "No forbidden items or hidden capabilities are assumed."},
    "claim_discipline": {"score": 1, "reason": "Claims are mostly disciplined but still somewhat broad."}
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
  "summary": "Feasible and resource-aware first-pass proposal that is acceptable for silver-quality training data."
}
```

## Dataset Construction Rules

### SFT

Input pool:

- `accepted_silver`

Default record source:

- judged strong candidates only

Current script alignment:

- `08_build_sft_dataset.py` already consumes `overall_verdict == "accept_silver"`

### DPO

Chosen pool:

- `accepted_silver`

Rejected pool:

- `rejected`

Matching rule:

- pair within the same `task_id`

Current script alignment:

- `09_build_dpo_dataset.py` already groups judged rows by `task_id`

### Validator

Input pool:

- all judged candidates

Recommended `v0` addition:

- preserve `candidate_source` so validator training can later compare strong vs weak-baseline proposal quality

### Weak Baseline

Default `v0` handling:

- keep weak-baseline candidates out of the default SFT positive pool
- keep weak-baseline candidates out of the default DPO `chosen` pool
- optionally use weak-baseline judged rows as extra negatives later

This keeps the baseline set separate until the weak-baseline judge path is actually wired in code.

## Optional Materialized Views

To stay fully compatible with the current repo, bucket membership can remain logical-only in `v0`.

If the team wants materialized bucket files later, the recommended paths are:

- `data/processed/judged/accepted_silver.jsonl`
- `data/processed/judged/rejected.jsonl`
- `data/processed/judged/judged_weak_rag_candidates.jsonl`

These are recommended future views, not current required producer outputs.

## V0 Summary

The `v0` dataset plan makes one key alignment choice:

- keep the current raw JSONL files exactly where the existing scripts expect them

It also makes one key storage choice:

- treat `accepted_silver`, `rejected`, and `weak_baseline` as explicit dataset buckets, with `weak_baseline` defined by provenance rather than by quality score

That keeps the current scaffold usable while giving the integration branch a clear storage contract for later code alignment.
