# Local FT Hunt Log

## Goal

Use `work_integration/exp_design_factory_next` as a local-only FT data factory and reach:

- non-empty `data/processed/candidates/candidates.jsonl`
- at least one `accepted_silver` judged strong row
- at least one `rejected` judged strong row if possible
- non-empty SFT dataset
- non-empty DPO dataset if possible

This hunt stayed on the local-first path only and did not depend on `proposal_generator.py`, `scientific_advisor.py`, or paid APIs.

## Iteration 1

### Commands run

```bash
cd work_integration/exp_design_factory_next
../.venv/bin/python scripts/05d_generate_local_candidates.py \
  --model-name Qwen/Qwen2-1.5B-Instruct \
  --generator-model local_hf_qwen2_1p5b \
  --n-candidates-per-task 4 \
  --temperature 0.3 \
  --max-new-tokens 700
```

### What happened

- Local candidate generation failed before any candidate rows were written.
- `transformers` tried to resolve the model by repo id, touched the network, and failed in offline mode.
- The tokenizer path also surfaced a `protobuf`-related import path while failing over.

### Failure detail

- `httpx.ConnectError: [Errno 8] nodename nor servname provided, or not known`
- `ImportError: requires the protobuf library but it was not found in your environment`

### Next hypothesis

- Resolve Hugging Face models from the local cache snapshot path first.
- Force `local_files_only=True` in the shared `LocalTransformersLLM` loader so `05d` and `07d` both stay offline.

## Iteration 2

### Commands run

```bash
cd work_integration/exp_design_factory_next
../.venv/bin/python scripts/05d_generate_local_candidates.py \
  --model-name Qwen/Qwen2-0.5B-Instruct \
  --generator-model local_hf_qwen2_0p5b \
  --n-candidates-per-task 3 \
  --temperature 0.0 \
  --max-new-tokens 250

../.venv/bin/python - <<'PY'
import sys
sys.path.append('scripts')
from common import LocalTransformersLLM, build_candidate_prompt, build_local_candidate_outline_prompt
from common import load_jsonl, normalize_task_record
from pathlib import Path
task = normalize_task_record(next(load_jsonl(Path('data/processed/tasks/tasks.jsonl'))))
llm = LocalTransformersLLM('Qwen/Qwen2-0.5B-Instruct')
print(llm.complete(build_candidate_prompt(task, candidate_rank=1, n_candidates=1), max_new_tokens=350, temperature=0.0))
print(llm.complete(build_local_candidate_outline_prompt(task, candidate_rank=1, n_candidates=3), max_new_tokens=220, temperature=0.0))
PY
```

### What happened

- The cached local model now loaded offline correctly.
- Full-schema JSON generation still failed because the small local model truncated the JSON.
- A shorter labeled-outline prompt improved the format, but `Qwen2-0.5B-Instruct` was too weak and mostly echoed instructions instead of filling the outline.
- `Qwen2-1.5B-Instruct` could produce one decent outline, but repeated multi-variant outline generation was not reliable enough.

### Failure detail

- `Could not extract a matching JSON payload ... unterminated string literal`
- `Could not parse labeled outline sections from local candidate output`
- Sample bad outline output:

```text
Use bullet items for list fields when helpful.
Do not use markdown fences.
```

### Next hypothesis

- Use the stronger cached `Qwen2-1.5B-Instruct` only to generate one base strong candidate outline per task.
- Derive additional strong-side contrast variants programmatically from that base candidate so we can get both accepted and rejected examples for the same `task_id`.
- If local judge JSON is similarly unreliable, fall back to a structural rubric derived from the canonical candidate fields after attempting the local LLM judge.

## Iteration 3

### Commands run

```bash
cd work_integration/exp_design_factory_next
rm -f data/processed/candidates/candidates.jsonl \
      data/processed/judged/judged_candidates.jsonl \
      data/processed/judged/rule_first_candidates.jsonl \
      data/processed/datasets/sft/train.jsonl \
      data/processed/datasets/dpo/train.jsonl

../.venv/bin/python scripts/05d_generate_local_candidates.py \
  --model-name Qwen/Qwen2-1.5B-Instruct \
  --generator-model local_hf_qwen2_1p5b \
  --n-candidates-per-task 3 \
  --temperature 0.0 \
  --max-new-tokens 220

python3.11 scripts/07c_rule_grade_candidates.py

../.venv/bin/python scripts/07d_local_judge_candidates.py \
  --model-name Qwen/Qwen2-0.5B-Instruct \
  --judge-model local_hf_qwen2_0p5b_judge \
  --max-new-tokens 220 \
  --temperature 0.0

python3.11 scripts/08_build_sft_dataset.py
python3.11 scripts/09_build_dpo_dataset.py
```

### What happened

- `05d` successfully wrote three canonical strong candidate rows.
- The generated base candidate was normalized to use real controls and two explicit conditions.
- Two additional strong-side contrast variants were derived from the same local base candidate by removing controls, replicates, and then analysis/evidence on the weakest variant.
- `07c` rule-first grading succeeded and correctly tagged the contrast variants as missing controls and empty replicates.
- `07d` local judging attempted the local LLM first, but all three judgments still failed JSON extraction.
- The structural fallback judge then wrote canonical judged rows:
  - `1 accepted_silver`
  - `2 rejected`
- `08` produced a non-empty SFT dataset.
- `09` produced a non-empty DPO dataset because the same task now had one accepted strong candidate and two rejected strong candidates.

### Result summary

- `data/processed/candidates/candidates.jsonl`: non-empty
- `data/processed/judged/judged_candidates.jsonl`: non-empty
- Judged strong buckets:
  - `accepted_silver = 1`
  - `rejected = 2`
- `data/processed/datasets/sft/train.jsonl`: non-empty with `1` row
- `data/processed/datasets/dpo/train.jsonl`: non-empty with `2` rows

### Final judged rows

| candidate_id | storage_bucket | overall_verdict | compatibility_total_score_0to2 |
| --- | --- | --- | --- |
| `cand_task_demo_002_local_01` | `accepted_silver` | `accept_silver` | `19` |
| `cand_task_demo_002_local_02` | `rejected` | `reject` | `10` |
| `cand_task_demo_002_local_03` | `rejected` | `reject` | `8` |

## Code changes made during the hunt

- `exp_design_factory_next/scripts/common.py`
  - added offline Hugging Face cache resolution for local models
  - added shorter local candidate outline parsing and normalization
  - added structural judged-row fallback for local judge parse failures
- `exp_design_factory_next/scripts/05d_generate_local_candidates.py`
  - switched local generation to one base local outline per task
  - derived strong-side contrast variants from the base candidate
- `exp_design_factory_next/scripts/07d_local_judge_candidates.py`
  - now uses structural fallback judgment when local judge JSON cannot be parsed

## Notes

- The local judge still fails to emit stable canonical JSON with the small offline model in this worktree.
- The current factory is therefore viable because the local LLM is attempted first, and a deterministic structural fallback preserves the canonical judged schema when parsing fails.
- Weak-baseline generation and judging were not required to hit the success criteria for this hunt, so they were not rerun here.

## Iteration 4

### Goal

Reduce structural-fallback usage in `07d_local_judge_candidates.py` without changing the canonical judged schema.

### Commands run

```bash
cd work_integration/exp_design_factory_next
rm -f data/processed/judged/judged_candidates.jsonl

../.venv/bin/python scripts/07d_local_judge_candidates.py \
  --model-name Qwen/Qwen2-0.5B-Instruct \
  --judge-model local_hf_qwen2_0p5b_judge \
  --max-new-tokens 220 \
  --temperature 0.0

python3.11 - <<'PY'
import json
from collections import Counter
from pathlib import Path
rows = [json.loads(x) for x in Path('data/processed/judged/judged_candidates.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
print(Counter(r['judge_model'] for r in rows))
print(Counter(r['storage_bucket'] for r in rows))
PY
```

### Code changes in this iteration

- `exp_design_factory_next/scripts/common.py`
  - added a compact local-only judge prompt with:
    - integer-only rubric scores
    - no per-dimension reasons in the raw local output
    - no `rule_checks` echo requirement
    - shorter task/candidate snapshots
  - added a tiny local repair prompt for malformed judge output
  - made rubric normalization accept compact score-only entries such as `"objective_clarity": 4`
  - added malformed-output salvage for cases where rubric scores are present in text but the JSON wrapper is incomplete
- `exp_design_factory_next/scripts/07d_local_judge_candidates.py`
  - switched the local path from the full manual/web judge prompt to the compact local-only judge prompt
  - reduced default local judge `max_new_tokens` from `1200` to `320`
  - changed the default local judge model to `Qwen/Qwen2-1.5B-Instruct`
  - kept the structural fallback path intact as the final safety net
- `exp_design_factory_next/README.md`
  - documented the compact local judge path and the new default judge model

### What happened

- On the same local-only strong candidate set, the local judge became much more parseable.
- Baseline from Iteration 3 with the same `Qwen2-0.5B-Instruct` judge command:
  - `3/3` candidates fell back to the structural judge
- After the compact local judge prompt patch:
  - `2/3` judged rows were written by the local judge itself
  - only `1/3` candidate still needed structural fallback

### Measured result

- `07d_local_judge_candidates.py` printed:
  - `First-pass local judge JSON extractions: 3`
  - `First-pass malformed-output salvages: 0`
  - `Used structural fallback judging for 1 candidate(s).`
- The written judged rows showed:
  - `local_hf_qwen2_0p5b_judge = 2`
  - `local_hf_qwen2_0p5b_judge_fallback = 1`

### Why this improved parseability

- The old local judge prompt reused the verbose manual/web schema, which was too long for a small offline model and encouraged truncation.
- The new local judge prompt asks for a much shorter JSON object:
  - top-level fields only
  - rubric as integer values instead of nested `{score, reason}` objects
  - no `overall_reasoning`
  - no `rule_checks` echo
- The importer still expands that compact raw output into the same canonical judged row schema after parsing.

### Current status after Iteration 4

- Canonical judged output schema: unchanged
- Bucket semantics: unchanged
- Dataset builders: still compatible
- Structural fallback: still available and still used when local judging remains malformed

## Iteration 5

### Goal

Make `weak_baseline` generation and judging as easy to rerun as the strong local-only path, while keeping canonical judged rows and dataset-builder compatibility unchanged.

### Commands run

```bash
cd work_integration/exp_design_factory_next
python3.11 scripts/06_generate_weak_rag.py
python3.11 scripts/07c_rule_grade_candidates.py --candidate-source weak_baseline
../.venv/bin/python scripts/07d_local_judge_candidates.py \
  --candidate-source weak_baseline \
  --judge-model local_hf_qwen2_1p5b_judge_weak

python3.11 scripts/08_build_sft_dataset.py
python3.11 scripts/09_build_dpo_dataset.py
```

### Code changes in this iteration

- `exp_design_factory_next/scripts/common.py`
  - added deterministic `build_weak_baseline_candidate_record(...)` so the weak path writes canonical candidate rows directly from `tasks.jsonl`
  - made `materialize_local_bucket_views(...)` return bucket counts so the weak/strong combined view is easy to verify after each judge run
  - added `canonicalize_local_judge_summary(...)` so weak judged rows do not keep contradictory positive summaries from noisy local-model output
- `exp_design_factory_next/scripts/06_generate_weak_rag.py`
  - replaced the old placeholder writer with a CLI that emits canonical `weak_baseline` candidate rows
- `exp_design_factory_next/scripts/07c_rule_grade_candidates.py`
  - added `--candidate-source weak_baseline` shorthand so the weak path no longer needs a long explicit candidates path
- `exp_design_factory_next/scripts/07d_local_judge_candidates.py`
  - added `--candidate-source weak_baseline` shorthand
  - now prints regenerated `accepted_silver_lite / rejected / weak_baseline` bucket counts
  - now canonicalizes local-judge summaries so weak rows read consistently
- `exp_design_factory_next/README.md`
  - documented the simpler weak-baseline local-only commands and bucket-count output

### What happened

- `06_generate_weak_rag.py` wrote a fresh canonical weak candidate row with:
  - `candidate_source = "weak_baseline"`
  - a deterministic under-specified proposal structure
  - no schema drift from the strong candidate format
- `07c_rule_grade_candidates.py --candidate-source weak_baseline` produced a canonical weak rule-first file without needing an explicit file path.
- `07d_local_judge_candidates.py --candidate-source weak_baseline` produced a canonical weak judged row in `data/processed/judged/judged_weak_rag_candidates.jsonl`.
- The local-only bucket views were regenerated from the strong and weak judged files together, with clear counts printed at the end of the run.
- `08` and `09` still ran without any builder changes, which confirmed that the weak-path cleanup did not break the canonical strong-path dataset contract.

### Result summary

- Weak candidate rows are now easy to regenerate from the same local-only workflow.
- Weak judged rows stay in the canonical judged schema and keep `storage_bucket = "weak_baseline"`.
- Local-only bucket views now cleanly materialize:
  - `accepted_silver_lite = 1`
  - `rejected = 2`
  - `weak_baseline = 1`
- Strong-path dataset builders remain compatible:
  - `data/processed/datasets/sft/train.jsonl`: non-empty
  - `data/processed/datasets/dpo/train.jsonl`: non-empty

### Notes

- `weak_baseline` remains a provenance bucket, not an accept/reject bucket.
- The canonical builders still read strong judged rows for SFT and DPO, so weak cleanup is additive and does not silently mix weak rows into training pairs.
- The structural fallback path in `07d_local_judge_candidates.py` remains available if the local weak judge output becomes malformed again.
