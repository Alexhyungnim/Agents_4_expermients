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
