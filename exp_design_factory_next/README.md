# exp_design_factory

Scaffold for an automated dataset factory for experiment-design model training.

## Pipeline
1. Collect metadata
2. Download PDFs
3. Parse PDFs with GROBID
4. Extract relevant methods/process/equipment chunks
5. Build tasks from evidence
6. Generate strong candidates
7. Generate weak-RAG candidates
8. Judge candidates with rubric + rules
9. Build SFT / DPO / validator datasets
10. Train SFT and DPO models

## Quick start
```bash
python scripts/init_project.py
python scripts/00_collect_metadata.py --config configs/collection/am_niti.yaml
python scripts/03_extract_relevant_chunks.py --config configs/collection/am_niti.yaml
python scripts/04_build_tasks.py --config configs/collection/am_niti.yaml
python scripts/05_generate_candidates.py --collection configs/collection/am_niti.yaml --model configs/models/generator.yaml
python scripts/07_judge_candidates.py --judge configs/models/judge.yaml --rubric configs/rubric/experiment_rubric_v1.yaml
python scripts/08_build_sft_dataset.py
accelerate launch scripts/11_train_sft.py --config configs/models/train.yaml
```

## Semi-manual GPT-web workflow
This path is for exporting prompt files, pasting GPT-web outputs into `manual_raw/`, and importing canonical JSONL files back into the scaffold.

### Candidate generation
```bash
python scripts/05a_export_candidate_prompts.py
# paste one raw GPT-web response per file into:
# data/manual/candidates/manual_raw/
python scripts/05b_import_manual_candidates.py --allow-plain-final-proposal
```

Defaults:
- prompt files: `data/manual/candidates/prompts/`
- pasted raw outputs: `data/manual/candidates/manual_raw/`
- manifest: `data/manual/candidates/manifest.jsonl`
- imported strong candidates: `data/processed/candidates/candidates.jsonl`

If you want a manual weak-baseline pool instead of strong candidates:
```bash
python scripts/05a_export_candidate_prompts.py --candidate-source weak_baseline --n-candidates-per-task 1
python scripts/05b_import_manual_candidates.py --manifest-path data/manual/candidates/manifest.jsonl
```

### Local-only zero-cost workflow
This path uses:
- current main-side `rag1/rag2` outputs as one strong-candidate source
- a local Hugging Face model as another strong-candidate source
- existing `weak_rag` candidates as the weak-baseline source
- rule-first grading before local LLM judging
- manual web judging only for a small calibration subset

For a clean small-batch local-only run, build or refresh the task file first:

```bash
python3.11 scripts/04_build_tasks.py --task-source demo --max-tasks 4
python scripts/05c_import_rag_candidates.py --task-id YOUR_TASK_ID
../.venv/bin/python scripts/05d_generate_local_candidates.py --max-tasks 4 --n-candidates-per-task 4 --replace-output
python scripts/06_generate_weak_rag.py --max-tasks 4 --replace-output
python scripts/07c_rule_grade_candidates.py
../.venv/bin/python scripts/07d_local_judge_candidates.py --replace-output
python scripts/07c_rule_grade_candidates.py --candidate-source weak_baseline
../.venv/bin/python scripts/07d_local_judge_candidates.py --candidate-source weak_baseline --replace-output
python scripts/08_build_sft_dataset.py
python scripts/09_build_dpo_dataset.py
```

Notes:
- `04_build_tasks.py` now supports `--task-source auto|demo|chunks` and `--max-tasks N`. In this worktree, `--task-source demo` is the easiest zero-cost way to create a small multi-task batch.
- The built-in demo batch is no longer a single repeated SEM-style template. It now spans multiple experiment profiles, including porosity imaging, geometry measurement, phase-fraction XRD, microhardness mapping, Archimedes density, and surface profilometry tasks with different equipment lists and dependent-variable structures.
- `05c_import_rag_candidates.py` reads `../outputs/rag1_latest_output.json` and `../outputs/rag2_advice.json` by default and converts them into canonical candidate rows.
- When imported `rag1/rag2` outputs do not carry exact FT `evidence_chunk_ids`, the importer falls back to the matched task's `evidence_chunk_ids` so rule-first grading does not treat them as a false evidence mismatch.
- `05d_generate_local_candidates.py` appends local-model candidates into the same canonical `candidates.jsonl`.
- `05d_generate_local_candidates.py` and `06_generate_weak_rag.py` support `--max-tasks N` and `--replace-output`, which makes it easy to rerun a clean batch without manually deleting old rows.
- `05d_generate_local_candidates.py` now protects `candidate_rank = 1` as a conservative baseline-safe candidate with explicit numeric factor levels, listed resources only, strong controls, clear measurements, and task evidence alignment.
- `05d_generate_local_candidates.py` derives `candidate_rank >= 2` as controlled contrast variants from that protected baseline instead of asking the local LLM to freestyle every variant from scratch. With the current defaults, each task gets one baseline-safe strong row plus three contrast rows.
- The baseline-safe candidate builder is now task-aware. It adapts the measurement plan, resources, controls, and analysis wording to the task profile so different task types do not all collapse into the same SEM-and-porosity proposal structure.
- `07c_rule_grade_candidates.py` tags rule-based failures before any local LLM judging.
- `06_generate_weak_rag.py` now writes deterministic canonical `weak_baseline` candidate rows directly into `data/processed/candidates/weak_rag_candidates.jsonl`.
- `07d_local_judge_candidates.py` writes canonical judged rows for both strong and weak sources and also materializes local-only bucket views under `data/processed/judged/local_only_buckets/` from the canonical strong and weak judged files together.
- `07d_local_judge_candidates.py` supports `--replace-output`, which makes batch reruns safer when you want the judged JSONL to match only the current task batch.
- `07d_local_judge_candidates.py` now uses a compact local-only judge prompt that asks for integer-only rubric scores, attempts one small local repair pass for malformed JSON, and keeps the structural fallback as a safety net.
- `07d_local_judge_candidates.py` treats uniform extreme local rubrics like all-`1` or all-`5` as collapsed low-trust outputs and falls back to structural judging instead of trusting them.
- For `weak_baseline`, the local judge path now canonicalizes contradictory or missing summaries so weak rows read consistently even when the raw local model text is noisy.
- The local-only bucket view `accepted_silver_lite.jsonl` is a convenience sidecar for local-first review. Canonical judged rows still use `storage_bucket == "accepted_silver"` for dataset-builder compatibility.
- Each `07d_local_judge_candidates.py` run prints the regenerated bucket counts for `accepted_silver_lite`, `rejected`, and `weak_baseline`.
- Local generation and judging require `transformers` and `torch`, and the requested model weights must already be available locally or in your Hugging Face cache, so the local-only commands above use `../.venv/bin/python`.
- The default local judge model is now `Qwen/Qwen2-1.5B-Instruct` for a better parseability/speed balance. For faster but weaker judging, you can still pass `--model-name Qwen/Qwen2-0.5B-Instruct`.
- To calibrate the local judge, use `07a_export_judge_prompts.py` and `07b_import_manual_judgments.py` on a small subset only.
- For that calibration subset, you can export only a few candidates with `python scripts/07a_export_judge_prompts.py --limit 10`.

### Judging
```bash
python scripts/07a_export_judge_prompts.py
# paste one raw GPT-web judgment per file into:
# data/manual/judgments/manual_raw/
python scripts/07b_import_manual_judgments.py
```

Defaults:
- prompt files: `data/manual/judgments/prompts/`
- pasted raw outputs: `data/manual/judgments/manual_raw/`
- manifest: `data/manual/judgments/manifest.jsonl`
- imported strong judgments: `data/processed/judged/judged_candidates.jsonl`
- imported weak-baseline judgments: `data/processed/judged/judged_weak_rag_candidates.jsonl`

Fallback behavior in the manual importers:
- `05b_import_manual_candidates.py` accepts raw text with extra prose or fenced code blocks and extracts JSON safely.
- Before JSON extraction, the importers strip a leading BOM and convert U+2028/U+2029 line separators into normal newlines, because pasted GPT-web text sometimes includes them.
- `05b_import_manual_candidates.py --allow-plain-final-proposal` wraps a bare `final_proposal` object with a fallback `reasoning_trace`.
- If `reasoning_trace` is a plain string, the candidate importer wraps it as `{"note": ...}`.
- `07b_import_manual_judgments.py` accepts `rubric` or legacy `scores` in raw judge output.
- Manual judgments now preserve the raw 1..5 rubric under `rubric`, derive `rubric_compatibility` on 0..2 automatically, and compute both `raw_total_score_1to5` and `compatibility_total_score_0to2`.
- `07b_import_manual_judgments.py` continues to compute `overall_verdict` and `storage_bucket` from the compatibility score plus hard-fail logic, and keeps `total_score` as the compatibility-score alias for downstream pipeline compatibility.
- If `rule_checks` are missing in raw judge output, the importer fills deterministic fallbacks from the candidate and task records.
- If `hard_fail_reasons` or `failure_tags` are missing in raw judge output, the importer defaults them to empty lists.
- If a per-dimension rubric `reason` is missing, the importer fills an empty string.
- If `summary` is missing in raw judge output, the importer writes `Manual judgment imported without summary.`

### Dataset builders
```bash
python scripts/08_build_sft_dataset.py
python scripts/09_build_dpo_dataset.py
```

Builder rules:
- `08_build_sft_dataset.py` builds SFT rows from `storage_bucket == "accepted_silver"` only.
- `09_build_dpo_dataset.py` builds DPO pairs from `accepted_silver` vs `rejected` rows for the same `task_id`.
- `weak_baseline` is preserved as a separate provenance bucket and is not silently mixed into SFT or DPO.
- `total_score` in judged rows is treated as the compatibility-score alias, and the builders also preserve `compatibility_total_score_0to2` and `raw_total_score_1to5` when available.

## Notes
- External API calls are placeholders/stubs.
- Replace provider calls inside `05_generate_candidates.py` and `07_judge_candidates.py`.
- GROBID parsing is scaffolded; wire your local or remote GROBID endpoint.
