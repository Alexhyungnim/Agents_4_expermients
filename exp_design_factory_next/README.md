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
- `07b_import_manual_judgments.py` computes canonical `total_score`, `detailed_verdict`, `overall_verdict`, and `storage_bucket` during import.
- If `rule_checks` are missing in raw judge output, the importer fills deterministic fallbacks from the candidate and task records.
- If `hard_fail_reasons` or `failure_tags` are missing in raw judge output, the importer defaults them to empty lists.
- If a per-dimension rubric `reason` is missing, the importer fills an empty string.
- If `summary` is missing in raw judge output, the importer writes `Manual judgment imported without summary.`

## Notes
- External API calls are placeholders/stubs.
- Replace provider calls inside `05_generate_candidates.py` and `07_judge_candidates.py`.
- GROBID parsing is scaffolded; wire your local or remote GROBID endpoint.
