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

## Notes
- External API calls are placeholders/stubs.
- Replace provider calls inside `05_generate_candidates.py` and `07_judge_candidates.py`.
- GROBID parsing is scaffolded; wire your local or remote GROBID endpoint.
