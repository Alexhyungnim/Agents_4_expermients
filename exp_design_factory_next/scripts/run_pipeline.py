from __future__ import annotations
import subprocess

STEPS = [
    ["python", "scripts/00_collect_metadata.py", "--config", "configs/collection/am_niti.yaml"],
    ["python", "scripts/01_download_pdfs.py", "--config", "configs/collection/am_niti.yaml"],
    ["python", "scripts/02_parse_with_grobid.py", "--config", "configs/collection/am_niti.yaml"],
    ["python", "scripts/03_extract_relevant_chunks.py", "--config", "configs/collection/am_niti.yaml"],
    ["python", "scripts/04_build_tasks.py", "--config", "configs/collection/am_niti.yaml"],
    ["python", "scripts/05_generate_candidates.py", "--collection", "configs/collection/am_niti.yaml", "--model", "configs/models/generator.yaml"],
    ["python", "scripts/06_generate_weak_rag.py"],
    ["python", "scripts/07_judge_candidates.py", "--judge", "configs/models/judge.yaml", "--rubric", "configs/rubric/experiment_rubric_v1.yaml"],
    ["python", "scripts/08_build_sft_dataset.py"],
    ["python", "scripts/09_build_dpo_dataset.py"],
    ["python", "scripts/10_build_validator_dataset.py"],
]

for step in STEPS:
    print("RUN:", " ".join(step))
    subprocess.run(step, check=True)
