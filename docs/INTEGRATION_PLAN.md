# Integration Plan

## Purpose

This branch (`chani/integration-ft-v0`) is the safe integration point between:

- the current `main`-side RAG / proposal-generation pipeline
- the `260407_chani` rubric, smoke-analysis, and fine-tuning scaffold

This document records the current file layout and ownership without refactoring or deleting anything.

## Current Branch Snapshot

- `chani/integration-ft-v0` is currently `main` plus one integration commit: `3dd50f2` (`Bring rubric and FT scaffold into integration branch`).
- The root-level `main` pipeline files are still present:
  - `embed.py`
  - `cards_divider.py`
  - `proposal_generator.py`
  - `scientific_advisor.py`
  - `RAG_system.ipynb`
- The integration commit added the Chani-side code surfaces:
  - `rubricsetup.py`
  - `analyze_smoke_run.py`
  - `exp_design_factory_next/`
- The integration branch does **not** currently include the sample artifact directories and text files that existed on `origin/260407_chani`:
  - `generated_prompts/`
  - `proposed_experiments.txt`
  - `smoke_run/`
  - `smoke_run_analysis/`
  - `smoke_run_analysis_by_case/`
  - `NEXT_STEPS_exp_design_factory.md`
  - `NEXT_STEPS_exp_design_factory_EN.md`

## Main-Side Pipeline vs Chani-Side Pipeline

### Main-Side Pipeline

The current `main` side is a root-level RAG and proposal-generation flow. The active script chain is:

1. `embed.py`
   - chunks PDFs and builds an embedding store
   - writes chunk metadata plus vector-store artifacts
2. `cards_divider.py`
   - labels chunks as experiment / science / both / neither
   - builds paper-memory cards and a retrieval store from labeled chunks
3. `proposal_generator.py`
   - retrieves experiment-grounded evidence
   - generates the RAG1 experiment proposal
4. `scientific_advisor.py`
   - reads the RAG1 proposal
   - retrieves science-grounded evidence
   - generates RAG2 advisory output

What this side currently owns:

- literature chunking and retrieval artifacts
- paper-memory card generation
- RAG1 proposal generation
- RAG2 advisory generation

What is important about the current implementation:

- The root scripts are more concrete than the FT scaffold. They contain actual retrieval and generation logic, not just placeholder provider calls.
- `RAG_system.ipynb` exists, but the integration-safe operational contracts are currently encoded in the root Python scripts rather than the notebook.

### Chani-Side Rubric / FT Pipeline

The Chani-side work brought into this branch is a separate rubric and training-data scaffold with two distinct pieces:

- `rubricsetup.py` and `analyze_smoke_run.py`
- `exp_design_factory_next/`

What this side currently owns:

- a detailed prompt-based judging rubric (`rubricsetup.py`)
- smoke-run result parsing and visualization (`analyze_smoke_run.py`)
- a staged dataset factory for building:
  - experiment-design tasks
  - candidate proposals
  - judged candidate records
  - SFT / DPO / validator datasets

What is important about the current implementation:

- `exp_design_factory_next/` is a scaffold, not yet a fully wired production pipeline.
- Several stages use placeholder provider calls.
- The weak-RAG baseline is generated but not yet connected to the judging or dataset-building path.

## Source-of-Truth Files

### 1. RAG / Proposal Generation Outputs

Current source-of-truth files:

- `embed.py`
- `cards_divider.py`
- `proposal_generator.py`
- `scientific_advisor.py`

Current output contracts:

| Area | Source-of-truth file(s) | Current canonical output(s) |
| --- | --- | --- |
| Literature chunking and embedding | `embed.py` | `lit_embedding_store/chunks.jsonl`, `lit_embedding_store/storage/`, `lit_embedding_store/summary.json` by default |
| Chunk labeling and paper-memory cards | `cards_divider.py` | `outputs/chunks_labeled.jsonl`, `outputs/paper_memory_cards.jsonl`, `outputs/paper_memory_storage/` |
| RAG1 proposal | `proposal_generator.py` | `outputs/rag1_latest_output.json` |
| RAG2 advice | `scientific_advisor.py` | `outputs/rag2_advice.json` |

Important integration note:

- The root RAG path contracts are not fully aligned yet.
- `embed.py` defaults to `lit_embedding_store/...`, while `cards_divider.py` defaults to reading `outputs/chunks.jsonl`.
- `cards_divider.py` writes one combined `outputs/paper_memory_storage/`, while:
  - `proposal_generator.py` expects `outputs/paper_memory_storage_experiment`
  - `scientific_advisor.py` expects `outputs/paper_memory_storage_science`

For integration purposes, the root scripts above are still the source of truth, but their intermediate-path contract needs to be unified later.

### 2. Rubric / Judging

Current source-of-truth files:

- `rubricsetup.py`
- `exp_design_factory_next/configs/rubric/experiment_rubric_v1.yaml`
- `exp_design_factory_next/configs/rubric/rule_checks_v1.yaml`
- `exp_design_factory_next/scripts/07_judge_candidates.py`

Current ownership split:

- `rubricsetup.py` is the richest and most explicit rubric definition in the repo today.
  - It contains the detailed evaluation prompt
  - It defines the rubric fields
  - It defines hard-fail logic and verdict bands
  - It generates prompt artifacts under `generated_prompts/`
- `exp_design_factory_next/configs/rubric/*.yaml` is the machine-readable scaffold for the future FT pipeline.
- `exp_design_factory_next/scripts/07_judge_candidates.py` is the automated judging stage inside the FT scaffold.

Important integration note:

- The scaffold judge is not yet fully wired to the rubric configs.
- `07_judge_candidates.py` currently uses placeholder `llm_judge_call(...)` logic and internal `rule_checks(...)`.
- The `--judge` and `--rubric` args are accepted, but the script does not yet load YAML rubric content or prompt files.

Practical conclusion:

- Today, the detailed rubric logic lives in `rubricsetup.py`.
- The FT pipeline's future integration point is `exp_design_factory_next/scripts/07_judge_candidates.py` plus the YAML configs.
- These two judging definitions should be unified later, but not in this documentation-only pass.

### 3. Smoke-Run Analysis

Current source-of-truth files:

- `analyze_smoke_run.py`

Expected I/O contract:

- input directory: `smoke_run/`
- output directory: `smoke_run_analysis_by_case/`

Important integration note:

- The analysis script is present on this branch, but the sample inputs and generated analysis artifacts are not.
- The example smoke-run data and generated reports currently exist only on `origin/260407_chani`, not in this integration checkout.

Practical conclusion:

- The analysis logic source of truth is `analyze_smoke_run.py`.
- The checked-in smoke-run example artifacts are not yet part of the safe integration branch.

### 4. Fine-Tuning Dataset Building

Current source-of-truth files:

- `exp_design_factory_next/scripts/04_build_tasks.py`
- `exp_design_factory_next/scripts/05_generate_candidates.py`
- `exp_design_factory_next/scripts/06_generate_weak_rag.py`
- `exp_design_factory_next/scripts/07_judge_candidates.py`
- `exp_design_factory_next/scripts/08_build_sft_dataset.py`
- `exp_design_factory_next/scripts/09_build_dpo_dataset.py`
- `exp_design_factory_next/scripts/10_build_validator_dataset.py`

Current dataset contracts:

| Stage | Source-of-truth file | Output |
| --- | --- | --- |
| Build tasks | `04_build_tasks.py` | `data/processed/tasks/tasks.jsonl` |
| Generate strong candidates | `05_generate_candidates.py` | `data/processed/candidates/candidates.jsonl` |
| Generate weak-RAG candidates | `06_generate_weak_rag.py` | `data/processed/candidates/weak_rag_candidates.jsonl` |
| Judge candidates | `07_judge_candidates.py` | `data/processed/judged/judged_candidates.jsonl` |
| Build SFT dataset | `08_build_sft_dataset.py` | `data/processed/datasets/sft/train.jsonl` |
| Build DPO dataset | `09_build_dpo_dataset.py` | `data/processed/datasets/dpo/train.jsonl` |
| Build validator dataset | `10_build_validator_dataset.py` | `data/processed/datasets/validator/train.jsonl` |

Important integration note:

- The weak-RAG branch is not yet wired into judging or dataset creation.
- `06_generate_weak_rag.py` writes `weak_rag_candidates.jsonl`, but the downstream judged / dataset scripts currently read only `candidates.jsonl`.

Practical conclusion:

- The FT dataset-building source of truth is the `04` through `10` script chain under `exp_design_factory_next/scripts/`.
- Training consumers are downstream:
  - `11_train_sft.py`
  - `12_train_dpo.py`

## Files Currently Coming From Each Side

### Currently From `main`

These files represent the current teammate pipeline for RAG1 / RAG2 / related outputs:

- `README.md`
- `RAG_system.ipynb`
- `embed.py`
- `cards_divider.py`
- `proposal_generator.py`
- `scientific_advisor.py`
- `requirements.txt`

### Currently From Chani / `260407_chani`

These are the Chani-side additions already staged into this integration branch:

- `rubricsetup.py`
- `analyze_smoke_run.py`
- `exp_design_factory_next/`

These Chani-side artifacts still exist only on the original branch and were **not** copied into this safe integration branch:

- `generated_prompts/`
- `proposed_experiments.txt`
- `smoke_run/`
- `smoke_run_analysis/`
- `smoke_run_analysis_by_case/`
- `NEXT_STEPS_exp_design_factory.md`
- `NEXT_STEPS_exp_design_factory_EN.md`

## Integration Gaps To Preserve and Resolve Later

These are the main gaps that should be resolved in a later refactor pass, but not changed yet:

1. Path-contract mismatch inside the root RAG pipeline
   - `embed.py` defaults to `lit_embedding_store/...`
   - downstream scripts expect `outputs/...`

2. Retrieval-store mismatch inside the root RAG pipeline
   - `cards_divider.py` writes one `paper_memory_storage/`
   - downstream scripts expect separate experiment and science stores

3. Dual rubric definitions
   - `rubricsetup.py` contains the detailed rubric prompt
   - `exp_design_factory_next/configs/rubric/*.yaml` contains a lighter scaffold definition

4. Placeholder FT scaffold stages
   - `05_generate_candidates.py` and `07_judge_candidates.py` still use stub provider logic

5. Weak-RAG path not yet connected
   - `weak_rag_candidates.jsonl` is produced but not judged or converted into training datasets

6. Missing sample artifacts in the integration branch
   - the integration branch contains the logic, but not the sample prompt/smoke outputs from the original Chani branch

## Safe Integration Plan

The safe next integration sequence should be:

1. Freeze file ownership first
   - Keep the root `main` scripts as the source of truth for RAG1 / RAG2 generation
   - Keep `exp_design_factory_next/` as the source of truth for FT dataset scaffolding
   - Keep `rubricsetup.py` as the source of truth for the detailed rubric text until the scaffold judge is wired to configs

2. Normalize contracts before refactoring behavior
   - choose one intermediate artifact namespace
   - choose one paper-memory store layout
   - choose one rubric schema and verdict vocabulary

3. Decide artifact policy
   - decide whether `generated_prompts/`, `smoke_run/`, and analysis outputs should be:
     - versioned example artifacts
     - ignored runtime outputs
     - or moved under a dedicated `examples/` or `reports/` area

4. Wire the FT scaffold to real judging and baseline comparison
   - connect `07_judge_candidates.py` to the rubric config and prompt assets
   - decide whether weak-RAG outputs should be judged in the same pipeline

5. Only after the above, refactor directory structure
   - no deletes or code moves should happen before the ownership and artifact contracts are explicit

## Short Working Summary

If we treat this branch as the safe integration point today:

- root-level scripts remain the current source of truth for RAG / proposal generation
- `rubricsetup.py` remains the current source of truth for the detailed judging rubric
- `analyze_smoke_run.py` remains the current source of truth for smoke-run analysis logic
- `exp_design_factory_next/scripts/04` through `10` remain the current source of truth for FT dataset building

The next pass should unify contracts and wiring, not re-decide ownership from scratch.
