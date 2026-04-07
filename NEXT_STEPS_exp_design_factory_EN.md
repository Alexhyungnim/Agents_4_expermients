# Next Steps for `exp_design_factory_next`

This document tracks the immediate roadmap for integrating the new experiment-design training scaffold into the existing `Agents_4_expermients` repository.

## 1. Immediate Repository Actions

- Keep `exp_design_factory_next/` as a separate subfolder for now.
- Do **not** merge its files into the repository root yet.
- Do **not** overwrite the current `README.md`, `requirements.txt`, notebook files, or existing scripts.
- Commit the scaffold directory first so that the project structure is safely versioned.

### Recommended commands

```bash
git add exp_design_factory_next
# optionally also add this roadmap file if placed in repo root
# git add NEXT_STEPS_exp_design_factory_EN.md

git commit -m "Add experiment design scaffold and roadmap"
git push
```

---

## 2. Priority Files to Review First

These are the highest-priority files in the new scaffold.

### Priority 1
`exp_design_factory_next/scripts/03_extract_relevant_chunks.py`
- Compare this file against the current chunk filtering logic in `RAG_system.ipynb`.
- Reuse the strongest parts of the notebook logic here.
- Confirm that chunk scoring, bad-section filtering, and metadata writing match the current project needs.

### Priority 2
`exp_design_factory_next/scripts/04_build_tasks.py`
- Decide how many tasks should be created per paper.
- Start with 2-4 tasks per paper instead of trying to support everything at once.
- Focus first on:
  - experiment design
  - measurement plan
  - feasibility check
  - revision of weak proposals

### Priority 3
`exp_design_factory_next/scripts/05_generate_candidates.py`
- Replace the placeholder LLM call with the actual provider/API.
- Decide which strong model to use for candidate generation.
- Keep output strictly schema-constrained.

### Priority 4
`exp_design_factory_next/scripts/07_judge_candidates.py`
- Replace the placeholder judge call with the actual evaluator model/API.
- Lock down the rubric format and verdict thresholds.
- Add additional rule-based checks for unsupported equipment, empty controls, evidence mismatch, and missing replicates.

### Priority 5
`exp_design_factory_next/scripts/11_train_sft.py`
- Verify that the script runs in the local environment.
- Confirm model choice, tokenizer loading, QLoRA settings, and JSONL dataset loading.

---

## 3. Mapping Between the Existing Repo and the New Scaffold

The current repository already contains useful prototype logic.

### Existing repo components
- `RAG_system.ipynb`
- `rubricsetup.py`
- `analyze_smoke_run.py`
- `smoke_run/`
- `smoke_run_analysis/`
- `smoke_run_analysis_by_case/`

### Suggested mapping

#### `RAG_system.ipynb`
Use as source material for:
- `03_extract_relevant_chunks.py`
- `04_build_tasks.py`
- `05_generate_candidates.py`

#### `rubricsetup.py`
Use as source material for:
- rubric YAML files in `configs/rubric/`
- `07_judge_candidates.py`
- `10_build_validator_dataset.py`

#### `analyze_smoke_run.py`
Use as source material for:
- judge result aggregation
- validator dataset auditing
- score distribution plots
- acceptance/rejection diagnostics

#### `smoke_run*` outputs
Use as source material for:
- weak/rejected example pools
- rubric error-type analysis
- future DPO rejected examples

---

## 4. Recommended Short-Term Development Order

### Phase A: Make the pipeline executable end-to-end once
Goal: run the scaffold once with a very small batch.

Tasks:
- [ ] wire real LLM API into `05_generate_candidates.py`
- [ ] wire real judge API into `07_judge_candidates.py`
- [ ] confirm `03_extract_relevant_chunks.py` works on a small paper set
- [ ] run 5-10 papers through the pipeline
- [ ] inspect accepted vs rejected outputs manually

### Phase B: Lock the schemas
Goal: stabilize the data format before scaling.

Tasks:
- [ ] finalize `task` JSON schema
- [ ] finalize `candidate` JSON schema
- [ ] finalize `judged_candidate` JSON schema
- [ ] finalize SFT JSONL format
- [ ] finalize DPO JSONL format

### Phase C: Build the first useful dataset
Goal: create a small but valid training dataset.

Targets:
- 50-100 accepted-silver examples
- 100-300 rejected examples
- 20-30 manually audited examples for calibration

### Phase D: Run the first SFT baseline
Goal: verify that the training stack works.

Tasks:
- [ ] generate `data/processed/datasets/sft/train.jsonl`
- [ ] run `11_train_sft.py`
- [ ] test the resulting model on held-out tasks
- [ ] compare against the untuned baseline

### Phase E: Add DPO
Goal: use accepted vs rejected examples for preference learning.

Tasks:
- [ ] generate `data/processed/datasets/dpo/train.jsonl`
- [ ] run `12_train_dpo.py`
- [ ] compare SFT-only vs SFT+DPO

---

## 5. Data Strategy

### Accepted examples
Use these for SFT.
These should come from:
- strong LLM candidate generation
- independent judge approval
- rule-based validation pass

### Rejected examples
Use these for:
- DPO rejected examples
- validator training
- failure analysis

Sources:
- weak RAG generation
- rubric-failing candidates
- evidence-mismatch candidates
- deliberately corrupted or weak variants

### Important rule
Do **not** mix weak/rejected examples into the SFT gold/silver pool.
Weak outputs should strengthen the preference and validator stages, not the supervised target set.

---

## 6. Recommended Initial Scope

To avoid overbuilding too early, start with a narrow domain.

### First domain
- additive manufacturing
- NiTi / DED / shape memory alloy papers

### Later expansion
- welding design
- battery experiment design
- cross-domain transfer evaluation

### Initial paper budget
Start with:
- 20-40 papers for pipeline validation
- then expand to 100+ once schemas and judging are stable

---

## 7. First Experiments to Run

### Experiment 1
Strong generator + judge + rule checks on 10 papers.

Question:
- Are accepted proposals actually useful and evidence-grounded?

### Experiment 2
Weak RAG proposals vs strong proposals on the same tasks.

Question:
- Are the rejected examples meaningfully worse, or just stylistically different?

### Experiment 3
SFT on accepted-silver only.

Question:
- Can a small open model learn the structured proposal format reliably?

### Experiment 4
DPO with accepted vs rejected examples.

Question:
- Does the model learn to avoid common bad proposal patterns?

---

## 8. Suggested Acceptance Logic for Now

A proposal should be marked `accept_silver` only if:
- total rubric score is above threshold
- no hard fail is triggered
- no unsupported equipment is used
- no evidence mismatch is detected
- controls are not empty
- replicates are not empty when required

Everything else should be routed to `reject` or `weak_reject`.

---

## 9. This Week’s Practical Checklist

- [ ] commit `exp_design_factory_next/` into the repo
- [ ] add this roadmap file to the repo
- [ ] review `03_extract_relevant_chunks.py`
- [ ] review `04_build_tasks.py`
- [ ] connect real API calls in `05_generate_candidates.py`
- [ ] connect real judge calls in `07_judge_candidates.py`
- [ ] run the pipeline on a very small paper subset
- [ ] inspect outputs manually
- [ ] refine rubric thresholds
- [ ] build first SFT dataset

---

## 10. Medium-Term Goal

The medium-term goal is not just to retrieve and summarize papers.
It is to build a fully structured training pipeline that can:
- collect papers automatically
- extract evidence automatically
- generate structured experiment proposals automatically
- score them automatically
- separate accepted and rejected examples automatically
- train a smaller open-source model using SFT and DPO
- later evaluate transfer across different engineering domains

---

## 11. Final Principle

Do **not** try to fully merge the scaffold into the existing repository immediately.

First:
- stabilize the scaffold
- make it run end-to-end once
- confirm the schemas
- confirm the evaluator
- confirm the training loop

Only after that should selected files be promoted into the main repo structure.
