# Judge module

This folder adds a rubric-based proposal judge to the Agents_4_expermients repo.

## Files
- `rubricsetup.py`: judge rubric prompt
- `common.py`: shared prompt-building utilities
- `run_local_judge_candidates.py`: local judging script
- `samples/`: small example inputs/outputs

## Purpose
Given an experiment-design proposal and task context, the judge scores:
- feasibility
- BOM compliance
- methodological quality
- overall recommendation

## Notes
This folder is added without changing the core RAG generation pipeline.
