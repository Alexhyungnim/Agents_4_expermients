# Output Schema V0

## Purpose

This document defines the provisional `v0` output contract for the current main-side RAG / proposal pipeline.

Goals:

- stay compatible with the current repo layout
- document the current producer outputs without refactoring code
- resolve the main path-contract ambiguity at the documentation level

This is a documentation-only alignment pass. No files or paths are changed here.

## Scope

This contract covers the current root-level pipeline:

1. `embed.py`
2. `cards_divider.py`
3. `proposal_generator.py`
4. `scientific_advisor.py`

It does not define the FT dataset JSONL contracts. Those are documented in:

- [DATASET_PLAN_V0.md](/Users/chanisong/Desktop/2026S/24880/project/260412_agents_ft/work_integration/docs/DATASET_PLAN_V0.md)
- [JUDGE_CONTRACT_V0.md](/Users/chanisong/Desktop/2026S/24880/project/260412_agents_ft/work_integration/docs/JUDGE_CONTRACT_V0.md)

## Provisional V0 Path Map

The current repo has one known path mismatch between producers and consumers. For `v0`, the path contract is:

| Semantic artifact | Current producer | Current path | V0 status |
| --- | --- | --- | --- |
| Raw embedded chunks | `embed.py` | `lit_embedding_store/chunks.jsonl` | canonical producer output |
| Raw embedding index | `embed.py` | `lit_embedding_store/storage/` | canonical producer output |
| Embedding run summary | `embed.py` | `lit_embedding_store/summary.json` | canonical producer output |
| Labeled chunks | `cards_divider.py` | `outputs/chunks_labeled.jsonl` | canonical producer output |
| Paper memory cards | `cards_divider.py` | `outputs/paper_memory_cards.jsonl` | canonical producer output |
| Combined paper memory index | `cards_divider.py` | `outputs/paper_memory_storage/` | canonical producer output |
| Latest RAG1 proposal | `proposal_generator.py` | `outputs/rag1_latest_output.json` | canonical producer output |
| Latest RAG2 advice | `scientific_advisor.py` | `outputs/rag2_advice.json` | canonical producer output |

## Alignment Rules

### 1. Combined Paper-Memory Store Is Canonical In V0

For `v0`, the only documented producer-owned paper-memory retrieval store is:

- `outputs/paper_memory_storage/`

This is important because `cards_divider.py` currently writes one combined store, while:

- `proposal_generator.py` expects `outputs/paper_memory_storage_experiment`
- `scientific_advisor.py` expects `outputs/paper_memory_storage_science`

The split experiment/science store names are treated as future consumer aliases, not as current canonical producer outputs.

### 2. `lit_embedding_store/` Remains Valid In V0

`embed.py` currently writes to `lit_embedding_store/` by default. That remains valid for `v0`.

The `outputs/` directory is the canonical home for downstream RAG artifacts, but `embed.py` is not redefined here.

### 3. `rag1_latest_output.json` And `rag2_advice.json` Are Latest-Run Singleton Files

For `v0`, the main-side pipeline writes the latest run into singleton JSON files:

- `outputs/rag1_latest_output.json`
- `outputs/rag2_advice.json`

These are not append-only history files. They are current-state outputs.

## JSONL And JSON Schemas

### 1. `lit_embedding_store/chunks.jsonl`

Producer:

- `embed.py`

Format:

- JSONL
- one row per chunk extracted from a PDF

Required fields:

- `source_path`: string
- `file_name`: string
- `title`: string
- `chunk_id`: integer
- `chunk_start`: integer
- `chunk_end`: integer
- `char_length`: integer
- `text`: string

Notes:

- this is the raw chunk source for later labeling
- `doi` is not guaranteed at this stage

Example:

```json
{
  "source_path": "/abs/path/lit/sample_paper.pdf",
  "file_name": "sample_paper.pdf",
  "title": "sample paper",
  "chunk_id": 7,
  "chunk_start": 12600,
  "chunk_end": 14380,
  "char_length": 1780,
  "text": "Experimental procedure: friction welding was performed on..."
}
```

### 2. `outputs/chunks_labeled.jsonl`

Producer:

- `cards_divider.py`

Format:

- JSONL
- one row per labeled chunk

Required fields:

- `source_path`: string
- `file_name`: string
- `title`: string
- `doi`: string
- `chunk_id`: integer
- `chunk_start`: integer
- `chunk_end`: integer
- `char_length`: integer
- `chunk_label`: `experiment | science | both | neither`
- `chunk_confidence`: number
- `chunk_reason`: string
- `experiment_score`: integer
- `science_score`: integer
- `text`: string

Example:

```json
{
  "source_path": "/abs/path/lit/sample_paper.pdf",
  "file_name": "sample_paper.pdf",
  "title": "Friction Welding of Carbon Steel",
  "doi": "10.1000/example-doi",
  "chunk_id": 7,
  "chunk_start": 12600,
  "chunk_end": 14380,
  "char_length": 1780,
  "chunk_label": "experiment",
  "chunk_confidence": 0.92,
  "chunk_reason": "experiment dominant",
  "experiment_score": 11,
  "science_score": 3,
  "text": "Experimental procedure: friction welding was performed on..."
}
```

### 3. `outputs/paper_memory_cards.jsonl`

Producer:

- `cards_divider.py`

Format:

- JSONL
- one row per source paper
- a row may contain an `experiment_card`, a `science_card`, or both

Required top-level fields:

- `source_path`: string
- `source_title`: string
- `doi`: string

Optional card fields:

- `experiment_card`: object
- `science_card`: object

`experiment_card` fields:

- `process_families`: string[]
- `material_families`: string[]
- `material_systems`: string[]
- `equipment`: string[]
- `consumables`: string[]
- `controllable_parameters`: string[]
- `measurements_outputs`: string[]
- `literature_experiment_summary`: string

`science_card` fields:

- `dominant_mechanisms`: string[]
- `microstructure_terms`: string[]
- `property_relations`: string[]
- `causal_factors`: string[]
- `science_hypotheses`: string[]
- `literature_science_summary`: string

Example:

```json
{
  "source_path": "/abs/path/lit/sample_paper.pdf",
  "source_title": "Friction Welding of Carbon Steel",
  "doi": "10.1000/example-doi",
  "experiment_card": {
    "process_families": ["Friction Welding"],
    "material_families": ["steel"],
    "material_systems": ["carbon steel"],
    "equipment": ["friction welding machine", "thermocouple", "microhardness tester"],
    "consumables": [],
    "controllable_parameters": ["rotation speed", "friction pressure", "upset pressure"],
    "measurements_outputs": ["joint strength", "hardness", "macrostructure"],
    "literature_experiment_summary": "The paper varies friction pressure and upset pressure for steel joints."
  },
  "science_card": {
    "dominant_mechanisms": ["dynamic recrystallization"],
    "microstructure_terms": ["refined grains", "flash region"],
    "property_relations": ["higher upset pressure correlated with strength increase"],
    "causal_factors": ["heat generation", "plastic deformation"],
    "science_hypotheses": ["grain refinement may improve joint strength"],
    "literature_science_summary": "The paper links heat input and deformation to grain refinement near the weld interface."
  }
}
```

### 4. `outputs/rag1_latest_output.json`

Producer:

- `proposal_generator.py`

Format:

- JSON
- latest-run singleton payload

Required fields:

- `available_bom`: object
- `rag1_proposal`: string

Expected `available_bom` fields in current practice:

- `material_family`: string
- `process_family`: string
- `materials`: string[]
- `equipment`: string[]
- `forbidden_items`: string[]
- `goal`: string

Example:

```json
{
  "available_bom": {
    "material_family": "steel",
    "process_family": "Friction Welding",
    "materials": ["carbon steel", "stainless steel", "thermocouple"],
    "equipment": [
      "friction welding machine",
      "thermocouple",
      "microhardness tester",
      "tensile testing machine",
      "optical microscope",
      "SEM"
    ],
    "forbidden_items": ["EBSD", "synchrotron", "CFD simulation"],
    "goal": "Propose one feasible friction welding experiment on steel-based materials."
  },
  "rag1_proposal": "Candidate Experiment:\nCompare two friction pressures and two upset pressures for carbon steel joints...\n\nSource Papers Used:\n- Friction Welding of Carbon Steel"
}
```

### 5. `outputs/rag2_advice.json`

Producer:

- `scientific_advisor.py`

Format:

- JSON
- latest-run singleton payload

Required fields:

- `bom_check`: object
- `missing_or_limited_items`: object[]
- `narrowing_advice`: object[]
- `message_to_rag1`: string

`bom_check` fields:

- `status`: `feasible | mostly_feasible | limited | not_feasible | parse_failed`
- `reason`: string
- `literature_support`: object[]

`missing_or_limited_items` row fields:

- `item`: string
- `why_it_matters`: string
- `literature_support`: object[]

`narrowing_advice` row fields:

- `advice`: string
- `why`: string
- `literature_support`: object[]

`literature_support` row fields:

- `reference_id`: string
- `why_it_supports_this`: string

Example:

```json
{
  "bom_check": {
    "status": "mostly_feasible",
    "reason": "The listed machine and test equipment support a small first-pass welding study, but microstructural interpretation is limited without higher-resolution tools.",
    "literature_support": [
      {
        "reference_id": "SR1",
        "why_it_supports_this": "SR1 describes a comparable friction welding setup with hardness and tensile measurements."
      }
    ]
  },
  "missing_or_limited_items": [
    {
      "item": "precise temperature history measurement",
      "why_it_matters": "It limits how confidently heat-input differences can be interpreted.",
      "literature_support": [
        {
          "reference_id": "SR2",
          "why_it_supports_this": "SR2 relies on thermal measurements to explain microstructural change."
        }
      ]
    }
  ],
  "narrowing_advice": [
    {
      "advice": "Start with two process factors instead of a broad multi-factor sweep.",
      "why": "This makes the first experiment easier to execute and interpret with the current BOM.",
      "literature_support": [
        {
          "reference_id": "SR1",
          "why_it_supports_this": "SR1 demonstrates interpretable comparisons using a smaller parameter set."
        }
      ]
    }
  ],
  "message_to_rag1": "Keep the first study narrow: compare a small pressure matrix, document hardness and tensile response, and avoid claiming mechanism proof without stronger characterization."
}
```

## V0 Summary

The `v0` output contract intentionally preserves the current repo behavior:

- `embed.py` owns `lit_embedding_store/`
- `cards_divider.py` owns `outputs/chunks_labeled.jsonl`, `outputs/paper_memory_cards.jsonl`, and `outputs/paper_memory_storage/`
- `proposal_generator.py` owns `outputs/rag1_latest_output.json`
- `scientific_advisor.py` owns `outputs/rag2_advice.json`

The only alignment choice made here is documentation-level:

- the combined `outputs/paper_memory_storage/` is the canonical retrieval-store contract for `v0`
- split experiment/science store names remain future work, not the `v0` producer contract
