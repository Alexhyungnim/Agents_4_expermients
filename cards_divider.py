from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_huggingface import HuggingFaceEmbeddings

# =========================================================
# Local Qwen LLM (no Ollama)
# =========================================================
class LocalQwenLLM:
    def __init__(self, model_name: str = "Qwen/Qwen3-8B"):
        self.model_name = model_name
        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        print(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def complete(self, prompt: str, max_new_tokens: int = 512) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True)


# =========================================================
# Keyword priors
# =========================================================
EXPERIMENT_KEYWORDS = [
    "experimental procedure",
    "experimental setup",
    "materials and methods",
    "welding procedure",
    "process parameters",
    "sample preparation",
    "specimen preparation",
    "heat input",
    "cooling rate",
    "current",
    "voltage",
    "travel speed",
    "shielding gas",
    "interpass temperature",
    "preheat",
    "post weld heat treatment",
    "filler metal",
    "base metal",
    "equipment",
    "welding machine",
    "wire feeder",
    "torch",
    "thermocouple",
    "optical microscope",
    "SEM",
    "EDS",
    "microhardness",
    "tensile test",
    "impact test",
    "charpy",
]

SCIENCE_KEYWORDS = [
    "mechanism",
    "microstructure evolution",
    "phase transformation",
    "transformation behavior",
    "grain boundary ferrite",
    "acicular ferrite",
    "bainite",
    "martensite",
    "M/A constituent",
    "segregation",
    "diffusion",
    "nucleation",
    "growth",
    "thermodynamics",
    "kinetics",
    "strengthening mechanism",
    "fracture mechanism",
    "property relationship",
    "microstructure-property relationship",
    "causal factor",

    # softer / more realistic science phrasing
    "attributed to",
    "due to the",
    "because of",
    "resulted in",
    "led to",
    "promoted",
    "suppressed",
    "facilitated",
    "associated with",
    "correlated with",
    "can be explained by",
    "is explained by",
    "is related to",
    "contributed to",
    "formation of",
    "refinement of",
    "coarsening of",
    "microstructural change",
    "microstructural evolution",
    "toughness improvement",
    "hardness increase",
    "fracture surface",
]

SCIENCE_BONUS_TERMS = [
    "mechanism",
    "microstructure",
    "phase transformation",
    "fracture surface",
    "microstructure-property relationship",
    "strengthening mechanism",
    "fracture mechanism",
]

EXPERIMENT_CARD_TEMPLATE = {
    "process_families": [],
    "material_families": [],
    "material_systems": [],
    "equipment": [],
    "consumables": [],
    "controllable_parameters": [],
    "measurements_outputs": [],
    "joint_or_sample_types": [],
    "weldability_quality_factors": [],
    "common_constraints": [],
    "literature_experiment_summary": "",
}

SCIENCE_CARD_TEMPLATE = {
    "dominant_mechanisms": [],
    "microstructure_terms": [],
    "property_relations": [],
    "causal_factors": [],
    "science_hypotheses": [],
    "literature_science_summary": "",
}


# =========================================================
# Args
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paper-level experiment/science cards from existing chunks.jsonl"
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=Path("outputs/chunks.jsonl"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs"),
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-8B",
    )
    parser.add_argument(
        "--max-chunks-per-card",
        type=int,
        default=8,
    )
    return parser.parse_args()


# =========================================================
# IO utils
# =========================================================
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_documents_from_chunks(chunks_path: Path) -> list[Document]:
    rows = read_jsonl(chunks_path)
    documents: list[Document] = []

    for row in rows:
        text = row.get("text", "")
        metadata = {
            "source_path": row.get("source_path", ""),
            "file_name": row.get("file_name", ""),
            "title": row.get("title", ""),
            "doi": row.get("doi", ""),
            "chunk_id": row.get("chunk_id", -1),
            "chunk_start": row.get("chunk_start", -1),
            "chunk_end": row.get("chunk_end", -1),
            "char_length": row.get("char_length", len(text)),
        }
        documents.append(Document(text=text, metadata=metadata))

    return documents


# =========================================================
# Prompt / JSON helpers
# =========================================================
def extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    text = text.strip()

    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output.")

    obj, end = decoder.raw_decode(text[start:])
    if not isinstance(obj, dict):
        raise ValueError("Parsed JSON is not an object.")
    return obj


def normalize_list_field(v: Any) -> list[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v.strip()] if v.strip() else []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return [str(v).strip()]


def retry_json_prompt(llm: LocalQwenLLM, prompt: str, max_attempts: int = 2, max_new_tokens: int = 512) -> dict[str, Any]:
    last_error = None

    for attempt in range(max_attempts):
        final_prompt = prompt
        if attempt > 0:
            final_prompt += (
                "\n\nReturn ONLY one valid JSON object. "
                "No markdown. No explanation. No trailing text."
            )

        try:
            raw_text = llm.complete(final_prompt, max_new_tokens=max_new_tokens)
            return extract_json_object(raw_text)
        except Exception as exc:
            last_error = exc

    raise ValueError(f"Failed to parse JSON: {last_error}")


# =========================================================
# Labeling
# =========================================================
def keyword_score(text: str, keywords: list[str]) -> int:
    text_l = (text or "").lower()
    score = 0

    for kw in keywords:
        kw_l = kw.lower()
        if kw_l in text_l:
            score += 2
        else:
            parts = [p for p in re.split(r"[^a-z0-9]+", kw_l) if len(p) > 2]
            hits = sum(1 for p in parts if p in text_l)
            if hits >= max(1, len(parts) // 2):
                score += 1
    return score


def science_bonus_score(text: str) -> int:
    text_l = (text or "").lower()
    bonus = 0
    for term in SCIENCE_BONUS_TERMS:
        if term in text_l:
            bonus += 1
    return bonus


def classify_chunk_type(llm: LocalQwenLLM, text: str, exp_score: int, sci_score: int) -> dict[str, Any]:
    prompt = f"""
You are classifying one chunk from welding literature.

Classify the chunk into exactly one of:
- experiment
- science
- both
- neither

Definitions:
- experiment = setup, procedure, materials, equipment, consumables, process parameters, testing protocol, measurement protocol
- science = mechanism, interpretation, theory, causal explanation, microstructure-property reasoning
- both = substantial content from both
- neither = references, acknowledgements, publisher text, generic intro, weakly relevant text

Weak prior:
experiment_score = {exp_score}
science_score = {sci_score}

Chunk:
\"\"\"
{text}
\"\"\"

Return ONLY valid JSON:
{{
  "label": "experiment",
  "confidence": 0.0,
  "reason": ""
}}
"""
    data = retry_json_prompt(llm, prompt, max_attempts=2, max_new_tokens=160)

    label = str(data.get("label", "neither")).strip().lower()
    if label not in {"experiment", "science", "both", "neither"}:
        label = "neither"

    try:
        confidence = float(data.get("confidence", 0.0))
    except Exception:
        confidence = 0.0

    return {
        "label": label,
        "confidence": max(0.0, min(1.0, confidence)),
        "reason": str(data.get("reason", "")).strip(),
    }


def annotate_chunks(documents: list[Document], llm: LocalQwenLLM) -> None:
    for i, d in enumerate(documents, start=1):
        text = d.text or ""
        exp_score = keyword_score(text, EXPERIMENT_KEYWORDS)
        sci_score = keyword_score(text, SCIENCE_KEYWORDS)
        sci_score += science_bonus_score(text)

        if exp_score >= 9 and exp_score - sci_score >= 4:
            cls = {"label": "experiment", "confidence": 0.92, "reason": "experiment dominant"}
        elif sci_score >= 7 and sci_score - exp_score >= 3:
            cls = {"label": "science", "confidence": 0.92, "reason": "science dominant"}
        elif exp_score >= 7 and sci_score >= 5 and abs(exp_score - sci_score) <= 2:
            cls = {"label": "both", "confidence": 0.88, "reason": "balanced evidence for both"}
        elif exp_score <= 1 and sci_score <= 1:
            cls = {"label": "neither", "confidence": 0.75, "reason": "no strong evidence"}
        else:
            try:
                cls = classify_chunk_type(llm, text, exp_score, sci_score)
            except Exception as e:
                cls = {
                    "label": "neither",
                    "confidence": 0.0,
                    "reason": f"classification_failed: {str(e)}",
                }

        d.metadata["chunk_label"] = cls["label"]
        d.metadata["chunk_confidence"] = cls["confidence"]
        d.metadata["chunk_reason"] = cls["reason"]
        d.metadata["experiment_score"] = exp_score
        d.metadata["science_score"] = sci_score

        print(
            f"[Chunk {i}/{len(documents)}] "
            f"label={cls['label']} conf={cls['confidence']:.2f} "
            f"exp_score={exp_score} sci_score={sci_score}"
        )


# =========================================================
# Card generation
# =========================================================
def build_evidence_text(chunks: list[Document], max_chunks: int = 8) -> str:
    ranked = sorted(
        chunks,
        key=lambda d: (
            d.metadata.get("chunk_confidence", 0),
            d.metadata.get("experiment_score", 0) + d.metadata.get("science_score", 0),
            d.metadata.get("char_length", 0),
        ),
        reverse=True,
    )

    selected = ranked[:max_chunks]
    blocks = []

    for i, d in enumerate(selected, 1):
        blocks.append(
            f"""[Evidence Chunk {i}]
Title: {d.metadata.get("title", "")}
DOI: {d.metadata.get("doi", "")}
Chunk Label: {d.metadata.get("chunk_label")}
Chunk Confidence: {d.metadata.get("chunk_confidence")}
Chunk Start: {d.metadata.get("chunk_start")}
Chunk ID: {d.metadata.get("chunk_id")}

Text:
{d.text}
"""
        )

    return "\n\n" + ("\n\n" + "=" * 80 + "\n\n").join(blocks)

def card_has_real_content(card: dict[str, Any]) -> bool:
    for v in card.values():
        if isinstance(v, list) and len(v) > 0:
            return True
        if isinstance(v, str) and v.strip():
            return True
    return False


def build_experiment_card(llm: LocalQwenLLM, chunks: list[Document], max_chunks: int = 8) -> dict[str, Any]:
    evidence_text = build_evidence_text(chunks, max_chunks=max_chunks)

    prompt = f"""
You are creating ONE experiment card for ONE welding paper.

Use ONLY the evidence below.
Do not invent facts.

Evidence:
{evidence_text}

Return ONLY valid JSON:
{{
  "process_families": [],
  "material_families": [],
  "material_systems": [],
  "equipment": [],
  "consumables": [],
  "controllable_parameters": [],
  "measurements_outputs": [],
  "literature_experiment_summary": ""
}}
"""
    try:
        card = retry_json_prompt(llm, prompt, max_attempts=2, max_new_tokens=320)
    except Exception:
        card = {}

    out = EXPERIMENT_CARD_TEMPLATE.copy()
    for k in out:
        if k in card:
            out[k] = card[k]

    for k in out:
        if isinstance(EXPERIMENT_CARD_TEMPLATE[k], list):
            out[k] = normalize_list_field(out[k])
        else:
            out[k] = str(out[k]).strip()

    return out


def build_science_card(llm: LocalQwenLLM, chunks: list[Document], max_chunks: int = 8) -> dict[str, Any]:
    evidence_text = build_evidence_text(chunks, max_chunks=max_chunks)

    prompt = f"""
You are creating ONE science card for ONE welding paper.

Use ONLY the evidence below.
Do not invent facts.

Evidence:
{evidence_text}

Return ONLY valid JSON:
{{
  "dominant_mechanisms": [],
  "microstructure_terms": [],
  "property_relations": [],
  "causal_factors": [],
  "science_hypotheses": [],
  "literature_science_summary": ""
}}
"""
    try:
        card = retry_json_prompt(llm, prompt, max_attempts=2, max_new_tokens=320)
    except Exception:
        card = {}

    out = SCIENCE_CARD_TEMPLATE.copy()
    for k in out:
        if k in card:
            out[k] = card[k]

    for k in out:
        if isinstance(SCIENCE_CARD_TEMPLATE[k], list):
            out[k] = normalize_list_field(out[k])
        else:
            out[k] = str(out[k]).strip()

    return out


# =========================================================
# Retrieval docs
# =========================================================
def experiment_card_to_doc(title: str, doi: str, card: dict[str, Any]) -> Document:
    text = f"""
Source Title: {title}
DOI: {doi}
Card Type: experiment

Process Families: {", ".join(card["process_families"])}
Material Families: {", ".join(card["material_families"])}
Material Systems: {", ".join(card["material_systems"])}
Equipment: {", ".join(card["equipment"])}
Consumables: {", ".join(card["consumables"])}
Controllable Parameters: {", ".join(card["controllable_parameters"])}
Measurements / Outputs: {", ".join(card["measurements_outputs"])}
Joint / Sample Types: {", ".join(card["joint_or_sample_types"])}
Weldability / Quality Factors: {", ".join(card["weldability_quality_factors"])}
Common Constraints: {", ".join(card["common_constraints"])}

Literature Experiment Summary:
{card["literature_experiment_summary"]}
""".strip()

    return Document(
        text=text,
        metadata={
            "card_type": "experiment",
            "source_title": title,
            "doi": doi,
        },
    )


def science_card_to_doc(title: str, doi: str, card: dict[str, Any]) -> Document:
    text = f"""
Source Title: {title}
DOI: {doi}
Card Type: science

Dominant Mechanisms: {", ".join(card["dominant_mechanisms"])}
Microstructure Terms: {", ".join(card["microstructure_terms"])}
Property Relations: {", ".join(card["property_relations"])}
Causal Factors: {", ".join(card["causal_factors"])}
Science Hypotheses: {", ".join(card["science_hypotheses"])}

Literature Science Summary:
{card["literature_science_summary"]}
""".strip()

    return Document(
        text=text,
        metadata={
            "card_type": "science",
            "source_title": title,
            "doi": doi,
        },
    )


def build_paper_memory_cards(documents: list[Document], llm: LocalQwenLLM, out_dir: Path, max_chunks: int = 8) -> list[dict[str, Any]]:
    paper_groups: dict[tuple[str, str, str], list[Document]] = defaultdict(list)

    for d in documents:
        key = (
            d.metadata.get("source_path", ""),
            d.metadata.get("title", ""),
            d.metadata.get("doi", ""),
        )
        paper_groups[key].append(d)

    outputs = []
    card_docs = []

    for idx, ((source_path, title, doi), chunk_docs) in enumerate(paper_groups.items(), start=1):
        print(f"[Paper {idx}/{len(paper_groups)}] {title}")

        exp_chunks = []
        sci_chunks = []

        for d in chunk_docs:
            label = d.metadata.get("chunk_label", "neither")
            if label in {"experiment", "both"}:
                exp_chunks.append(d)
            if label in {"science", "both"}:
                sci_chunks.append(d)

        row = {
            "source_path": source_path,
            "source_title": title,
            "doi": doi,
        }

        if exp_chunks:
            exp_card = build_experiment_card(llm, exp_chunks, max_chunks=max_chunks)
            if card_has_real_content(exp_card):
                row["experiment_card"] = exp_card
                card_docs.append(experiment_card_to_doc(title, doi, exp_card))

        if sci_chunks:
            sci_card = build_science_card(llm, sci_chunks, max_chunks=max_chunks)
            if card_has_real_content(sci_card):
                row["science_card"] = sci_card
                card_docs.append(science_card_to_doc(title, doi, sci_card))

        if "experiment_card" in row or "science_card" in row:
            outputs.append(row)

    cards_path = out_dir / "paper_memory_cards.jsonl"
    write_jsonl(cards_path, outputs)

    if card_docs:
        index = VectorStoreIndex.from_documents(card_docs, show_progress=True)
        index.storage_context.persist(persist_dir=str(out_dir / "paper_memory_storage"))

    print(f"Saved {len(outputs)} paper memory cards to {cards_path}")
    print(f"Persisted vector store to {out_dir / 'paper_memory_storage'}")
    return outputs


# =========================================================
# Main
# =========================================================
def main() -> None:
    args = parse_args()
    chunks_path = args.chunks_path.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not chunks_path.exists():
        raise SystemExit(f"chunks.jsonl does not exist: {chunks_path}")

    documents = load_documents_from_chunks(chunks_path)
    if not documents:
        raise SystemExit("No documents loaded from chunks.jsonl")

    llm = LocalQwenLLM(model_name=args.model_name)
    Settings.embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
)

    annotate_chunks(documents, llm)

    labeled_rows = []
    for d in documents:
        labeled_rows.append({
            "source_path": d.metadata.get("source_path", ""),
            "file_name": d.metadata.get("file_name", ""),
            "title": d.metadata.get("title", ""),
            "doi": d.metadata.get("doi", ""),
            "chunk_id": d.metadata.get("chunk_id", -1),
            "chunk_start": d.metadata.get("chunk_start", -1),
            "chunk_end": d.metadata.get("chunk_end", -1),
            "char_length": d.metadata.get("char_length", 0),
            "chunk_label": d.metadata.get("chunk_label", ""),
            "chunk_confidence": d.metadata.get("chunk_confidence", 0.0),
            "chunk_reason": d.metadata.get("chunk_reason", ""),
            "experiment_score": d.metadata.get("experiment_score", 0),
            "science_score": d.metadata.get("science_score", 0),
            "text": d.text,
        })

    write_jsonl(out_dir / "chunks_labeled.jsonl", labeled_rows)
    build_paper_memory_cards(documents, llm, out_dir, max_chunks=args.max_chunks_per_card)


if __name__ == "__main__":
    main()
