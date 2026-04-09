import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.core import Settings, load_index_from_storage, StorageContext
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_huggingface import HuggingFaceEmbeddings


# =========================================================
# Simple local LLM wrapper
# =========================================================
class LocalQwenLLM:
    def __init__(self, model_name="Qwen/Qwen3-8B"):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def complete(self, prompt, max_new_tokens=2500):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        return text


# =========================================================
# Embedding model
# =========================================================
LOCAL_EMBED_MODEL = "hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

Settings.embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(
        model_name=LOCAL_EMBED_MODEL,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
)


# =========================================================
# LLM
# =========================================================
llm = LocalQwenLLM("Qwen/Qwen3-8B")


# =========================================================
# Paths
# =========================================================
RAG1_OUTPUT_PATH = Path("outputs/rag1_latest_output.json")
SCIENCE_STORAGE_DIR = "outputs/paper_memory_storage_science"   # change if needed
CHUNKS_LABELED_PATH = Path("outputs/chunks_labeled.jsonl")
RAG2_OUTPUT_PATH = Path("outputs/rag2_advice.json")


# =========================================================
# Helpers
# =========================================================
def overlap_score_text(text, bom):
    text = (text or "").lower()
    score = 0

    for item in bom.get("materials", []):
        if item.lower() in text:
            score += 2

    for item in bom.get("equipment", []):
        if item.lower() in text:
            score += 2

    for item in bom.get("forbidden_items", []):
        if item.lower() in text:
            score -= 4

    process_family = bom.get("process_family", "")
    if process_family and process_family.lower() in text:
        score += 4

    material_family = bom.get("material_family", "")
    if material_family and material_family.lower() in text:
        score += 2

    return score


def load_labeled_rows(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def recover_supporting_chunks(shortlist, labeled_rows, bom, top_chunks_per_paper=2):
    selected_source_paths = {
        node.metadata.get("source_path", "") for node in shortlist
    }

    paper_to_chunks = {}

    for source_path in selected_source_paths:
        candidate_rows = [
            row for row in labeled_rows
            if row.get("source_path", "") == source_path
            and row.get("chunk_label") in {"science", "both"}
        ]

        ranked_rows = sorted(
            candidate_rows,
            key=lambda row: (
                overlap_score_text(row.get("text", ""), bom),
                row.get("chunk_confidence", 0.0),
                row.get("science_score", 0) + row.get("experiment_score", 0),
            ),
            reverse=True,
        )

        paper_to_chunks[source_path] = ranked_rows[:top_chunks_per_paper]

    return paper_to_chunks


def build_rag2_query(bom, proposal_text):
    return f"""
Process family: {bom.get("process_family", "")}
Material family: {bom.get("material_family", "")}
Goal: {bom.get("goal", "")}

RAG1 experiment proposal:
{proposal_text[:2200]}

Retrieve scientific literature that helps:
- verify whether this proposed experiment is feasible
- identify missing practical items or limited capability
- reduce the number of variables
- simplify the first experiment into a narrower, executable study
""".strip()


def extract_json_block(text: str):
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found.")

    brace_count = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if escape:
            escape = False
            continue

        if ch == "\\":
            escape = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if not in_string:
            if ch == "{":
                brace_count += 1
            elif ch == "}":
                brace_count -= 1

                if brace_count == 0:
                    return text[start:i + 1]

    raise ValueError("No complete JSON object found.")

# =========================================================
# Load RAG1 output
# =========================================================
if not RAG1_OUTPUT_PATH.exists():
    raise FileNotFoundError(f"RAG1 output not found: {RAG1_OUTPUT_PATH}")

with RAG1_OUTPUT_PATH.open("r", encoding="utf-8") as f:
    rag1_data = json.load(f)

available_bom = rag1_data["available_bom"]
rag1_proposal = rag1_data["rag1_proposal"]


# =========================================================
# Load science index
# =========================================================
storage_context = StorageContext.from_defaults(
    persist_dir=SCIENCE_STORAGE_DIR
)
science_index = load_index_from_storage(storage_context)


# =========================================================
# Retrieve science papers
# =========================================================
rag2_query = build_rag2_query(available_bom, rag1_proposal)

retriever = science_index.as_retriever(similarity_top_k=6)
retrieved_science = retriever.retrieve(rag2_query)

shortlist = retrieved_science[:4]

print("\nRetrieved science references for RAG2:")
for i, node in enumerate(shortlist, 1):
    print(f"\n[{i}] {node.metadata.get('source_title')}")
    print(f"Source Title: {node.metadata.get('source_title')}")
    print(f"DOI: {node.metadata.get('doi', '')}")
    print(f"Card Type: {node.metadata.get('card_type', '')}")
    print((node.text or "")[:900], "...\n")


# =========================================================
# Recover raw supporting chunks from chunks_labeled
# =========================================================
labeled_rows = load_labeled_rows(CHUNKS_LABELED_PATH)
paper_to_chunks = recover_supporting_chunks(
    shortlist=shortlist,
    labeled_rows=labeled_rows,
    bom=available_bom,
    top_chunks_per_paper=2,
)

print("Recovered supporting chunks:")
for node in shortlist:
    source_path = node.metadata.get("source_path", "")
    title = node.metadata.get("source_title", "")
    n = len(paper_to_chunks.get(source_path, []))
    print(f"{title}: {n} chunks")


# =========================================================
# Build science card context
# =========================================================
science_blocks = []
reference_map = {}

for i, node in enumerate(shortlist, 1):
    ref_id = f"SR{i}"
    title = node.metadata.get("source_title", f"Science Paper {i}")
    doi = node.metadata.get("doi", "")

    reference_map[ref_id] = {
        "title": title,
        "doi": doi,
    }

    science_blocks.append(
        f"""[Science Reference {ref_id}]
Title: {title}
DOI: {doi}

{node.text}
"""
    )

science_card_context = "\n\n" + ("\n\n" + "=" * 80 + "\n\n").join(science_blocks)


# =========================================================
# Build supporting chunk context
# =========================================================
chunk_blocks = []
chunk_idx = 1

source_path_to_ref_id = {
    node.metadata.get("source_path", ""): f"SR{i}"
    for i, node in enumerate(shortlist, 1)
}

for node in shortlist:
    source_path = node.metadata.get("source_path", "")
    title = node.metadata.get("source_title", "")
    ref_id = source_path_to_ref_id.get(source_path, "SRx")

    for row in paper_to_chunks.get(source_path, []):
        chunk_blocks.append(
            f"""[Supporting Chunk {chunk_idx}]
Reference ID: {ref_id}
Title: {title}
Chunk ID: {row.get('chunk_id')}
Text:
{row.get('text', '')}
"""
        )
        chunk_idx += 1

supporting_chunk_context = "\n\n" + ("\n\n" + "=" * 80 + "\n\n").join(chunk_blocks)


# =========================================================
# Prompt
# =========================================================
rag2_prompt = f"""
You are RAG2, a literature-grounded experiment feasibility advisor.

Your role:
- Review the RAG1 experiment proposal
- Use the available BOM
- Use the retrieved science literature
- Do NOT rewrite the whole experiment
- Do NOT directly replace RAG1
- Give practical advice back to RAG1

Focus only on:
1. Is the experiment feasible with the current BOM?
2. Are any important materials, equipment, or measurement capabilities missing or limited?
3. Is the proposal too broad for a first experiment?
4. How should the proposal be narrowed so it becomes easier to execute and interpret?

Keep the advice practical and simple.
Prioritize executable first-step experiments over ambitious broad studies.
Prefer reducing variables and measurements rather than expanding the design.
Focus on what can realistically be done in the current lab with the listed BOM.
Prefer identifying missing practical lab items over abstract evaluation metrics.

Do not give deep theoretical criticism unless it directly affects feasibility or clarity.
Do not invent unavailable equipment.
If something is optional rather than essential, say that clearly.

Use the Science Literature Cards for overall context.
Use the Supporting Evidence Chunks as the primary evidence for practical advice.
Prefer advice that is directly supported by the supporting chunks.

Do NOT rewrite, shorten, or invent paper titles.
Do NOT cite papers outside the retrieved references.
Use the actual retrieved Science Reference IDs such as SR1, SR2, SR3, etc.
The JSON schema below uses SRx only as a placeholder example.

Do not include any explanation before the JSON.
Your first character must be the opening brace of the JSON object.

Available BOM:
{json.dumps(available_bom, indent=2)}

RAG1 Proposal:
{rag1_proposal}

Science Literature Cards:
{science_card_context}

Supporting Evidence Chunks:
{supporting_chunk_context}

Return valid JSON only, in exactly this structure:

{{
  "bom_check": {{
    "status": "feasible / mostly_feasible / limited / not_feasible",
    "reason": "short explanation",
    "literature_support": [
      {{
        "reference_id": "SRx",
        "why_it_supports_this": "short explanation"
      }}
    ]
  }},
  "missing_or_limited_items": [
    {{
      "item": "item 1",
      "why_it_matters": "short explanation",
      "literature_support": [
        {{
          "reference_id": "SRx",
          "why_it_supports_this": "short explanation"
        }}
      ]
    }}
  ],
  "narrowing_advice": [
    {{
      "advice": "advice 1",
      "why": "short explanation",
      "literature_support": [
        {{
          "reference_id": "SRx",
          "why_it_supports_this": "short explanation"
        }}
      ]
    }}
  ],
  "message_to_rag1": "Short advisory message to RAG1"
}}
""".strip()


# =========================================================
# Generate advice
# =========================================================
raw_output = llm.complete(rag2_prompt, max_new_tokens=1200)

try:
    json_text = extract_json_block(raw_output)
    rag2_advice = json.loads(json_text)
except Exception as e:
    rag2_advice = {
        "bom_check": {
            "status": "parse_failed",
            "reason": f"Model output could not be parsed as JSON: {str(e)}",
            "literature_support": []
        },
        "missing_or_limited_items": [],
        "narrowing_advice": [],
        "message_to_rag1": raw_output.strip()
    }


# =========================================================
# Save advice only
# =========================================================
RAG2_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with RAG2_OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(rag2_advice, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 100)
print("RAG2 ADVICE")
print("=" * 100)
print(json.dumps(rag2_advice, indent=2, ensure_ascii=False))
print(f"\nSaved RAG2 output to: {RAG2_OUTPUT_PATH}")


# =========================================================
# Optional: pretty print reference map for debugging
# =========================================================
print("\nReference Map:")
for ref_id, meta in reference_map.items():
    print(f"{ref_id}: {meta['title']} | DOI: {meta['doi']}")
