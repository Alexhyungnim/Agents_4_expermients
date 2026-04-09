import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.core import Settings, load_index_from_storage, StorageContext
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter, FilterCondition
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

    def complete(self, prompt, max_new_tokens=550):
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

        # lightweight cleanup
        if "<think>" in text.lower():
            idx = text.lower().find("candidate experiment:")
            if idx != -1:
                text = text[idx:].strip()

        return text


# =========================================================
# Embedding model
# =========================================================
LOCAL_EMBED_MODEL = "huggingface/hub/models--BAAI--bge-small-en-v1.5/snapshots/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

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
# BOM
# =========================================================
available_bom = {
    "material_family": "steel",
    "process_family": "Friction Welding",
    "materials": [
        "carbon steel",
        "stainless steel",
        "thermocouple",
    ],
    "equipment": [
        "friction welding machine",
        "thermocouple",
        "microhardness tester",
        "tensile testing machine",
        "optical microscope",
        "SEM",
    ],
    "forbidden_items": [
        "EBSD",
        "synchrotron",
        "CFD simulation",
    ],
    "goal": "Propose one feasible friction welding experiment on steel-based materials."
}


# =========================================================
# Paths
# =========================================================
EXPERIMENT_STORAGE_DIR = "outputs/paper_memory_storage_experiment"
CHUNKS_LABELED_PATH = Path("outputs/chunks_labeled.jsonl")


# =========================================================
# Load index
# =========================================================
storage_context = StorageContext.from_defaults(
    persist_dir=EXPERIMENT_STORAGE_DIR
)
paper_card_index = load_index_from_storage(storage_context)


# =========================================================
# Helpers
# =========================================================
def overlap_score_text(text, bom):
    text = (text or "").lower()
    score = 0

    for item in bom["materials"]:
        if item.lower() in text:
            score += 2

    for item in bom["equipment"]:
        if item.lower() in text:
            score += 2

    for item in bom["forbidden_items"]:
        if item.lower() in text:
            score -= 4

    return score


def overlap_score_node(node, bom):
    return overlap_score_text(node.text or "", bom)


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
            and row.get("chunk_label") in {"experiment", "both"}
        ]

        ranked_rows = sorted(
            candidate_rows,
            key=lambda row: (
                overlap_score_text(row.get("text", ""), bom),
                row.get("chunk_confidence", 0.0),
                row.get("experiment_score", 0) + row.get("science_score", 0),
            ),
            reverse=True,
        )

        paper_to_chunks[source_path] = ranked_rows[:top_chunks_per_paper]

    return paper_to_chunks


def trim_repeated_sections(text: str) -> str:
    marker = "Candidate Experiment:"
    first = text.find(marker)
    if first == -1:
        return text.strip()

    second = text.find(marker, first + len(marker))
    if second != -1:
        return text[:second].strip()

    return text.strip()


# =========================================================
# Retrieve cards
# =========================================================
# filters = MetadataFilters(
#     filters=[
#         MetadataFilter(
#             key="process_families",
#             value=available_bom["process_family"]
#         ),
#     ],
#     condition=FilterCondition.AND,
# )

retriever = paper_card_index.as_retriever(
    similarity_top_k=8,
    # filters=filters,
)
retrieved_cards = retriever.retrieve(available_bom["goal"])

# Extra Python-side filter, since retriever filter can still be noisy
filtered_cards = []
for node in retrieved_cards:
    process_vals = " ".join(node.metadata.get("process_families", []))
    if available_bom["process_family"].lower() in process_vals.lower():
        filtered_cards.append(node)

ranked = sorted(
    filtered_cards,
    key=lambda n: overlap_score_node(n, available_bom),
    reverse=True,
)

shortlist = ranked[:4]


print("Shortlisted paper cards:")
for i, node in enumerate(shortlist, 1):
    print(f"\n[{i}] {node.metadata.get('source_title')}")
    print(f"process_families={node.metadata.get('process_families')}")
    print(f"material_families={node.metadata.get('material_families')}")
    print((node.text or "")[:1000], "...\n")


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
# Build card context
# =========================================================
card_blocks = []
for i, node in enumerate(shortlist, 1):
    card_blocks.append(
        f"""[Paper Card {i}]
Title: {node.metadata.get('source_title')}
DOI: {node.metadata.get('doi')}

{node.text}
"""
    )

card_context = "\n\n" + ("\n\n" + "=" * 80 + "\n\n").join(card_blocks)


# =========================================================
# Build chunk context
# =========================================================
chunk_blocks = []
idx = 1

for node in shortlist:
    source_path = node.metadata.get("source_path", "")
    title = node.metadata.get("source_title", "")

    print("CARD PATH:", repr(source_path))
    for row in paper_to_chunks.get(source_path, []):
        print("MATCHED CHUNK PATH:", repr(row.get("source_path")))
        chunk_blocks.append(
            f"""[Supporting Chunk {idx}]
Title: {title}
Chunk ID: {row.get('chunk_id')}
Text:
{row.get('text', '')}
"""
        )
        idx += 1

chunk_context = "\n\n" + ("\n\n" + "=" * 80 + "\n\n").join(chunk_blocks)


source_titles = [
    node.metadata.get("source_title")
    for node in shortlist
    if node.metadata.get("source_title")
]


# =========================================================
# Prompt
# =========================================================
proposal_prompt = f"""
You are proposing ONE candidate new experiment.

Use the literature cards as precedent summaries.
Use the supporting chunks as grounded evidence.
Do NOT pretend the experiment is already published.
You may synthesize across multiple papers.
You must respect the constrained BOM.
Answer in English only.
Do not output reasoning.
Do not output <think>.

Available BOM / Lab Capability:
{json.dumps(available_bom, indent=2)}

Literature Cards:
{card_context}

Supporting Evidence Chunks:
{chunk_context}

Task:
Propose ONE feasible experiment that is inspired by the retrieved papers and is compatible with the BOM.

Return in this exact format:

Candidate Experiment:
Why This Is Feasible With Current BOM:
Borrowed Literature Precedents:
New Adaptation / Novel Twist:
Needed Equipment:
Needed Materials / Consumables:
Key Process Parameters To Sweep:
Measurements / Outputs:
Main Risk / Failure Mode:
Missing Capability / Assumption:

Rules:
- Do not claim the experiment is novel with certainty.
- Distinguish clearly between literature-backed elements and your proposed adaptation.
- Do not include equipment or materials not supported by the BOM unless you put them under "Missing Capability / Assumption".
- If the BOM is insufficient, say so explicitly.
""".strip()


# =========================================================
# Generate proposal
# =========================================================
proposal = llm.complete(proposal_prompt, max_new_tokens=550)

if "Candidate Experiment:" in proposal:
    proposal = proposal[proposal.find("Candidate Experiment:"):].strip()

proposal = trim_repeated_sections(proposal)

proposal += "\n\nSource Papers Used:\n" + "\n".join(f"- {t}" for t in source_titles)

print("\n" + "=" * 100)
print("RAG1 PROPOSAL")
print("=" * 100)
print(proposal)

# =========================================================
# Save BOM + proposal for RAG2
# =========================================================
RAG1_OUTPUT_PATH = Path("outputs/rag1_latest_output.json")

rag1_output_payload = {
    "available_bom": available_bom,
    "rag1_proposal": proposal,
}

RAG1_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with RAG1_OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(rag1_output_payload, f, indent=2, ensure_ascii=False)

print(f"\nSaved RAG1 output to: {RAG1_OUTPUT_PATH}")
