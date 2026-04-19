from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain_huggingface import HuggingFaceEmbeddings

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


WHITESPACE_RE = re.compile(r"\s+")
import os
import sys
print("hello from cluster")
print("python executable:", sys.executable)
print("ELSEVIER_APIKEY exists:", "ELSEVIER_APIKEY" in os.environ)

@dataclass
class ChunkRecord:
    source_path: str
    file_name: str
    title: str
    chunk_id: int
    chunk_start: int
    chunk_end: int
    char_length: int
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed every PDF under lit/ as chunked documents."
    )
    parser.add_argument(
        "--lit-dir",
        type=Path,
        default=Path("lit"),
        help="Directory that contains the literature PDFs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("lit_embedding_store"),
        help="Output directory for chunk metadata and persisted vector store.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1800,
        help="Chunk size in characters.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=400,
        help="Chunk overlap in characters.",
    )
    parser.add_argument(
        "--model-name",
        default="BAAI/bge-small-en-v1.5",
        help="Embedding model name for HuggingFaceEmbeddings.",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=200,
        help="Skip tiny chunks shorter than this many characters.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def extract_pdf_text(pdf_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: pypdf\n"
            "Install it with: pip install pypdf"
        ) from exc

    reader = PdfReader(str(pdf_path))
    pages: list[str] = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(page_text)

    return normalize_text("\n".join(pages))


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> Iterable[tuple[int, int, str]]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    step = chunk_size - chunk_overlap
    for start in range(0, len(text), step):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            yield start, end, chunk
        if end >= len(text):
            break


def build_embedding_model(model_name: str) -> LangchainEmbedding:
    device = "cpu"
    if torch is not None and torch.cuda.is_available():
        device = "cuda"

    return LangchainEmbedding(
        HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
    )


def title_from_path(pdf_path: Path) -> str:
    return pdf_path.stem.replace("_", " ")


def collect_documents(
    lit_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    min_chars: int,
) -> tuple[list[Document], list[ChunkRecord], list[dict[str, str]], int]:
    pdf_paths = sorted(path for path in lit_dir.rglob("*.pdf") if path.is_file())
    documents: list[Document] = []
    chunk_records: list[ChunkRecord] = []
    failures: list[dict[str, str]] = []

    for pdf_idx, pdf_path in enumerate(pdf_paths, start=1):
        print(f"[{pdf_idx}/{len(pdf_paths)}] Reading {pdf_path}")

        try:
            text = extract_pdf_text(pdf_path)
        except Exception as exc:  # pragma: no cover
            failures.append({"path": str(pdf_path), "error": str(exc)})
            print(f"  failed: {exc}")
            continue

        if not text:
            failures.append({"path": str(pdf_path), "error": "empty_text"})
            print("  skipped: empty text")
            continue

        title = title_from_path(pdf_path)
        chunk_count = 0

        for chunk_id, (start, end, chunk) in enumerate(
            chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ):
            if len(chunk) < min_chars:
                continue

            metadata = {
                "source_path": str(pdf_path),
                "file_name": pdf_path.name,
                "title": title,
                "chunk_id": chunk_id,
                "chunk_start": start,
                "chunk_end": end,
                "char_length": len(chunk),
            }
            documents.append(Document(text=chunk, metadata=metadata))
            chunk_records.append(ChunkRecord(text=chunk, **metadata))
            chunk_count += 1

        print(f"  chunks kept: {chunk_count}")

    return documents, chunk_records, failures, len(pdf_paths)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    lit_dir = args.lit_dir.resolve()
    out_dir = args.out_dir.resolve()

    if not lit_dir.exists():
        raise SystemExit(f"lit directory does not exist: {lit_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    documents, chunk_records, failures, pdf_count = collect_documents(
        lit_dir=lit_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chars=args.min_chars,
    )

    if not documents:
        raise SystemExit("No chunked documents were created.")

    chunks_path = out_dir / "chunks.jsonl"
    failures_path = out_dir / "failures.jsonl"
    write_jsonl(chunks_path, (asdict(record) for record in chunk_records))
    write_jsonl(failures_path, failures)

    Settings.embed_model = build_embedding_model(args.model_name)
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    storage_dir = out_dir / "storage"
    index.storage_context.persist(persist_dir=str(storage_dir))

    summary = {
        "lit_dir": str(lit_dir),
        "pdf_count": pdf_count,
        "chunk_count": len(chunk_records),
        "failure_count": len(failures),
        "embedding_model": args.model_name,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "min_chars": args.min_chars,
        "storage_dir": str(storage_dir),
        "chunks_path": str(chunks_path),
        "failures_path": str(failures_path),
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nEmbedding complete.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
