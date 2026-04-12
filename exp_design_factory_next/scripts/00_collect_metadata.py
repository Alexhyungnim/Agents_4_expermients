from __future__ import annotations
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    out = Path("data/processed/papers/papers.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Placeholder metadata rows.
    rows = [
        {
            "paper_id": "demo_000001",
            "title": "Demo paper placeholder",
            "doi": "10.0000/demo",
            "journal": "Demo Journal",
            "year": 2025,
            "domain": "demo",
            "pdf_path": "data/raw/pdf/demo_000001.pdf",
            "parse_path": "data/raw/parse_cache/demo_000001.tei.xml",
            "source": "placeholder",
            "license_ok": True,
        }
    ]

    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved placeholder metadata to {out}")
    print(f"Config used: {args.config}")


if __name__ == "__main__":
    main()
