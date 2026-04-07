from __future__ import annotations
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    pdf_dir = Path("data/raw/pdf")
    pdf_dir.mkdir(parents=True, exist_ok=True)
    print("PDF download step is scaffolded. Plug in your provider/API here.")
    print(f"Config used: {args.config}")


if __name__ == "__main__":
    main()
