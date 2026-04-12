from __future__ import annotations
import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    out_dir = Path("data/raw/parse_cache")
    out_dir.mkdir(parents=True, exist_ok=True)
    print("GROBID parsing step is scaffolded. Connect your local/remote GROBID service here.")
    print(f"Config used: {args.config}")


if __name__ == "__main__":
    main()
