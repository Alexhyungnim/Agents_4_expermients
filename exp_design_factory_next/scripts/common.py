from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Any


def load_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return iter(())
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
