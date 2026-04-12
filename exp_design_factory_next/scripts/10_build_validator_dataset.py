from __future__ import annotations
import json
from pathlib import Path


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    cands = {x["candidate_id"]: x for x in load_jsonl(Path("data/processed/candidates/candidates.jsonl"))}
    judged = Path("data/processed/judged/judged_candidates.jsonl")
    out = Path("data/processed/datasets/validator/train.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for row in load_jsonl(judged):
            cand = cands[row["candidate_id"]]
            label = {k: v["score"] for k, v in row["rubric"].items()}
            label["overall_accept"] = row["overall_verdict"] == "accept_silver"
            out_row = {
                "prompt": "Score the proposal using the rubric fields.",
                "proposal": cand["final_proposal"],
                "label": label,
            }
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
    print(f"Saved validator dataset to {out}")


if __name__ == "__main__":
    main()
