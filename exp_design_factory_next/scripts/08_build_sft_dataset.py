from __future__ import annotations
import json
from pathlib import Path


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    tasks = {x["task_id"]: x for x in load_jsonl(Path("data/processed/tasks/tasks.jsonl"))}
    cands = {x["candidate_id"]: x for x in load_jsonl(Path("data/processed/candidates/candidates.jsonl"))}
    judged_path = Path("data/processed/judged/judged_candidates.jsonl")
    out = Path("data/processed/datasets/sft/train.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for judged in load_jsonl(judged_path):
            if judged["overall_verdict"] != "accept_silver":
                continue
            cand = cands[judged["candidate_id"]]
            task = tasks[cand["task_id"]]
            prompt = (
                "You are an engineering experiment design assistant. Return JSON only.\n\n"
                f"Task:\n{json.dumps(task, ensure_ascii=False, indent=2)}\n"
            )
            answer = json.dumps(cand["final_proposal"], ensure_ascii=False)
            row = {"text": f"<|user|>\n{prompt}\n<|assistant|>\n{answer}"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved SFT dataset to {out}")


if __name__ == "__main__":
    main()
