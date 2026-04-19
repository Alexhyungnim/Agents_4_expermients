"""Project orchestrator entrypoint.

Default behavior uses the multi-agent loop in `Scripts/orchestrator.py`.
The older single-pass retrieve->propose flow is still available via
`--mode legacy-simple` for prompt-based testing.
"""

from __future__ import annotations

import argparse
import json
import importlib.util
import sys
from pathlib import Path
from datetime import datetime

from agents.llm_client import LocalLLMClient
from agents.rag_cards import PaperCardRAG
from agents.proposer import ProposerAgent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=["scripts-loop", "legacy-simple"],
        default="scripts-loop",
        help="Execution mode. scripts-loop calls Scripts/orchestrator.py; legacy-simple runs the original single-pass flow.",
    )
    p.add_argument("--index_dir", help="Required in legacy-simple mode.")
    p.add_argument("--prompt_file", help="Required in legacy-simple mode.")
    p.add_argument("--bom_file", help="Required in legacy-simple mode.")
    p.add_argument("--user_goal", default="Propose one feasible experiment that matches the available BOM.")
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--output_file", default="outputs/orchestrator_output.json")
    p.add_argument(
        "--scripts_dir",
        default="Scripts",
        help="Directory that contains the reference multi-agent loop files.",
    )
    return p.parse_args()


def build_retrieval_query(bom: dict, user_goal: str) -> str:
    equipment = ", ".join(bom.get("available_equipment", [])[:10])
    consumables = ", ".join(bom.get("available_consumables", [])[:10])
    constraints = json.dumps(bom.get("goal_constraints", {}), sort_keys=True)
    return (
        f"{user_goal}\n"
        f"Available equipment: {equipment}\n"
        f"Available consumables: {consumables}\n"
        f"Constraints: {constraints}"
    )


def _run_legacy_simple(args: argparse.Namespace) -> None:
    if not args.index_dir or not args.prompt_file or not args.bom_file:
        raise SystemExit(
            "legacy-simple mode requires --index_dir, --prompt_file, and --bom_file"
        )

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.bom_file, "r", encoding="utf-8") as f:
        bom = json.load(f)

    llm_client = LocalLLMClient()
    rag_cards = PaperCardRAG(index_dir=args.index_dir, top_k=args.top_k)
    proposer = ProposerAgent(prompt_file=args.prompt_file, llm_client=llm_client)

    retrieval_query = build_retrieval_query(bom, args.user_goal)
    retrieved_cards = rag_cards.retrieve(retrieval_query, top_k=args.top_k)
    proposal_text = proposer.run(bom=bom, user_goal=args.user_goal, retrieved_cards=retrieved_cards)

    result = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "user_goal": args.user_goal,
        "retrieval_query": retrieval_query,
        "retrieved_cards": retrieved_cards,
        "proposal_text": proposal_text,
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    print(f"\nSaved orchestrator output to: {out_path}")


def _run_scripts_loop(args: argparse.Namespace) -> None:
    scripts_dir = Path(args.scripts_dir).resolve()
    script_path = scripts_dir / "orchestrator.py"

    if not script_path.exists():
        raise SystemExit(f"Scripts loop entrypoint not found: {script_path}")

    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    spec = importlib.util.spec_from_file_location("scripts_orchestrator", script_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Could not load scripts orchestrator module from: {script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "main"):
        raise SystemExit("Scripts orchestrator module does not define main()")

    print("Running Scripts multi-agent loop (RAG1 <-> RAG2)...")
    final_result = module.main()

    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False)

    print(f"Saved loop summary to: {out_path}")


def main() -> None:
    args = parse_args()
    if args.mode == "legacy-simple":
        _run_legacy_simple(args)
        return

    _run_scripts_loop(args)
from proposal_generator import run_rag1
from scientific_advisor import run_rag2


MAX_ROUNDS = 3

def should_stop(advice: dict, round_idx: int) -> bool:
    status = advice.get("bom_check", {}).get("status", "").strip()

    if advice.get("parse_failed", False) and round_idx >= 1:
        return True

    if status == "strongly_feasible" and round_idx >= 1:
        return True

    if status == "mostly_feasible" and round_idx >= 2:
        return True

    return False

def print_round_summary(round_idx: int, rag1_output: dict, rag2_advice: dict) -> None:
    proposal = rag1_output.get("rag1_proposal", "")
    status = rag2_advice.get("bom_check", {}).get("status", "unknown")
    reason = rag2_advice.get("bom_check", {}).get("reason", "")

    print("\n" + "=" * 100)
    print(f"ROUND {round_idx + 1} SUMMARY")
    print("=" * 100)
    print(f"RAG2 status: {status}")
    print(f"Reason: {reason}")

    msg = rag2_advice.get("message_to_rag1", "")
    if msg:
        print(f"\nMessage to RAG1:\n{msg}")

    narrowing = rag2_advice.get("narrowing_advice", [])
    if narrowing:
        print("\nTop narrowing advice:")
        for i, item in enumerate(narrowing[:3], 1):
            advice_text = item.get("advice", "")
            why_text = item.get("why", "")
            print(f"{i}. {advice_text}")
            if why_text:
                print(f"   why: {why_text}")

    if proposal:
        print("\nLatest proposal preview:")
        print(proposal[:1500])


def main():
    rag2_feedback = None
    rag1_output = None
    rag2_advice = None
    cached_exp_evidence = None
    cached_sci_evidence = None

    for round_idx in range(MAX_ROUNDS):
        print("\n" + "=" * 100)
        print(f"MULTI-AGENT ROUND {round_idx + 1}")
        print("=" * 100)

        # 1) RAG1 generates or revises the experiment proposal
        rag1_output = run_rag1(
            rag2_feedback=rag2_feedback,
            cached_exp_evidence=cached_exp_evidence,
            save_output=True,
        )

        if cached_exp_evidence is None:
            cached_exp_evidence = rag1_output.get("cached_exp_evidence")

        # 2) RAG2 critiques the current proposal
        rag2_result = run_rag2(
            rag1_output=rag1_output,
            cached_sci_evidence=cached_sci_evidence,
            save_output=True,
        )

        rag2_advice = rag2_result["rag2_advice"]

        if cached_sci_evidence is None:
            cached_sci_evidence = rag2_result["cached_sci_evidence"]

        # 3) Print round summary
        print_round_summary(round_idx, rag1_output, rag2_advice)

        # 4) Decide whether to stop
        if should_stop(rag2_advice, round_idx):
            print("\nStopping condition met.")
            break

        # 5) Pass advice directly to next RAG1 round
        rag2_feedback = rag2_advice

    print("\n" + "=" * 100)
    print("MULTI-AGENT LOOP FINISHED")
    print("=" * 100)

    return {
        "final_rag1_output": rag1_output,
        "final_rag2_advice": rag2_advice,
    }


if __name__ == "__main__":
    main()
