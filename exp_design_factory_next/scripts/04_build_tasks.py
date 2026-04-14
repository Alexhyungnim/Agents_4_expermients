from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def build_demo_task_templates() -> list[dict[str, object]]:
    templates = [
        {
            "task_id": "task_demo_002",
            "paper_id": "paper_demo_002",
            "goal": "Design one feasible experiment to study how laser power affects porosity in a DED NiTi process.",
            "constraint": "Use explicit numeric laser power levels.",
            "equipment": [
                "DED machine",
                "SEM",
                "precision saw",
                "mounting press",
                "grinding and polishing kit",
            ],
            "materials": [
                "NiTi powder",
                "argon gas",
                "mounting resin",
            ],
            "evidence_chunk_id": "paper_002_chunk_01",
            "evidence_text": (
                "The DED NiTi process uses argon shielding and a validated stable operating window for laser power. "
                "Samples can be sectioned, mounted, ground, and polished for SEM cross-section imaging to quantify porosity."
            ),
        },
        {
            "task_id": "task_demo_003",
            "paper_id": "paper_demo_003",
            "goal": "Design one feasible experiment to study how scan speed affects track width and build height in a DED NiTi process.",
            "constraint": "Use explicit numeric scan speed levels.",
            "equipment": [
                "DED machine",
                "optical microscope",
                "digital calipers",
            ],
            "materials": [
                "NiTi powder",
                "argon gas",
            ],
            "evidence_chunk_id": "paper_003_chunk_01",
            "evidence_text": (
                "The DED NiTi process has a validated scan-speed window for stable deposition. "
                "Top-view optical microscopy and caliper checks can be used to compare track width and build height across scan-speed settings."
            ),
        },
        {
            "task_id": "task_demo_004",
            "paper_id": "paper_demo_004",
            "goal": "Design one feasible experiment to study how solution treatment temperature affects retained austenite fraction in DED NiTi coupons.",
            "constraint": "Use explicit numeric solution treatment temperatures.",
            "equipment": [
                "DED machine",
                "vacuum furnace",
                "XRD",
            ],
            "materials": [
                "NiTi powder",
                "argon gas",
            ],
            "evidence_chunk_id": "paper_004_chunk_01",
            "evidence_text": (
                "DED NiTi coupons can be heat treated in a vacuum furnace at controlled temperatures, and XRD is the evidence-supported method for comparing retained austenite fraction afterward."
            ),
        },
        {
            "task_id": "task_demo_005",
            "paper_id": "paper_demo_005",
            "goal": "Design one feasible experiment to study how hatch spacing affects microhardness in a DED NiTi process.",
            "constraint": "Use explicit numeric hatch spacing levels.",
            "equipment": [
                "DED machine",
                "microhardness tester",
                "precision saw",
                "mounting press",
                "grinding and polishing kit",
            ],
            "materials": [
                "NiTi powder",
                "argon gas",
                "mounting resin",
            ],
            "evidence_chunk_id": "paper_005_chunk_01",
            "evidence_text": (
                "Hatch spacing can be varied within a stable DED NiTi window, and polished cross-sections can be used for microhardness mapping after preparation with the listed metallography tools."
            ),
        },
        {
            "task_id": "task_demo_006",
            "paper_id": "paper_demo_006",
            "goal": "Design one feasible experiment to study how powder feed rate affects relative density in a DED NiTi process.",
            "constraint": "Use explicit numeric powder feed rate levels.",
            "equipment": [
                "DED machine",
                "analytical balance",
                "Archimedes density kit",
            ],
            "materials": [
                "NiTi powder",
                "argon gas",
                "ethanol",
            ],
            "evidence_chunk_id": "paper_006_chunk_01",
            "evidence_text": (
                "The DED NiTi setup has a repeatable powder-feed range, and an Archimedes density kit with an analytical balance is available for relative density measurements."
            ),
        },
        {
            "task_id": "task_demo_007",
            "paper_id": "paper_demo_007",
            "goal": "Design one feasible experiment to study how laser power affects surface roughness in a DED NiTi process.",
            "constraint": "Use explicit numeric laser power levels.",
            "equipment": [
                "DED machine",
                "optical profilometer",
            ],
            "materials": [
                "NiTi powder",
                "argon gas",
            ],
            "evidence_chunk_id": "paper_007_chunk_01",
            "evidence_text": (
                "Laser power can be swept inside a stable DED NiTi window, and an optical profilometer is available to compare surface roughness after deposition."
            ),
        },
    ]

    tasks: list[dict[str, object]] = []
    for template in templates:
        tasks.append(
            {
                "task_id": template["task_id"],
                "paper_id": template["paper_id"],
                "domain": "additive_manufacturing",
                "task_type": "experiment_design",
                "goal": template["goal"],
                "available_resources": {
                    "equipment": template["equipment"],
                    "materials": template["materials"],
                    "constraints": [
                        "Use only listed equipment and materials.",
                        str(template["constraint"]),
                        "Keep all non-target process settings fixed.",
                    ],
                },
                "evidence_chunk_ids": [template["evidence_chunk_id"]],
                "evidence_summary": template["evidence_text"],
                "evidence": [
                    {
                        "source_id": template["evidence_chunk_id"],
                        "text": template["evidence_text"],
                    }
                ],
            }
        )
    return tasks


def build_chunk_tasks(chunks_path: Path) -> list[dict[str, object]]:
    chunks = list(load_jsonl(chunks_path) or [])
    if not chunks:
        return []

    by_paper: dict[str, list[dict[str, object]]] = {}
    for row in chunks:
        by_paper.setdefault(str(row["paper_id"]), []).append(row)

    tasks: list[dict[str, object]] = []
    for paper_id, rows in by_paper.items():
        evidence_rows = rows[:4]
        evidence_ids = [str(r["chunk_id"]) for r in evidence_rows]
        evidence_summary = " ".join(str(r.get("text", "")).strip() for r in evidence_rows if str(r.get("text", "")).strip())
        evidence_summary = evidence_summary or "Placeholder evidence summary."
        tasks.append(
            {
                "task_id": f"task_{paper_id}_01",
                "paper_id": paper_id,
                "domain": "demo",
                "task_type": "experiment_design",
                "goal": "Design one feasible evidence-grounded experiment based on the paper.",
                "available_resources": {
                    "equipment": [],
                    "materials": [],
                    "constraints": ["Use only evidence-supported resources."],
                },
                "evidence_chunk_ids": evidence_ids,
                "evidence_summary": evidence_summary,
            }
        )
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build canonical task rows either from processed chunks or from a built-in multi-task demo batch."
    )
    parser.add_argument("--config", required=False)
    parser.add_argument(
        "--task-source",
        choices=["auto", "demo", "chunks"],
        default="auto",
        help="Task source. 'auto' uses processed chunks when available, otherwise the built-in demo batch.",
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=Path("data/processed/chunks/chunks.jsonl"),
        help="Processed chunks JSONL path used when task-source is 'chunks' or when auto mode finds it.",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("data/processed/tasks/tasks.jsonl"),
        help="Output tasks JSONL path.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Optional cap on the number of tasks to write.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_tasks is not None and args.max_tasks < 1:
        raise SystemExit("--max-tasks must be at least 1 when provided.")

    if args.task_source == "demo":
        source_name = "demo"
        tasks = build_demo_task_templates()
    elif args.task_source == "chunks":
        source_name = "chunks"
        tasks = build_chunk_tasks(args.chunks_path)
    else:
        chunk_tasks = build_chunk_tasks(args.chunks_path)
        if chunk_tasks:
            source_name = "chunks"
            tasks = chunk_tasks
        else:
            source_name = "demo"
            tasks = build_demo_task_templates()

    if not tasks:
        raise SystemExit(
            f"No tasks could be built from source={source_name}. "
            "If you want the local-only demo batch, rerun with --task-source demo."
        )

    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]

    out = args.out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    print(f"Saved {len(tasks)} task row(s) to {out}")
    print(f"task_source={source_name}")


if __name__ == "__main__":
    main()
