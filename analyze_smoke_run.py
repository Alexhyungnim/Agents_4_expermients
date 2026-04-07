from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import matplotlib.pyplot as plt


SMOKE_DIR = "smoke_run"
OUTPUT_DIR = "smoke_run_analysis_by_case"


def extract_json_from_text(text: str) -> Dict[str, Any]:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced_match:
        return json.loads(fenced_match.group(1))

    fenced_match = re.search(r"```\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced_match:
        return json.loads(fenced_match.group(1))

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])

    raise ValueError("Could not find valid JSON in file.")


def parse_filename(path: Path) -> Dict[str, Optional[str]]:
    stem = path.stem
    m = re.match(r"(.+?)_(trail|trial)(\d+)$", stem)
    if m:
        return {"case_id": m.group(1), "trial_id": m.group(3)}
    return {"case_id": stem, "trial_id": None}


def flatten_result(file_path: Path, result: Dict[str, Any]) -> Dict[str, Any]:
    meta = parse_filename(file_path)

    row: Dict[str, Any] = {
        "file_name": file_path.name,
        "case_id": meta["case_id"],
        "trial_id": meta["trial_id"],
        "hard_fail": result.get("hard_fail"),
        "total_score": result.get("total_score"),
        "overall_verdict": result.get("overall_verdict"),
        "summary": result.get("summary"),
    }

    scores = result.get("scores", {})
    for rubric_name, rubric_info in scores.items():
        row[f"score__{rubric_name}"] = rubric_info.get("score")
        row[f"reason__{rubric_name}"] = rubric_info.get("reason")

    return row


def load_results(smoke_dir: str) -> pd.DataFrame:
    smoke_path = Path(smoke_dir)
    rows: List[Dict[str, Any]] = []

    for file_path in sorted(smoke_path.glob("*.txt")):
        try:
            text = file_path.read_text(encoding="utf-8")
            result = extract_json_from_text(text)
            rows.append(flatten_result(file_path, result))
        except Exception as e:
            print(f"[WARN] Failed to parse {file_path.name}: {e}")

    if not rows:
        raise ValueError("No valid result files were parsed.")

    df = pd.DataFrame(rows)
    df["trial_id_num"] = pd.to_numeric(df["trial_id"], errors="coerce")
    df["total_score"] = pd.to_numeric(df["total_score"], errors="coerce")
    return df


def get_score_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("score__")]


def make_output_dir(output_dir: str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", str(name))


def save_case_summary(case_df: pd.DataFrame, case_dir: Path) -> None:
    score_cols = get_score_columns(case_df)

    case_df.to_csv(case_dir / "raw_results.csv", index=False)

    summary_rows = []
    for col in score_cols:
        rubric = col.replace("score__", "")
        vals = pd.to_numeric(case_df[col], errors="coerce")
        summary_rows.append({
            "rubric": rubric,
            "mean": vals.mean(),
            "std": vals.std(),
            "min": vals.min(),
            "max": vals.max(),
            "trial1": vals.iloc[0] if len(vals) > 0 else None,
            "trial2": vals.iloc[1] if len(vals) > 1 else None,
            "trial3": vals.iloc[2] if len(vals) > 2 else None,
        })

    pd.DataFrame(summary_rows).to_csv(case_dir / "rubric_summary.csv", index=False)

    total_summary = pd.DataFrame([{
        "case_id": case_df["case_id"].iloc[0],
        "n_trials": len(case_df),
        "total_mean": case_df["total_score"].mean(),
        "total_std": case_df["total_score"].std(),
        "total_min": case_df["total_score"].min(),
        "total_max": case_df["total_score"].max(),
    }])
    total_summary.to_csv(case_dir / "total_summary.csv", index=False)


def plot_case_total_scores(case_df: pd.DataFrame, case_dir: Path) -> None:
    plot_df = case_df.sort_values("trial_id_num")

    plt.figure(figsize=(6, 4))
    plt.bar(plot_df["trial_id"].astype(str), plot_df["total_score"])
    plt.xlabel("Trial")
    plt.ylabel("Total score")
    plt.title(f"Total score by trial: {plot_df['case_id'].iloc[0]}")
    plt.tight_layout()
    plt.savefig(case_dir / "total_score_by_trial.png", dpi=200)
    plt.close()


def plot_case_rubric_bars(case_df: pd.DataFrame, case_dir: Path) -> None:
    score_cols = get_score_columns(case_df)
    plot_df = case_df.sort_values("trial_id_num")

    n = len(score_cols)
    fig, axes = plt.subplots(n, 1, figsize=(8, max(3 * n, 8)))

    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, score_cols):
        rubric = col.replace("score__", "")
        vals = pd.to_numeric(plot_df[col], errors="coerce")
        ax.bar(plot_df["trial_id"].astype(str), vals)
        ax.set_ylim(-0.1, 2.1)
        ax.set_ylabel("Score")
        ax.set_title(rubric)

    plt.xlabel("Trial")
    plt.tight_layout()
    plt.savefig(case_dir / "rubric_scores_by_trial.png", dpi=200)
    plt.close()


def plot_case_rubric_heatmap(case_df: pd.DataFrame, case_dir: Path) -> None:
    score_cols = get_score_columns(case_df)
    plot_df = case_df.sort_values("trial_id_num").copy()

    heatmap_df = pd.DataFrame({
        col.replace("score__", ""): pd.to_numeric(plot_df[col], errors="coerce").values
        for col in score_cols
    }, index=[f"trial{t}" for t in plot_df["trial_id"].astype(str)])

    plt.figure(figsize=(max(8, len(score_cols) * 0.7), 3.5))
    plt.imshow(heatmap_df.values, aspect="auto", vmin=0, vmax=2)
    plt.xticks(range(len(heatmap_df.columns)), heatmap_df.columns, rotation=45, ha="right")
    plt.yticks(range(len(heatmap_df.index)), heatmap_df.index)
    plt.colorbar(label="Score")
    plt.title(f"Rubric score heatmap: {plot_df['case_id'].iloc[0]}")
    plt.tight_layout()
    plt.savefig(case_dir / "rubric_heatmap.png", dpi=200)
    plt.close()


def plot_case_rubric_distribution(case_df: pd.DataFrame, case_dir: Path) -> None:
    """
    For each rubric in this case, count how many times score 0/1/2 occurred across the 3 trials.
    """
    score_cols = get_score_columns(case_df)

    rubrics = []
    count0 = []
    count1 = []
    count2 = []

    for col in score_cols:
        rubric = col.replace("score__", "")
        vals = pd.to_numeric(case_df[col], errors="coerce")
        rubrics.append(rubric)
        count0.append((vals == 0).sum())
        count1.append((vals == 1).sum())
        count2.append((vals == 2).sum())

    x = range(len(rubrics))
    plt.figure(figsize=(max(10, len(rubrics) * 0.7), 5))
    plt.bar(x, count0, label="score 0")
    plt.bar(x, count1, bottom=count0, label="score 1")
    bottom2 = [a + b for a, b in zip(count0, count1)]
    plt.bar(x, count2, bottom=bottom2, label="score 2")

    plt.xticks(list(x), rubrics, rotation=45, ha="right")
    plt.ylabel("Count across trials")
    plt.title(f"Rubric score distribution within case: {case_df['case_id'].iloc[0]}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(case_dir / "rubric_distribution_stacked.png", dpi=200)
    plt.close()


def save_global_case_table(df: pd.DataFrame, outdir: Path) -> None:
    rows = []
    for case_id, case_df in df.groupby("case_id"):
        rows.append({
            "case_id": case_id,
            "n_trials": len(case_df),
            "mean_total_score": pd.to_numeric(case_df["total_score"], errors="coerce").mean(),
            "std_total_score": pd.to_numeric(case_df["total_score"], errors="coerce").std(),
            "min_total_score": pd.to_numeric(case_df["total_score"], errors="coerce").min(),
            "max_total_score": pd.to_numeric(case_df["total_score"], errors="coerce").max(),
        })

    pd.DataFrame(rows).sort_values("case_id").to_csv(outdir / "all_case_total_summary.csv", index=False)


def main() -> None:
    outdir = make_output_dir(OUTPUT_DIR)
    df = load_results(SMOKE_DIR)

    print(f"Parsed {len(df)} result files.")
    print(df[["file_name", "case_id", "trial_id", "total_score", "overall_verdict"]])

    df.to_csv(outdir / "all_results_flat.csv", index=False)
    save_global_case_table(df, outdir)

    for case_id, case_df in df.groupby("case_id"):
        case_name = sanitize_name(case_id)
        case_dir = outdir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)

        save_case_summary(case_df, case_dir)
        plot_case_total_scores(case_df, case_dir)
        plot_case_rubric_bars(case_df, case_dir)
        plot_case_rubric_heatmap(case_df, case_dir)
        plot_case_rubric_distribution(case_df, case_dir)

        print(f"Saved case analysis: {case_dir}")

    print(f"Done. Outputs saved to: {outdir}")


if __name__ == "__main__":
    main()