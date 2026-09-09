"""Cross-approach comparison: one pandas table over every scored run.

This turns the per-run :class:`eval.metrics.RunMetrics` objects into a single
summary a reviewer can read in seconds -- the rules baseline next to each LLM
prompt version next to the hybrid -- and writes it to ``reports/`` as Markdown,
CSV and JSON.

Run keys and display names live in :data:`RUN_REGISTRY`. The headline README
table shows exactly the four rows in :data:`README_ROWS`; the fuller report
under ``reports/`` includes every run that has a prediction file (so extra
prompt versions such as ``llm_v2`` still appear when present).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from eval.metrics import (
    DEFAULT_LABELS_PATH,
    DEFAULT_PREDICTIONS_DIR,
    DEFAULT_REPORTS_DIR,
    RunMetrics,
    load_labels,
    load_predictions,
    predictions_path,
    score,
)

# Run key -> display name. Order here is the order rows appear in the tables.
RUN_REGISTRY: list[tuple[str, str]] = [
    ("rules", "Rules (Sigma baseline)"),
    ("llm_v1", "LLM — prompt v1"),
    ("llm_v2", "LLM — prompt v2"),
    ("llm_v3", "LLM — prompt v3"),
    ("hybrid", "Hybrid (rules ∪ LLM)"),
]
_DISPLAY_NAME = dict(RUN_REGISTRY)

# The four rows the top-level README promises, in order.
README_ROWS: list[str] = ["rules", "llm_v1", "llm_v3", "hybrid"]


def display_name(run_key: str) -> str:
    """Human-readable name for a run key (falls back to a title-cased key)."""
    return _DISPLAY_NAME.get(run_key, run_key.replace("_", " ").title())


def discover_runs(pred_dir: Path | str = DEFAULT_PREDICTIONS_DIR) -> list[str]:
    """List run keys that have a prediction file, registry order first."""
    pred_dir = Path(pred_dir)
    present = {p.stem for p in pred_dir.glob("*.jsonl")}
    ordered = [key for key, _ in RUN_REGISTRY if key in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def summarize(
    labels_path: Path | str = DEFAULT_LABELS_PATH,
    pred_dir: Path | str = DEFAULT_PREDICTIONS_DIR,
    runs: Optional[list[str]] = None,
) -> list[RunMetrics]:
    """Score each requested run (default: every run with a prediction file)."""
    labels = load_labels(labels_path)
    run_keys = runs if runs is not None else discover_runs(pred_dir)
    results: list[RunMetrics] = []
    for run_key in run_keys:
        path = predictions_path(run_key, pred_dir)
        if not path.exists():
            continue
        results.append(
            score(labels, load_predictions(path), run=run_key, display_name=display_name(run_key))
        )
    return results


# --------------------------------------------------------------------------- #
# Frames and Markdown.
# --------------------------------------------------------------------------- #
def summary_frame(metrics: list[RunMetrics]) -> pd.DataFrame:
    """A full DataFrame (all columns) for the reports directory."""
    rows = [
        {
            "run": m.run,
            "approach": m.display_name,
            "precision": round(m.precision, 4),
            "recall": round(m.recall, 4),
            "f1": round(m.f1, 4),
            "fp_per_1000": round(m.fp_per_1000, 2),
            "tp": m.tp,
            "fp": m.fp,
            "fn": m.fn,
            "tn": m.tn,
            "technique_accuracy": (
                round(m.technique_accuracy, 3) if m.technique_accuracy is not None else None
            ),
            "n_units": m.n_units,
            "n_missing": m.n_missing,
        }
        for m in metrics
    ]
    return pd.DataFrame(rows, columns=[
        "run", "approach", "precision", "recall", "f1", "fp_per_1000",
        "tp", "fp", "fn", "tn", "technique_accuracy", "n_units", "n_missing",
    ])


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a GitHub-flavoured Markdown table (no external dependency)."""
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def readme_table_markdown(
    metrics: list[RunMetrics],
    rows: list[str] = README_ROWS,
    missing_note: str = "-- (run make run-llm)",
) -> str:
    """The four-column headline table for README.

    Rows are the fixed README set; a run with no prediction yet renders with a
    short pending note in its first metric cell (``missing_note``) so the table
    structure stays stable and stays honest about what has not been computed yet
    -- and tells the reader exactly how to fill it in.
    """
    by_key = {m.run: m for m in metrics}
    headers = ["Approach", "Precision", "Recall", "F1", "False positives / 1000"]
    table_rows: list[list[str]] = []
    for key in rows:
        name = display_name(key)
        metric = by_key.get(key)
        if metric is None:
            table_rows.append([name, missing_note, "--", "--", "--"])
        else:
            table_rows.append([
                name,
                f"{metric.precision:.2f}",
                f"{metric.recall:.2f}",
                f"{metric.f1:.2f}",
                f"{metric.fp_per_1000:.1f}",
            ])
    return _md_table(headers, table_rows)


def write_reports(
    metrics: list[RunMetrics],
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
) -> dict[str, Path]:
    """Write summary.md / summary.csv / summary.json and return their paths."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    frame = summary_frame(metrics)

    md_path = reports_dir / "summary.md"
    csv_path = reports_dir / "summary.csv"
    json_path = reports_dir / "summary.json"

    md_parts = ["# Results summary\n", readme_table_markdown(metrics), "\n"]
    if metrics:
        md_parts.append("\n## Full metrics\n")
        md_parts.append(_md_table(
            ["Run", "P", "R", "F1", "FP/1000", "TP", "FP", "FN", "TN", "Technique acc."],
            [[
                m.run,
                f"{m.precision:.3f}", f"{m.recall:.3f}", f"{m.f1:.3f}",
                f"{m.fp_per_1000:.1f}",
                str(m.tp), str(m.fp), str(m.fn), str(m.tn),
                (f"{m.technique_accuracy:.0%}" if m.technique_accuracy is not None else "—"),
            ] for m in metrics],
        ))
        md_parts.append("\n")
    md_path.write_text("\n".join(md_parts), encoding="utf-8")

    frame.to_csv(csv_path, index=False)
    frame.to_json(json_path, orient="records", indent=2)
    return {"markdown": md_path, "csv": csv_path, "json": json_path}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compare detection approaches into one table.")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS_PATH))
    parser.add_argument("--pred-dir", default=str(DEFAULT_PREDICTIONS_DIR))
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    args = parser.parse_args()

    metrics = summarize(args.labels, args.pred_dir)
    if not metrics:
        print(
            f"No prediction files found in {args.pred_dir}. "
            "Run the detector first (e.g. `make run`)."
        )
        return

    print(readme_table_markdown(metrics))
    print()
    paths = write_reports(metrics, args.reports_dir)
    print("Wrote:")
    for kind, path in paths.items():
        print(f"  {kind:>8}: {path}")


if __name__ == "__main__":
    main()
