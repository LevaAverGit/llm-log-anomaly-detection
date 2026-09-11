"""Cross-model comparison: the same corpus and prompts across many models.

Where :mod:`eval.compare` compares *approaches* (rules vs one LLM vs hybrid),
this module compares *models*: it runs the same labeled corpus and the same
prompt versions across several models (local Ollama models such as gemma3, a
Qwen and a DeepSeek, plus the cloud GigaChat) and lays the result out as a
``model x prompt-version x F1`` table, with a short auto-generated per-model
error breakdown that says where each model errs differently.

Design
------
- A run is one ``(provider, model, prompt version)`` triple. Its predictions
  live in ``reports/predictions/models/<run_key>.jsonl`` where ``run_key`` is
  :func:`run_key` — model-specific, so different models never collide.
- Scoring reuses :func:`eval.metrics.score` verbatim, the same code path the
  rules/LLM comparison uses, so the numbers are directly comparable.
- Rendering targets its **own** README markers (``cross_model_table`` and
  ``cross_model_errors``); it never touches the rules-vs-LLM section.

Model specs come from a small JSON config (see :func:`load_models_config`) or a
``--models`` CLI string, so the exact Qwen/DeepSeek/GigaChat model names are
parameters, not hard-coded.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from detector.schema import LabeledUnit, Prediction
from eval.metrics import (
    DEFAULT_LABELS_PATH,
    DEFAULT_REPORTS_DIR,
    RunMetrics,
    load_labels,
    load_predictions,
    score,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_README_PATH = _REPO_ROOT / "README.md"
DEFAULT_MODELS_CONFIG = _REPO_ROOT / "models.json"

# Cross-model prediction files live under this subdirectory of the predictions
# directory so they never overwrite the committed rules/LLM/hybrid runs.
MODELS_SUBDIR = "models"
RULES_RUN = "rules"

KNOWN_PROVIDERS = ("ollama", "gigachat", "mock")
DEFAULT_VERSIONS: tuple[str, ...] = ("v1", "v3")

MARKER_TABLE = "cross_model_table"
MARKER_ERRORS = "cross_model_errors"

_MISSING_NOTE = "-- (run make run-models)"


# --------------------------------------------------------------------------- #
# Model specs.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    """One model to evaluate: a provider, a model name and a display label."""

    provider: str
    model: str
    label: str

    @property
    def key(self) -> str:
        return model_key(self.provider, self.model)


def _slug(name: str) -> str:
    """Filesystem/id-safe slug (mirrors detector.llm's cache-name convention)."""
    return re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_")


def model_key(provider: str, model: str) -> str:
    """Stable id for a (provider, model) pair, e.g. ``ollama__gemma3_latest``."""
    return f"{_slug(provider)}__{_slug(model)}"


def run_key(provider: str, model: str, version: str) -> str:
    """Prediction run key for one (provider, model, version), e.g.
    ``ollama__gemma3_latest__v1``."""
    version = version if version.startswith("v") else f"v{version}"
    return f"{model_key(provider, model)}__{version}"


def default_label(provider: str, model: str) -> str:
    """A readable default label when the config gives none."""
    return f"{model} ({provider})"


def parse_model_spec(entry: str, default_provider: str = "ollama") -> ModelSpec:
    """Parse one ``--models`` entry into a :class:`ModelSpec`.

    Accepts ``provider:model`` (``ollama:qwen2.5:7b``, ``gigachat:GigaChat``) or a
    bare model name (``gemma3:latest``), which takes ``default_provider``. Only a
    leading token that is a *known* provider is treated as the provider, so the
    ``:`` inside a model tag like ``gemma3:latest`` is preserved.
    """
    entry = entry.strip()
    head, sep, tail = entry.partition(":")
    if sep and head.lower() in KNOWN_PROVIDERS:
        provider, model = head.lower(), tail.strip()
    else:
        provider, model = default_provider.lower(), entry
    if not model:
        raise ValueError(f"empty model name in --models entry {entry!r}")
    return ModelSpec(provider=provider, model=model, label=default_label(provider, model))


def parse_models_arg(arg: str, default_provider: str = "ollama") -> list[ModelSpec]:
    """Parse a comma-separated ``--models`` string into specs."""
    return [
        parse_model_spec(part, default_provider)
        for part in arg.split(",")
        if part.strip()
    ]


def load_models_config(
    path: Path | str = DEFAULT_MODELS_CONFIG,
) -> tuple[list[ModelSpec], list[str]]:
    """Load model specs and prompt versions from a JSON config file.

    Shape::

        {"models": [{"provider": "ollama", "model": "gemma3:latest",
                     "label": "gemma3"}, ...],
         "versions": ["v1", "v3"]}

    ``label`` is optional. Returns ``(specs, versions)``.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    specs: list[ModelSpec] = []
    for row in data.get("models", []):
        provider = str(row.get("provider", "ollama")).lower()
        model = str(row["model"])
        label = str(row.get("label") or default_label(provider, model))
        specs.append(ModelSpec(provider=provider, model=model, label=label))
    versions = [str(v) for v in data.get("versions", DEFAULT_VERSIONS)]
    return specs, versions


def normalize_versions(versions: list[str]) -> list[str]:
    """Normalise ``['1', 'v3']`` -> ``['v1', 'v3']`` preserving order."""
    out: list[str] = []
    for v in versions:
        v = v if str(v).startswith("v") else f"v{v}"
        if v not in out:
            out.append(v)
    return out


# --------------------------------------------------------------------------- #
# Scoring.
# --------------------------------------------------------------------------- #
def _predictions_path(pred_dir: Path, key: str) -> Path:
    return Path(pred_dir) / f"{key}.jsonl"


def score_models(
    labels: list[LabeledUnit],
    specs: list[ModelSpec],
    versions: list[str],
    models_dir: Path | str,
) -> dict[tuple[str, str], RunMetrics]:
    """Score every ``(model, version)`` run that has a prediction file.

    Returns a mapping ``(model_key, version) -> RunMetrics`` (runs without a
    prediction file are simply absent).
    """
    models_dir = Path(models_dir)
    out: dict[tuple[str, str], RunMetrics] = {}
    for spec in specs:
        for version in versions:
            key = run_key(spec.provider, spec.model, version)
            path = _predictions_path(models_dir, key)
            if not path.exists():
                continue
            out[(spec.key, version)] = score(
                labels, load_predictions(path), run=key, display_name=f"{spec.label} · {version}"
            )
    return out


def score_rules_baseline(
    labels: list[LabeledUnit], models_dir: Path | str
) -> Optional[RunMetrics]:
    """Score the rules baseline reference row if its prediction file exists."""
    path = _predictions_path(Path(models_dir), RULES_RUN)
    if not path.exists():
        return None
    return score(labels, load_predictions(path), run=RULES_RUN, display_name="Rules (Sigma baseline)")


# --------------------------------------------------------------------------- #
# Markdown rendering.
# --------------------------------------------------------------------------- #
def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def model_version_table_markdown(
    specs: list[ModelSpec],
    versions: list[str],
    metrics_by: dict[tuple[str, str], RunMetrics],
    rules_metric: Optional[RunMetrics] = None,
    missing_note: str = _MISSING_NOTE,
) -> str:
    """The ``model x prompt-version`` F1 table (one F1 per cell).

    Rows are the models (plus, when available, a Rules baseline reference row);
    columns are the prompt versions. A cell with no prediction yet renders a
    short pending note so the table stays stable and honest.
    """
    versions = normalize_versions(versions)
    headers = ["Model"] + [f"F1 ({v})" for v in versions]
    rows: list[list[str]] = []

    if rules_metric is not None:
        # Rules ignore the prompt, so the same F1 is shown in every column.
        rows.append(
            [rules_metric.display_name + " *"]
            + [f"{rules_metric.f1:.2f}" for _ in versions]
        )

    for spec in specs:
        cells = [spec.label]
        for version in versions:
            metric = metrics_by.get((spec.key, version))
            cells.append(f"{metric.f1:.2f}" if metric is not None else missing_note)
        rows.append(cells)

    table = _md_table(headers, rows)
    if rules_metric is not None:
        table += "\n\n_\\* Rules are deterministic and do not use a prompt; the same F1 is shown in every column for reference._"
    return table


def _breakdown_counts(
    labels: list[LabeledUnit], preds: list[Prediction]
) -> dict[str, object]:
    """Confusion counts plus per-source FP and per-technique FN for one run."""
    by_id = {p.unit_id: p for p in preds}
    tp = fp = fn = tn = 0
    fp_by_source: dict[str, int] = {}
    fn_by_technique: dict[str, int] = {}

    for unit in labels:
        pred = by_id.get(unit.unit_id)
        predicted = bool(pred and pred.is_incident)
        truth = unit.is_incident
        if truth and predicted:
            tp += 1
        elif truth and not predicted:
            fn += 1
            tech = unit.technique or "unspecified"
            fn_by_technique[tech] = fn_by_technique.get(tech, 0) + 1
        elif not truth and predicted:
            fp += 1
            src = unit.event.source.value
            fp_by_source[src] = fp_by_source.get(src, 0) + 1
        else:
            tn += 1

    n = len(labels)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall,
        "fp_per_1000": (fp / n * 1000.0) if n else 0.0,
        "fp_by_source": fp_by_source,
        "fn_by_technique": fn_by_technique,
    }


def _top(counter: dict[str, int], k: int = 3) -> str:
    items = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    return ", ".join(f"{name} ({count})" for name, count in items)


def per_model_error_breakdown_markdown(
    labels: list[LabeledUnit],
    specs: list[ModelSpec],
    versions: list[str],
    models_dir: Path | str,
    empty_note: str = "No model predictions yet — run `make run-models`.",
) -> str:
    """A short, auto-generated per-model error breakdown.

    For each model/version with predictions, reports recall, precision and the
    false-positive rate, then the classes it under-detects (incident techniques
    it misses) and over-detects (log sources it false-alarms on). Reading the
    bullets side by side shows where each model errs differently.
    """
    versions = normalize_versions(versions)
    models_dir = Path(models_dir)
    lines: list[str] = []

    for spec in specs:
        sub: list[str] = []
        for version in versions:
            path = _predictions_path(models_dir, run_key(spec.provider, spec.model, version))
            if not path.exists():
                continue
            c = _breakdown_counts(labels, load_predictions(path))
            under = _top(c["fn_by_technique"]) or "no incident classes missed"
            over = _top(c["fp_by_source"]) or "no false positives"
            sub.append(
                f"  - **{version}**: recall {c['recall']:.2f} "
                f"({c['tp']}/{c['tp'] + c['fn']}), precision {c['precision']:.2f}, "
                f"FP {c['fp_per_1000']:.0f}/1000. "
                f"Under-detects: {under}. Over-detects: {over}."
            )
        if sub:
            lines.append(f"- **{spec.label}**")
            lines.extend(sub)

    if not lines:
        return f"_{empty_note}_"
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# README rendering (own markers only).
# --------------------------------------------------------------------------- #
def _replace_block(text: str, marker: str, inner: str) -> tuple[str, int]:
    pattern = re.compile(
        r"(<!-- BEGIN:%s -->)(.*?)(<!-- END:%s -->)" % (re.escape(marker), re.escape(marker)),
        re.DOTALL,
    )

    def _sub(match: re.Match) -> str:
        return f"{match.group(1)}\n{inner}\n{match.group(3)}"

    return pattern.subn(_sub, text)


def write_reports(
    labels: list[LabeledUnit],
    specs: list[ModelSpec],
    versions: list[str],
    models_dir: Path | str,
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
) -> dict[str, Path]:
    """Write ``models_summary.{md,csv,json}`` with the full per-run metrics."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_by = score_models(labels, specs, versions, models_dir)
    rules_metric = score_rules_baseline(labels, models_dir)

    all_metrics = ([rules_metric] if rules_metric is not None else []) + [
        metrics_by[(spec.key, v)]
        for spec in specs
        for v in normalize_versions(versions)
        if (spec.key, v) in metrics_by
    ]

    import pandas as pd  # local: keep import cost off the hot path

    frame = pd.DataFrame(
        [
            {
                "run": m.run,
                "model": m.display_name,
                "precision": round(m.precision, 4),
                "recall": round(m.recall, 4),
                "f1": round(m.f1, 4),
                "fp_per_1000": round(m.fp_per_1000, 2),
                "tp": m.tp, "fp": m.fp, "fn": m.fn, "tn": m.tn,
                "n_units": m.n_units,
            }
            for m in all_metrics
        ],
        columns=["run", "model", "precision", "recall", "f1", "fp_per_1000",
                 "tp", "fp", "fn", "tn", "n_units"],
    )

    md_path = reports_dir / "models_summary.md"
    csv_path = reports_dir / "models_summary.csv"
    json_path = reports_dir / "models_summary.json"

    md = [
        "# Cross-model comparison\n",
        model_version_table_markdown(specs, versions, metrics_by, rules_metric),
        "\n\n## Per-model error breakdown\n",
        per_model_error_breakdown_markdown(labels, specs, versions, models_dir),
        "\n",
    ]
    md_path.write_text("\n".join(md), encoding="utf-8")
    frame.to_csv(csv_path, index=False)
    frame.to_json(json_path, orient="records", indent=2)
    return {"markdown": md_path, "csv": csv_path, "json": json_path}


def render(
    specs: list[ModelSpec],
    versions: list[str],
    models_dir: Path | str,
    readme_path: Path | str = DEFAULT_README_PATH,
    labels_path: Path | str = DEFAULT_LABELS_PATH,
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
) -> list[str]:
    """Fill the cross-model markers in README from the current predictions.

    Only the ``cross_model_table`` and ``cross_model_errors`` blocks are
    touched; the rules-vs-LLM section is left exactly as it is. Returns the list
    of markers actually written. Unlike the rules-vs-LLM render this never
    refuses on empty predictions: it writes an honest placeholder table so the
    section is coherent before any model has run.
    """
    readme_path = Path(readme_path)
    versions = normalize_versions(versions)
    labels = load_labels(labels_path)

    metrics_by = score_models(labels, specs, versions, models_dir)
    rules_metric = score_rules_baseline(labels, models_dir)
    write_reports(labels, specs, versions, models_dir, reports_dir)

    table_md = model_version_table_markdown(specs, versions, metrics_by, rules_metric)
    errors_md = per_model_error_breakdown_markdown(labels, specs, versions, models_dir)

    text = readme_path.read_text(encoding="utf-8")
    written: list[str] = []
    missing: list[str] = []
    for marker, inner in ((MARKER_TABLE, table_md), (MARKER_ERRORS, errors_md)):
        text, n = _replace_block(text, marker, inner)
        (written if n else missing).append(marker)
    readme_path.write_text(text, encoding="utf-8")

    if missing:
        joined = ", ".join(f"<!-- BEGIN:{m} -->/<!-- END:{m} -->" for m in missing)
        print(f"warning: cross-model marker pair(s) not found in {readme_path.name}: {joined}")
    return written


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Render the cross-model comparison into README (from existing predictions)."
    )
    parser.add_argument("--models-config", default=str(DEFAULT_MODELS_CONFIG))
    parser.add_argument("--models", default=None, help="Comma-separated provider:model specs (overrides config).")
    parser.add_argument("--versions", default=None, help="Comma-separated prompt versions (overrides config).")
    parser.add_argument("--readme", default=str(DEFAULT_README_PATH))
    parser.add_argument("--labels", default=str(DEFAULT_LABELS_PATH))
    parser.add_argument("--models-dir", default=None)
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    args = parser.parse_args()

    if args.models:
        specs = parse_models_arg(args.models)
        versions = args.versions.split(",") if args.versions else list(DEFAULT_VERSIONS)
    else:
        specs, versions = load_models_config(args.models_config)
        if args.versions:
            versions = args.versions.split(",")

    models_dir = args.models_dir or (Path(args.reports_dir) / "predictions" / MODELS_SUBDIR)
    written = render(
        specs, versions, models_dir,
        readme_path=args.readme, labels_path=args.labels, reports_dir=args.reports_dir,
    )
    print(f"Updated {args.readme}: wrote {', '.join(written) if written else 'nothing'}.")


if __name__ == "__main__":
    main()
