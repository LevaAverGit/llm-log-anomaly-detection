"""Automatic error analysis: where the rules and the LLM disagree.

The README's two example blocks are the honest heart of the comparison, so they
are selected mechanically from the predictions rather than cherry-picked:

- **Rules miss, LLM catches** -- ground-truth incidents the Sigma rules left as
  benign but the LLM flagged. These show the LLM's recall on attacks that have
  no crisp signature (semantic payloads, context a threshold rule cannot see).
- **LLM misfires, rules are right** -- units where the LLM is wrong and the
  rules are right: benign look-alikes the LLM over-flags (a false positive), or,
  failing that, incidents the rules matched but the LLM let through. These are
  where the LLM costs precision -- the negative result the project set out to
  measure.

For each example the analysis records the unit id, source, ground truth, both
predictions, a one-line "why" (drawn from the label rationale) and a raw log
snippet, so a reader can check the call themselves.

Selection is deterministic: candidates are ranked to spread across log sources
first, then by unit id, so the same corpus and predictions always yield the same
examples.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from detector.schema import Event, LabeledUnit, Prediction
from eval.metrics import (
    DEFAULT_LABELS_PATH,
    DEFAULT_PREDICTIONS_DIR,
    DEFAULT_REPORTS_DIR,
    load_labels,
    load_predictions,
    predictions_path,
)

DEFAULT_RULES_RUN = "rules"
DEFAULT_LLM_RUN = "llm_v3"
_N_EXAMPLES = 3
_WHY_MAX = 200
_RAW_MAX = 160
_MAX_RAW_LINES = 2


# --------------------------------------------------------------------------- #
# Raw-log extraction (handles every shape build_corpus.py emits).
# --------------------------------------------------------------------------- #
def raw_snippet(event: Event, max_lines: int = _MAX_RAW_LINES) -> list[str]:
    """Return up to ``max_lines`` human-readable raw lines for an event."""
    raw = event.raw or {}
    lines: list[str] = []

    if isinstance(raw.get("sample_lines"), list):
        lines = [str(x) for x in raw["sample_lines"]]
    elif isinstance(raw.get("line"), str):
        lines = [raw["line"]]
    elif isinstance(raw.get("event"), dict):
        lines = [json.dumps(raw["event"], ensure_ascii=False)]
    elif isinstance(raw.get("events"), list) and raw["events"]:
        lines = [json.dumps(raw["events"][0], ensure_ascii=False)]
    else:
        trimmed = {k: v for k, v in raw.items() if k != "source"}
        lines = [json.dumps(trimmed, ensure_ascii=False)]

    return [_truncate(line, _RAW_MAX) for line in lines[:max_lines]]


def _truncate(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _one_line_why(unit: LabeledUnit) -> str:
    """Condense the label rationale to a single clause."""
    rationale = " ".join(unit.rationale.split())
    # Prefer the first sentence; fall back to a hard truncation.
    match = re.match(r"(.+?[.!?])(\s|$)", rationale)
    text = match.group(1) if match else rationale
    return _truncate(text, _WHY_MAX)


# --------------------------------------------------------------------------- #
# Example selection.
# --------------------------------------------------------------------------- #
def _example(
    unit: LabeledUnit,
    rules_pred: Optional[Prediction],
    llm_pred: Optional[Prediction],
    headline: str,
) -> dict[str, object]:
    return {
        "unit_id": unit.unit_id,
        "source": unit.event.source.value,
        "truth": unit.label.value,
        "technique": unit.technique,
        "action": unit.event.action,
        "actor": unit.event.actor,
        "rules_is_incident": bool(rules_pred and rules_pred.is_incident),
        "llm_is_incident": bool(llm_pred and llm_pred.is_incident),
        "headline": headline,
        "why": _one_line_why(unit),
        "raw": raw_snippet(unit.event),
    }


def _pick_diverse(candidates: list[dict[str, object]], k: int) -> list[dict[str, object]]:
    """Greedily pick ``k`` examples that spread across sources, deterministically."""
    pool = sorted(candidates, key=lambda c: (str(c["source"]), str(c["unit_id"])))
    chosen: list[dict[str, object]] = []
    used_sources: set[str] = set()

    # First pass: one per not-yet-seen source.
    for cand in pool:
        if len(chosen) >= k:
            break
        if cand["source"] not in used_sources:
            chosen.append(cand)
            used_sources.add(str(cand["source"]))
    # Second pass: fill remaining slots with whatever is left, in order.
    for cand in pool:
        if len(chosen) >= k:
            break
        if cand not in chosen:
            chosen.append(cand)
    return chosen


def select_disagreements(
    labels: list[LabeledUnit],
    rules_preds: list[Prediction],
    llm_preds: list[Prediction],
    k: int = _N_EXAMPLES,
) -> dict[str, list[dict[str, object]]]:
    """Return the two ranked example sets keyed ``rules_miss_llm_catch`` and
    ``llm_misfire_rules_right``."""
    rules_by_id = {p.unit_id: p for p in rules_preds}
    llm_by_id = {p.unit_id: p for p in llm_preds}

    rules_miss_llm_catch: list[dict[str, object]] = []
    llm_fp_rules_right: list[dict[str, object]] = []   # LLM over-flags a benign unit
    llm_fn_rules_right: list[dict[str, object]] = []   # LLM misses an incident rules caught

    for unit in labels:
        r = rules_by_id.get(unit.unit_id)
        m = llm_by_id.get(unit.unit_id)
        r_inc = bool(r and r.is_incident)
        m_inc = bool(m and m.is_incident)

        if unit.is_incident and not r_inc and m_inc:
            rules_miss_llm_catch.append(
                _example(unit, r, m, "rules stayed silent, the LLM flagged it")
            )
        elif not unit.is_incident and m_inc and not r_inc:
            llm_fp_rules_right.append(
                _example(unit, r, m, "the LLM raised a false alarm, the rules stayed silent")
            )
        elif unit.is_incident and r_inc and not m_inc:
            llm_fn_rules_right.append(
                _example(unit, r, m, "the rules caught it, the LLM let it through")
            )

    # "LLM misfires but rules are right": false alarms first (the precision cost
    # this project set out to measure), then any incidents the LLM missed.
    llm_misfire = _pick_diverse(llm_fp_rules_right, k)
    if len(llm_misfire) < k:
        remaining = k - len(llm_misfire)
        llm_misfire += _pick_diverse(llm_fn_rules_right, remaining)

    return {
        "rules_miss_llm_catch": _pick_diverse(rules_miss_llm_catch, k),
        "llm_misfire_rules_right": llm_misfire,
    }


# --------------------------------------------------------------------------- #
# Rendering.
# --------------------------------------------------------------------------- #
def _render_examples(examples: list[dict[str, object]], empty_note: str) -> str:
    if not examples:
        return f"_{empty_note}_"
    lines: list[str] = []
    for ex in examples:
        truth = str(ex["truth"])
        if ex["technique"]:
            truth += f" · {ex['technique']}"
        lines.append(
            f"- **{ex['unit_id']}** · {ex['source']} · truth: {truth} — "
            f"{ex['headline']}. {ex['why']}"
        )
        for raw_line in ex["raw"]:
            lines.append(f"  - `{raw_line}`")
    return "\n".join(lines)


def rules_miss_markdown(sets: dict[str, list[dict[str, object]]]) -> str:
    examples = sets["rules_miss_llm_catch"]
    heading = f"**Where rules miss but the LLM catches it ({len(examples)} examples):**"
    return heading + "\n\n" + _render_examples(
        examples, "No such disagreements in the current predictions."
    )


def llm_misfire_markdown(sets: dict[str, list[dict[str, object]]]) -> str:
    examples = sets["llm_misfire_rules_right"]
    heading = f"**Where the LLM misfires but rules are right ({len(examples)} examples):**"
    return heading + "\n\n" + _render_examples(
        examples, "No such disagreements in the current predictions."
    )


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #
def analyze(
    labels_path: Path | str = DEFAULT_LABELS_PATH,
    pred_dir: Path | str = DEFAULT_PREDICTIONS_DIR,
    rules_run: str = DEFAULT_RULES_RUN,
    llm_run: str = DEFAULT_LLM_RUN,
) -> dict[str, list[dict[str, object]]]:
    """Load labels + the rules and LLM runs, return the two example sets."""
    labels = load_labels(labels_path)
    rules_preds = load_predictions(predictions_path(rules_run, pred_dir))
    llm_preds = load_predictions(predictions_path(llm_run, pred_dir))
    return select_disagreements(labels, rules_preds, llm_preds)


def write_reports(
    sets: dict[str, list[dict[str, object]]],
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
) -> dict[str, Path]:
    """Write error_analysis.json and error_analysis.md; return their paths."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "error_analysis.json"
    md_path = reports_dir / "error_analysis.md"

    json_path.write_text(json.dumps(sets, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(
        "# Error analysis\n\n"
        + rules_miss_markdown(sets)
        + "\n\n"
        + llm_misfire_markdown(sets)
        + "\n",
        encoding="utf-8",
    )
    return {"json": json_path, "markdown": md_path}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Auto-select 3+3 rules-vs-LLM disagreements.")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS_PATH))
    parser.add_argument("--pred-dir", default=str(DEFAULT_PREDICTIONS_DIR))
    parser.add_argument("--rules-run", default=DEFAULT_RULES_RUN)
    parser.add_argument("--llm-run", default=DEFAULT_LLM_RUN)
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    args = parser.parse_args()

    missing = [
        predictions_path(run, args.pred_dir)
        for run in (args.rules_run, args.llm_run)
        if not predictions_path(run, args.pred_dir).exists()
    ]
    if missing:
        joined = ", ".join(str(p) for p in missing)
        print(f"Missing prediction file(s): {joined}. Run the detector first (e.g. `make run`).")
        return

    sets = analyze(args.labels, args.pred_dir, args.rules_run, args.llm_run)
    print(rules_miss_markdown(sets))
    print()
    print(llm_misfire_markdown(sets))
    print()
    paths = write_reports(sets, args.reports_dir)
    print("Wrote:")
    for kind, path in paths.items():
        print(f"  {kind:>8}: {path}")


if __name__ == "__main__":
    main()
