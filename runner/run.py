"""Command-line runner for the rules-vs-LLM detection comparison.

This is the integration point of the repository. Given an approach it:

1. loads the labeled corpus (``corpus/labels.jsonl``),
2. builds the requested detector (rules, LLM, or the rules-first hybrid),
3. produces one :class:`~detector.schema.Prediction` per analysis unit,
4. writes those predictions to ``reports/predictions/<run_key>.jsonl`` (the
   contract the ``eval`` layer consumes), and
5. calls ``eval`` to score the runs, refresh ``reports/`` and, for a full run,
   regenerate the README results table and error examples.

Usage
-----
::

    python -m runner.run --approach rules
    python -m runner.run --approach llm  --prompt v1 --provider mock
    python -m runner.run --approach llm  --prompt v3 --from-cache
    python -m runner.run --approach hybrid --prompt v3 --from-cache
    python -m runner.run --all --from-cache      # the four README rows at once

Run keys and the prediction-file contract
------------------------------------------
Each scored run writes ``reports/predictions/<run_key>.jsonl``; every line is a
:class:`~detector.schema.Prediction` serialised with ``model_dump(mode="json")``
(``unit_id`` / ``is_incident`` / ``technique``). Run keys match
``eval.compare.RUN_REGISTRY``:

* ``rules`` -- the Sigma rules baseline,
* ``llm_v1`` / ``llm_v2`` / ``llm_v3`` -- the LLM detector per prompt version,
* ``hybrid`` -- the rules-first hybrid (its LLM arm uses prompt v3).

``--all`` produces the four README rows: ``rules``, ``llm_v1``, ``llm_v3`` and
``hybrid``.

Integration with the sibling components
---------------------------------------
The rules engine and LLM detector are batch-oriented, and the runner drives them
that way so results are correct and cheap:

* the rules engine needs a :class:`rules.engine.RuleContext` built from *all*
  events (two of its rules correlate across units), so the runner builds the
  context once and evaluates every unit against it;
* the LLM detector caches responses on disk and persists them at the end of a
  batch, so the runner classifies in one ``classify_many`` call per run;
* the hybrid runs the rules first and only asks the LLM about the units the
  rules left benign.

Scoring, the comparison table and the README are produced by the ``eval``
package (``eval.compare`` and ``eval.render_readme``); the runner never
re-implements the metrics. Sibling imports are done lazily and, when a component
is not available, the affected run is skipped with a clear message rather than
crashing the whole invocation.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Optional

from detector.hybrid import HybridDetector
from detector.schema import Event, LabeledUnit, Prediction, Verdict

# --------------------------------------------------------------------------- #
# Paths.
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
CORPUS_DEFAULT = ROOT / "corpus" / "labels.jsonl"
SIGMA_DIR = ROOT / "rules" / "sigma"
CACHE_DIR = ROOT / "corpus" / "llm_cache"
PRED_DIR_DEFAULT = ROOT / "reports" / "predictions"
REPORTS_DIR_DEFAULT = ROOT / "reports"

# Method / function names an engine or detector might expose that return a
# Verdict for one Event, in preference order (Prediction-returning ``predict``
# comes late; the result is normalised either way).
_ENTRY_NAMES = ("classify", "judge", "evaluate", "verdict", "detect", "run", "predict", "__call__")

# (run_key, approach, prompt) for --all. Matches eval.compare.README_ROWS.
_ALL_RUNS = (
    ("rules", "rules", None),
    ("llm_v1", "llm", "v1"),
    ("llm_v3", "llm", "v3"),
    ("hybrid", "hybrid", "v3"),
)

# Human-readable names, mirrored from eval.compare.RUN_REGISTRY (used only for
# the stderr progress line; the authoritative table comes from eval).
_DISPLAY = {
    "rules": "Rules (Sigma baseline)",
    "llm_v1": "LLM — prompt v1",
    "llm_v2": "LLM — prompt v2",
    "llm_v3": "LLM — prompt v3",
    "hybrid": "Hybrid (rules ∪ LLM)",
}


# --------------------------------------------------------------------------- #
# Corpus loading.
# --------------------------------------------------------------------------- #
def load_corpus(path: Path) -> list[LabeledUnit]:
    """Load and validate every row of ``labels.jsonl`` into a LabeledUnit."""
    if not path.exists():
        raise FileNotFoundError(f"corpus not found at {path} (build it with `make corpus`)")
    units: list[LabeledUnit] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                units.append(LabeledUnit.model_validate(json.loads(line)))
            except Exception as exc:  # a corrupt corpus is fatal
                raise ValueError(f"{path}:{lineno}: invalid LabeledUnit row: {exc}") from exc
    if not units:
        raise ValueError(f"corpus at {path} is empty")
    return units


# --------------------------------------------------------------------------- #
# Verdict normalisation and reflective helpers.
# --------------------------------------------------------------------------- #
def _as_verdict(result: Any) -> Verdict:
    """Coerce whatever a detector returned into a :class:`Verdict`.

    Detectors are contracted to return a Verdict; this is defensive so a
    component that hands back a Prediction, a bare bool or a plain dict still
    integrates instead of crashing the run.
    """
    if isinstance(result, Verdict):
        return result
    if isinstance(result, Prediction):
        return Verdict(
            is_incident=result.is_incident,
            technique=result.technique,
            confidence=1.0,
            reason="(derived from a Prediction with no confidence/reason)",
        )
    if isinstance(result, bool):
        return Verdict(is_incident=result, technique=None, confidence=1.0, reason="")
    if isinstance(result, dict):
        data = dict(result)
        data.setdefault("is_incident", False)
        data.setdefault("confidence", 1.0)
        data.setdefault("reason", "")
        return Verdict.model_validate(data)
    raise TypeError(f"detector returned an unsupported type: {type(result)!r}")


def _resolve_entry(obj: Any) -> Optional[Any]:
    """Return the first callable on ``obj`` whose name is a known entry point."""
    for name in _ENTRY_NAMES:
        attr = getattr(obj, name, None)
        if callable(attr):
            return attr
    return None


def _select_kwargs(callable_obj: Any, candidates: dict[str, Any]) -> dict[str, Any]:
    """Keep only the candidate kwargs that ``callable_obj`` actually declares."""
    try:
        params = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return {}
    return {name: value for name, value in candidates.items() if name in params}


# --------------------------------------------------------------------------- #
# Rules arm (batch: one RuleContext for the whole corpus).
# --------------------------------------------------------------------------- #
def rules_verdicts(units: list[LabeledUnit]) -> dict[str, Verdict]:
    """Run the rules baseline over every unit and return ``unit_id -> Verdict``.

    Prefers the engine's ``RuleContext``+``evaluate`` path so the cross-unit
    correlation rules (Windows account-after-failure, cloud IAM-after-failure)
    see the whole batch. Falls back to a batch ``run_rules`` and finally a
    per-event entry point.
    """
    from rules import engine as rules_engine  # lazy: may not exist yet

    events = [u.event for u in units]

    # Preferred: build the correlation context once, then evaluate each unit.
    rule_context = getattr(rules_engine, "RuleContext", None)
    evaluate = getattr(rules_engine, "evaluate", None)
    if rule_context is not None and callable(evaluate) and hasattr(rule_context, "build"):
        ctx = None
        for build_args in ((events, SIGMA_DIR), (events,)):
            try:
                ctx = rule_context.build(*build_args)
                break
            except TypeError:
                continue
        if ctx is not None:
            return {e.id: _as_verdict(evaluate(e, ctx)) for e in events}

    # Batch function returning one result per unit, in input order.
    for batch_name in ("run_rules", "run_batch", "predict_all", "evaluate_all"):
        batch_fn = getattr(rules_engine, batch_name, None)
        if callable(batch_fn):
            results = list(batch_fn(events))
            if len(results) == len(events):
                return {e.id: _as_verdict(r) for e, r in zip(events, results)}

    # Per-event class instance or module-level function.
    engine_cls = getattr(rules_engine, "RulesEngine", None) or getattr(rules_engine, "Engine", None)
    entry = None
    if engine_cls is not None:
        for args, kwargs in (((), {}), ((SIGMA_DIR,), {}), ((), {"sigma_dir": SIGMA_DIR})):
            try:
                entry = _resolve_entry(engine_cls(*args, **kwargs))
                if entry is not None:
                    break
            except Exception:
                continue
    if entry is None:
        entry = _resolve_entry(rules_engine)
    if entry is None:
        raise RuntimeError(
            "rules.engine imported but no usable entry point found "
            "(looked for RuleContext+evaluate, run_rules, a RulesEngine class or a module function)"
        )
    return {e.id: _as_verdict(entry(e)) for e in events}


# --------------------------------------------------------------------------- #
# LLM arm (batch: one classify_many call, which persists the cache).
# --------------------------------------------------------------------------- #
def _build_llm_detector(prompt: str, provider: str, refresh: bool, from_cache: bool = False) -> Any:
    from detector import llm as llm_mod  # lazy: may not exist yet

    detector_cls = (
        getattr(llm_mod, "LLMDetector", None)
        or getattr(llm_mod, "LlmDetector", None)
        or getattr(llm_mod, "Detector", None)
    )
    if detector_cls is None:
        # Module-level factory fallback.
        for factory_name in ("build_detector", "make_detector", "build"):
            factory = getattr(llm_mod, factory_name, None)
            if callable(factory):
                detector_cls = factory
                break
    if detector_cls is None:
        raise RuntimeError("detector.llm imported but no LLMDetector class or factory was found")

    candidates = {
        "prompt_version": prompt,
        "prompt": prompt,
        "prompt_name": prompt,
        "version": prompt,
        "provider": provider,
        "backend": provider,
        "refresh": refresh,
        "refresh_cache": refresh,
        "force_refresh": refresh,
        "use_cache": True,  # always keep the cache layer on
        "require_cache": from_cache,  # --from-cache: never touch the model on a miss
        "cache_dir": str(CACHE_DIR),
    }
    kwargs = _select_kwargs(detector_cls, candidates)
    try:
        return detector_cls(**kwargs)
    except Exception as exc:
        raise RuntimeError(f"could not construct the LLM detector with {kwargs!r}: {exc}") from exc


def llm_verdicts(
    events: list[Event], prompt: str, provider: str, refresh: bool, from_cache: bool = False
) -> dict[str, Verdict]:
    """Classify ``events`` with the LLM detector and return ``id -> Verdict``.

    Uses the batch method (``classify_many`` / ``predict_many``) when available
    so the on-disk response cache is written once at the end of the run.
    """
    if not events:
        return {}
    detector = _build_llm_detector(prompt, provider, refresh, from_cache)

    for batch_name in ("classify_many", "predict_many"):
        batch = getattr(detector, batch_name, None)
        if callable(batch):
            results = list(batch(events))
            verdicts = {e.id: _as_verdict(r) for e, r in zip(events, results)}
            _maybe_save(detector)
            return verdicts

    entry = _resolve_entry(detector)
    if entry is None:
        raise RuntimeError("the LLM detector exposes no usable classify/predict method")
    verdicts = {e.id: _as_verdict(entry(e)) for e in events}
    _maybe_save(detector)
    return verdicts


def _maybe_save(detector: Any) -> None:
    save = getattr(detector, "save", None)
    if callable(save):
        try:
            save()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Hybrid arm (rules first; LLM only on the units the rules left benign).
# --------------------------------------------------------------------------- #
def hybrid_verdicts(
    units: list[LabeledUnit], prompt: str, provider: str, refresh: bool, from_cache: bool = False
) -> dict[str, Verdict]:
    rv = rules_verdicts(units)
    events = [u.event for u in units]
    to_ask = [e for e in events if not rv[e.id].is_incident]
    lv = llm_verdicts(to_ask, prompt, provider, refresh, from_cache) if to_ask else {}

    hybrid = HybridDetector(lambda e: rv[e.id], lambda e: lv[e.id])
    return {e.id: hybrid.judge(e) for e in events}


def predict_all(
    approach: str,
    units: list[LabeledUnit],
    prompt: str,
    provider: str,
    refresh: bool,
    from_cache: bool = False,
) -> dict[str, Verdict]:
    """Produce ``unit_id -> Verdict`` for one approach over the whole corpus."""
    if approach == "rules":
        return rules_verdicts(units)
    if approach == "llm":
        return llm_verdicts([u.event for u in units], prompt, provider, refresh, from_cache)
    if approach == "hybrid":
        return hybrid_verdicts(units, prompt, provider, refresh, from_cache)
    raise ValueError(f"unknown approach: {approach!r}")


# --------------------------------------------------------------------------- #
# Prediction files (the contract eval consumes) + a small progress metric.
# --------------------------------------------------------------------------- #
def write_predictions(pred_dir: Path, run_key: str, units: list[LabeledUnit], verdicts: dict[str, Verdict]) -> Path:
    """Write ``reports/predictions/<run_key>.jsonl`` as one Prediction per line."""
    pred_dir.mkdir(parents=True, exist_ok=True)
    path = pred_dir / f"{run_key}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for unit in units:
            verdict = verdicts.get(unit.unit_id)
            if verdict is None:
                prediction = Prediction(unit_id=unit.unit_id, is_incident=False, technique=None)
            else:
                prediction = Prediction.from_verdict(unit.unit_id, verdict)
            fh.write(json.dumps(prediction.model_dump(mode="json"), ensure_ascii=False) + "\n")
    return path


def _quick_metrics(units: list[LabeledUnit], verdicts: dict[str, Verdict]) -> tuple[float, float, float, int, int, int]:
    """Precision / recall / F1 and tp/fp/fn for the stderr progress line only."""
    tp = fp = fn = 0
    for unit in units:
        pred = bool(verdicts.get(unit.unit_id) and verdicts[unit.unit_id].is_incident)
        truth = unit.is_incident
        if pred and truth:
            tp += 1
        elif pred and not truth:
            fp += 1
        elif not pred and truth:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1, tp, fp, fn


# --------------------------------------------------------------------------- #
# eval integration (authoritative tables / reports / README).
# --------------------------------------------------------------------------- #
def _tabulate_and_report(
    produced: list[str], labels_path: Path, pred_dir: Path, reports_dir: Path
) -> bool:
    """Score the produced runs via eval, print the table and write reports/.

    Returns True when eval handled it; False (with a printed note) if the eval
    package is unavailable, in which case the caller prints a minimal fallback.
    """
    try:
        from eval import compare  # lazy
    except Exception as exc:
        print(f"(eval.compare unavailable: {exc})", file=sys.stderr)
        return False
    metrics = compare.summarize(labels_path, pred_dir, runs=produced)
    if not metrics:
        return False
    print("\nResults:")
    print(compare.readme_table_markdown(metrics))
    paths = compare.write_reports(metrics, reports_dir)
    print(f"\nWrote {paths['markdown']}, {paths['csv']}, {paths['json']}.")
    return True


def _render_readme(labels_path: Path, pred_dir: Path, reports_dir: Path) -> None:
    """Best-effort: regenerate the README table + error examples via eval."""
    try:
        from eval import render_readme  # lazy
    except Exception as exc:
        print(f"(eval.render_readme unavailable: {exc})", file=sys.stderr)
        return
    try:
        written = render_readme.render(
            labels_path=labels_path, pred_dir=pred_dir, reports_dir=reports_dir
        )
        print(f"README updated ({', '.join(written) if written else 'no markers found'}).", file=sys.stderr)
    except SystemExit as exc:
        print(f"(README not updated: {exc})", file=sys.stderr)
    except Exception as exc:
        print(f"(README render failed: {exc})", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Orchestration.
# --------------------------------------------------------------------------- #
def _run_key(approach: str, prompt: str) -> str:
    if approach == "rules":
        return "rules"
    if approach == "llm":
        return f"llm_{prompt}"
    return "hybrid"


def _plan(args: argparse.Namespace) -> list[tuple[str, str, Optional[str]]]:
    if args.all:
        return list(_ALL_RUNS)
    return [(_run_key(args.approach, args.prompt), args.approach, args.prompt)]


def run(args: argparse.Namespace) -> int:
    labels_path = Path(args.corpus)
    pred_dir = Path(args.pred_dir)
    reports_dir = Path(args.reports_dir)

    # Mock-provider wall. The mock backend is a deterministic offline heuristic,
    # not a model, so its output must never land in the committed reports/ where
    # a later render could pick it up and report it as real LLM results. When the
    # mock provider is used against the default repository paths, redirect every
    # artifact (prediction files AND the summary reports) to a scratch directory
    # under reports/generated/ so nothing reaches reports/predictions/ or
    # reports/summary.*. Pass explicit --pred-dir/--reports-dir to override.
    if args.provider == "mock" and pred_dir == PRED_DIR_DEFAULT and reports_dir == REPORTS_DIR_DEFAULT:
        scratch = ROOT / "reports" / "generated" / "mock"
        pred_dir = scratch / "predictions"
        reports_dir = scratch
        print(
            "(mock provider: redirecting predictions and reports to "
            f"{scratch}/ so mock output never reaches the committed reports/)",
            file=sys.stderr,
        )

    reports_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    units = load_corpus(labels_path)
    incidents = sum(1 for u in units if u.is_incident)
    print(
        f"Loaded {len(units)} units from {labels_path} "
        f"({incidents} incidents, {incidents / len(units):.1%}).",
        file=sys.stderr,
    )

    produced: list[str] = []
    failures: list[str] = []
    for run_key, approach, prompt in _plan(args):
        prompt = prompt or args.prompt
        print(
            f"\n=== {run_key}: approach={approach}"
            + (f" prompt={prompt} provider={args.provider}" if approach != "rules" else "")
            + " ===",
            file=sys.stderr,
        )
        try:
            verdicts = predict_all(approach, units, prompt, args.provider, args.refresh, args.from_cache)
        except Exception as exc:  # component missing, model unreachable, etc.
            msg = f"[skip] {run_key}: {exc}"
            print(msg, file=sys.stderr)
            failures.append(msg)
            continue

        write_predictions(pred_dir, run_key, units, verdicts)
        produced.append(run_key)
        p, r, f1, tp, fp, fn = _quick_metrics(units, verdicts)
        print(
            f"    {_DISPLAY.get(run_key, run_key)}: "
            f"P={p:.3f} R={r:.3f} F1={f1:.3f} (tp={tp} fp={fp} fn={fn}) "
            f"-> {pred_dir / (run_key + '.jsonl')}",
            file=sys.stderr,
        )

    if not produced:
        print("\nNo approach produced results. Components not available yet:", file=sys.stderr)
        for msg in failures:
            print(f"  {msg}", file=sys.stderr)
        return 1

    if not _tabulate_and_report(produced, labels_path, pred_dir, reports_dir):
        # Minimal fallback table if the eval package could not be used.
        print("\nResults (fallback, eval package unavailable):")
        for run_key in produced:
            path = pred_dir / f"{run_key}.jsonl"
            preds = {json.loads(l)["unit_id"]: json.loads(l) for l in path.read_text().splitlines() if l.strip()}
            vd = {uid: Verdict(is_incident=row["is_incident"], technique=row.get("technique"),
                               confidence=1.0, reason="") for uid, row in preds.items()}
            p, r, f1, tp, fp, fn = _quick_metrics(units, vd)
            print(f"  {_DISPLAY.get(run_key, run_key):<26} P={p:.3f} R={r:.3f} F1={f1:.3f} "
                  f"FP/1000={fp / len(units) * 1000:.1f}")

    if args.all or args.render_readme:
        default_dirs = pred_dir == PRED_DIR_DEFAULT and reports_dir == REPORTS_DIR_DEFAULT
        if args.provider == "mock":
            print(
                "(skipping README render: the mock provider is a test heuristic and its "
                "numbers must never be reported as LLM results)",
                file=sys.stderr,
            )
        elif not default_dirs:
            print(
                "(skipping README render: --pred-dir/--reports-dir were redirected, so the "
                "repository README is left untouched)",
                file=sys.stderr,
            )
        else:
            _render_readme(labels_path, pred_dir, reports_dir)

    print(f"\nPrediction files in {pred_dir}/ ; reports in {reports_dir}/.")
    if failures:
        print("\nSome approaches were skipped (component not ready / model unreachable):", file=sys.stderr)
        for msg in failures:
            print(f"  {msg}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m runner.run",
        description="Run rules / LLM / hybrid incident detection over the labeled corpus and score it.",
    )
    parser.add_argument(
        "--approach",
        choices=["rules", "llm", "hybrid"],
        help="Which detector to run. Omit and pass --all to run every approach.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run the four README rows (rules, LLM v1, LLM v3, hybrid) and refresh the README.",
    )
    parser.add_argument("--prompt", choices=["v1", "v2", "v3"], default="v1", help="LLM prompt version.")
    parser.add_argument("--provider", choices=["ollama", "mock"], default="ollama", help="LLM backend.")
    parser.add_argument(
        "--from-cache",
        dest="from_cache",
        action="store_true",
        help="Reproduce from the committed response cache (do not recompute). "
        "Offline as long as the cache is complete.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Recompute LLM answers and overwrite the response cache.",
    )
    parser.add_argument(
        "--render-readme",
        dest="render_readme",
        action="store_true",
        help="After the run, regenerate the README table + error examples via eval.render_readme.",
    )
    parser.add_argument("--corpus", default=str(CORPUS_DEFAULT), help="Path to labels.jsonl.")
    parser.add_argument("--pred-dir", dest="pred_dir", default=str(PRED_DIR_DEFAULT),
                        help="Directory for per-run prediction files.")
    parser.add_argument("--reports-dir", dest="reports_dir", default=str(REPORTS_DIR_DEFAULT),
                        help="Directory for the summary / error-analysis reports.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.all and not args.approach:
        parser.error("specify --approach {rules,llm,hybrid} or --all")
    if args.from_cache and args.refresh:
        parser.error("--from-cache and --refresh are mutually exclusive")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
