"""Shared helpers for the test suite.

The four builders of this repository work in parallel, so the components under
test (``detector.normalize``, ``detector.llm`` and ``eval.metrics``) are pinned
by the *data contracts* in ``detector.schema`` and by the behaviour documented
in ``PLAN.md`` / ``corpus/README.md`` — not by frozen function names. To stay
robust to small naming differences between components while still testing real
behaviour, the resolvers below look up the entry points by a short list of
canonical names and adapt to the argument shape. If a component is missing or
exposes nothing recognisable, the resolver fails loudly with the list of names
it tried, rather than silently skipping.

Nothing here fabricates results: the mock LLM provider is only ever used so the
logic (parsing, determinism, metric wiring) can be exercised without Ollama.
"""

from __future__ import annotations

import importlib
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

# Make the repository root importable regardless of pytest's rootdir handling.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from detector.schema import (  # noqa: E402  (path set up above)
    Event,
    Label,
    LabeledUnit,
    Prediction,
    SourceType,
    Verdict,
)

RAW_DIR = REPO_ROOT / "corpus" / "raw"
LABELS_PATH = REPO_ROOT / "corpus" / "labels.jsonl"

_RAW_FILE = {
    SourceType.linux_auth: "linux_auth.log",
    SourceType.nginx_access: "nginx_access.log",
    SourceType.windows_security: "windows_security.jsonl",
    SourceType.cloud_audit: "cloud_audit.jsonl",
}


# --------------------------------------------------------------------------- #
# Corpus loading.
# --------------------------------------------------------------------------- #
def load_labeled_units() -> list[LabeledUnit]:
    """Load and validate every row of ``corpus/labels.jsonl``."""
    units: list[LabeledUnit] = []
    with LABELS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            units.append(LabeledUnit.model_validate(json.loads(line)))
    return units


def raw_path(source: SourceType) -> Path:
    return RAW_DIR / _RAW_FILE[source]


# --------------------------------------------------------------------------- #
# Generic import / attribute resolution.
# --------------------------------------------------------------------------- #
def import_module_or_fail(name: str):
    try:
        return importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - surface any import problem clearly
        raise AssertionError(
            f"Could not import '{name}'. Either the component is not built yet or "
            f"it fails to import: {exc!r}"
        ) from exc


def _attrs(mod) -> list[str]:
    return sorted(a for a in dir(mod) if not a.startswith("_"))


def resolve_callable(mod, names: Iterable[str]):
    """Return the first (callable, name) among ``names`` present on ``mod``."""
    for name in names:
        obj = getattr(mod, name, None)
        if callable(obj):
            return obj, name
    return None, None


# --------------------------------------------------------------------------- #
# Coercion to the shared contracts.
# --------------------------------------------------------------------------- #
def as_event(obj: Any) -> Event:
    if isinstance(obj, Event):
        return obj
    if isinstance(obj, dict):
        return Event.model_validate(obj)
    if hasattr(obj, "model_dump"):
        return Event.model_validate(obj.model_dump())
    raise AssertionError(f"Expected an Event / dict, got {type(obj)!r}: {obj!r}")


def as_verdict(obj: Any) -> Verdict:
    if isinstance(obj, Verdict):
        return obj
    if isinstance(obj, dict):
        return Verdict.model_validate(obj)
    if hasattr(obj, "model_dump"):
        return Verdict.model_validate(obj.model_dump())
    data = {}
    for field in ("is_incident", "technique", "confidence", "reason"):
        if hasattr(obj, field):
            data[field] = getattr(obj, field)
    if data:
        return Verdict.model_validate(data)
    raise AssertionError(f"Expected a Verdict / dict, got {type(obj)!r}: {obj!r}")


# --------------------------------------------------------------------------- #
# detector.normalize entry point.
# --------------------------------------------------------------------------- #
_NORMALIZE_DISPATCH = [
    "normalize_file",
    "normalize",
    "normalize_source",
    "normalize_log",
    "normalize_path",
    "normalize_raw",
    "normalize_events",
    "events_from_file",
    "to_events",
    "load_events",
    "parse_file",
]

_PATH_PARAM_NAMES = {
    "path", "file", "filepath", "file_path", "fname", "filename",
    "log_path", "logpath", "raw_path", "input_path", "source_path",
    "logfile", "log_file",
}
_TEXT_PARAM_NAMES = {
    "text", "content", "raw_text", "rawtext", "body", "data", "lines",
    "log_text", "logtext", "string", "raw", "log", "input",
}
_SOURCE_PARAM_NAMES = {
    "source", "src", "kind", "type", "source_type", "sourcetype", "log_type",
    "logtype", "fmt", "format", "stype",
}


def _invoke_normalize(fn, path: Path, text: str, source: SourceType) -> Optional[list]:
    """Call ``fn`` supplying a path, the file text and the source by param name.

    Handles the two common entry shapes — ``normalize_file(path, source)`` and
    ``normalize_source(source, text)`` / ``parse_<source>(text)`` — plus a
    positional fallback. Only ``TypeError`` is swallowed (a signature mismatch);
    anything else propagates so real bugs are not hidden.
    """
    try:
        params = [
            p for p in inspect.signature(fn).parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
    except (TypeError, ValueError):
        params = []

    roles: dict[str, str] = {}
    unmatched: list = []
    for p in params:
        low = p.name.lower()
        if low in _PATH_PARAM_NAMES:
            roles[p.name] = "path"
        elif low in _TEXT_PARAM_NAMES:
            roles[p.name] = "text"
        elif low in _SOURCE_PARAM_NAMES:
            roles[p.name] = "source"
        else:
            unmatched.append(p)

    required_unmatched = [p for p in unmatched if p.default is inspect.Parameter.empty]

    for source_arg in (source, source.value):
        base: dict[str, Any] = {}
        for p in params:
            role = roles.get(p.name)
            if role == "path":
                base[p.name] = path
            elif role == "text":
                base[p.name] = text
            elif role == "source":
                base[p.name] = source_arg

        # Fill any required-but-unnamed params with plausible values, trying a
        # few orderings (path-first, then text-first).
        fill_variants = [[path, source_arg, text], [text, source_arg, path]] if required_unmatched else [[]]
        for fillers in fill_variants:
            kwargs = dict(base)
            i = 0
            for p in required_unmatched:
                if i < len(fillers):
                    kwargs[p.name] = fillers[i]
                    i += 1
            try:
                result = fn(**kwargs)
            except TypeError:
                continue
            return list(result)
    return None


def run_normalize(mod, source: SourceType, path: Path) -> list[Event]:
    """Normalize one raw log file into a list of :class:`Event`.

    Tries a per-source function first (e.g. ``parse_linux_auth``), then a
    generic dispatcher, accepting the first that yields a non-empty list whose
    events all carry the expected source.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    candidates = [
        f"normalize_{source.value}", f"parse_{source.value}", f"{source.value}_events"
    ] + _NORMALIZE_DISPATCH

    tried: list[str] = []
    for name in candidates:
        fn = getattr(mod, name, None)
        if not callable(fn):
            continue
        tried.append(name)
        result = _invoke_normalize(fn, path, text, source)
        if not result:
            continue
        events = [as_event(x) for x in result]
        if events and all(ev.source == source for ev in events):
            return events

    raise AssertionError(
        "detector.normalize produced no usable events for "
        f"{source.value}. Tried {tried or 'nothing callable'}. "
        f"Available: {_attrs(mod)}"
    )


# --------------------------------------------------------------------------- #
# detector.llm — mock detector + response parser.
# --------------------------------------------------------------------------- #
_DETECTOR_FACTORIES = [
    "make_detector", "build_detector", "get_detector", "create_detector",
    "make_llm", "build_llm", "llm_detector", "load_detector",
]
_DETECTOR_CLASSES = [
    "LLMDetector", "LlmDetector", "LLMClassifier", "LlmClassifier",
    "OllamaDetector", "LLMJudge", "Detector", "MockDetector",
]
_CLASSIFY_METHODS = [
    "classify", "detect", "judge", "predict", "analyze", "verdict",
    "classify_event", "run", "__call__",
]
_PARSE_NAMES = [
    "parse_verdict", "parse_llm_response", "parse_response",
    "verdict_from_text", "verdict_from_response", "parse_verdict_json",
    "response_to_verdict", "to_verdict", "parse_output", "parse_model_output",
    "extract_verdict", "parse",
]

# Only mock providers are ever requested, so no network / Ollama is touched.
_MOCK_CTOR_ATTEMPTS = [
    ((), {"provider": "mock"}),
    ((), {"provider": "mock", "prompt": "v1"}),
    ((), {"provider": "mock", "prompt_version": "v1"}),
    ((), {"provider": "mock", "prompt": "v1", "model": "mock"}),
    (("mock",), {}),
    (("mock",), {"prompt": "v1"}),
]


def _try_construct(ctor):
    for args, kwargs in _MOCK_CTOR_ATTEMPTS:
        try:
            return ctor(*args, **kwargs)
        except TypeError:
            continue
        except Exception:  # noqa: BLE001 - a mock ctor should not need anything else
            continue
    return None


def make_mock_detector(mod):
    """Construct the LLM detector wired to the offline mock provider."""
    for name in _DETECTOR_FACTORIES:
        fn = getattr(mod, name, None)
        if callable(fn):
            inst = _try_construct(fn)
            if inst is not None:
                return inst
    for name in _DETECTOR_CLASSES:
        cls = getattr(mod, name, None)
        if isinstance(cls, type):
            inst = _try_construct(cls)
            if inst is not None:
                return inst
    raise AssertionError(
        "detector.llm exposes no mock-constructible detector. Tried factories "
        f"{_DETECTOR_FACTORIES} and classes {_DETECTOR_CLASSES}. "
        f"Available: {_attrs(mod)}"
    )


def classify_event(detector, event: Event) -> Verdict:
    """Ask a resolved detector for a verdict on one event."""
    for name in _CLASSIFY_METHODS:
        method = getattr(detector, name, None)
        if callable(method):
            try:
                result = method(event)
            except TypeError:
                continue
            return as_verdict(result)
    if callable(detector):
        return as_verdict(detector(event))
    raise AssertionError(
        f"Detector {type(detector)!r} has no recognised classify method "
        f"(tried {_CLASSIFY_METHODS})."
    )


def resolve_parser(mod):
    """Return the LLM-response parser function (raw text -> Verdict)."""
    fn, name = resolve_callable(mod, _PARSE_NAMES)
    if fn is None:
        raise AssertionError(
            "detector.llm exposes no response parser. Tried "
            f"{_PARSE_NAMES}. Available: {_attrs(mod)}"
        )
    return fn


# --------------------------------------------------------------------------- #
# eval.metrics — scoring function + metric accessors.
# --------------------------------------------------------------------------- #
_SCORE_NAMES = [
    "score", "compute_metrics", "evaluate", "score_predictions",
    "evaluate_predictions", "run_metrics", "metrics", "compute", "score_all",
]

PRECISION_NAMES = ["precision", "precision_score", "prec"]
RECALL_NAMES = ["recall", "recall_score", "rec", "sensitivity", "tpr"]
F1_NAMES = ["f1", "f1_score", "f_score", "fscore", "f"]
FP_COUNT_NAMES = ["false_positives", "fp", "n_fp", "num_false_positives", "fp_count"]
FP_PER_1000_NAMES = [
    "fp_per_1000", "false_positives_per_1000", "fp_per_1k", "fpper1000",
    "fp_rate_per_1000", "false_positive_per_1000", "fp_per_thousand",
]


def resolve_score_fn(mod):
    fn, _ = resolve_callable(mod, _SCORE_NAMES)
    if fn is None:
        raise AssertionError(
            "eval.metrics exposes no scoring function. Tried "
            f"{_SCORE_NAMES}. Available: {_attrs(mod)}"
        )
    return fn


def score_run(fn, predictions, labels):
    """Score with whichever argument order the function accepts.

    The scoring seam joins predictions to labels by ``unit_id``; the argument
    order is not part of the shared contract, so try labels-first (the common
    convention) and fall back to predictions-first. Used where only the
    resulting metric values matter (e.g. determinism checks).
    """
    predictions = list(predictions)
    labels = list(labels)
    for args in ((labels, predictions), (predictions, labels)):
        try:
            return fn(*args)
        except TypeError:
            continue
    raise AssertionError("scoring function rejected both argument orders")


def _metric_containers(result: Any):
    """Yield the result itself and a few likely nested per-class containers."""
    yield result
    for key in ("incident", "positive", "pos", "attack", "1", 1, True):
        sub = None
        try:
            sub = result[key]
        except Exception:  # noqa: BLE001
            sub = getattr(result, str(key), None)
        if sub is not None:
            yield sub


def get_metric(result: Any, names: Iterable[str], required: bool = True):
    """Read a metric value out of a dict / Series / object result."""
    names = list(names)
    for container in _metric_containers(result):
        for name in names:
            try:
                if name in container:  # dict / pandas.Series
                    return container[name]
            except Exception:  # noqa: BLE001 - container may not support `in`
                pass
            value = getattr(container, name, None)
            if value is not None:
                return value
    if required:
        raise AssertionError(
            f"None of {names} found in metrics result of type "
            f"{type(result)!r}: {result!r}"
        )
    return None


# --------------------------------------------------------------------------- #
# Toy fixtures for the metrics wiring test.
# --------------------------------------------------------------------------- #
def make_event(unit_id: str, source: SourceType = SourceType.linux_auth) -> Event:
    return Event(
        id=unit_id,
        source=source,
        timestamp=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        actor="10.0.0.1",
        action="ssh_failed_password",
        target="root",
        status="failure",
        raw={},
    )


def make_labeled_unit(unit_id: str, is_incident: bool, technique: Optional[str] = None) -> LabeledUnit:
    return LabeledUnit(
        unit_id=unit_id,
        event=make_event(unit_id),
        label=Label.incident if is_incident else Label.benign,
        technique=technique if is_incident else None,
        rationale="toy fixture for the metrics wiring test",
    )


def make_prediction(unit_id: str, is_incident: bool, technique: Optional[str] = None) -> Prediction:
    return Prediction(unit_id=unit_id, is_incident=is_incident, technique=technique)
