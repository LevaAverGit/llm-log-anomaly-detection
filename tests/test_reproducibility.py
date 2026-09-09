"""Reproducibility: a run must yield identical metrics when repeated.

The headline table in the README is meant to reproduce from a committed cache
without touching the network. These tests use the offline ``mock`` provider (no
Ollama) to assert the property that makes that possible: the detector is
deterministic over the same inputs, so two independent runs produce byte-for-
byte identical predictions and therefore identical metrics.

The mock is only used to exercise the run/score wiring deterministically; it
does not stand in for real detection quality.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _support as S  # noqa: E402
from detector.schema import Prediction  # noqa: E402


@pytest.fixture(scope="module")
def units():
    loaded = S.load_labeled_units()
    assert loaded, "corpus/labels.jsonl is empty"
    return loaded


@pytest.fixture(scope="module")
def llm_mod():
    return S.import_module_or_fail("detector.llm")


def _run_mock(llm_mod, units) -> list[Prediction]:
    """One full pass of the mock detector over the corpus -> predictions."""
    detector = S.make_mock_detector(llm_mod)
    predictions: list[Prediction] = []
    for unit in units:
        verdict = S.classify_event(detector, unit.event)
        predictions.append(Prediction.from_verdict(unit.event.id, verdict))
    return predictions


def _pred_key(preds):
    return [(p.unit_id, p.is_incident, p.technique) for p in preds]


def test_mock_detector_is_deterministic(llm_mod, units):
    """Two independent mock runs over the corpus produce identical predictions."""
    run_a = _run_mock(llm_mod, units)
    run_b = _run_mock(llm_mod, units)

    assert len(run_a) == len(units) == len(run_b)
    assert _pred_key(run_a) == _pred_key(run_b), "mock detector output is not reproducible"


def test_cached_run_yields_identical_metrics(llm_mod, units):
    """The scored metrics of two identical runs must be identical."""
    metrics_mod = S.import_module_or_fail("eval.metrics")
    score = S.resolve_score_fn(metrics_mod)

    run_a = _run_mock(llm_mod, units)
    run_b = _run_mock(llm_mod, units)

    result_a = S.score_run(score, run_a, units)
    result_b = S.score_run(score, run_b, units)

    for names in (S.PRECISION_NAMES, S.RECALL_NAMES, S.F1_NAMES):
        a = float(S.get_metric(result_a, names))
        b = float(S.get_metric(result_b, names))
        assert a == pytest.approx(b, rel=0, abs=0.0), (
            f"metric {names[0]} differed between identical runs: {a} != {b}"
        )

    fp_a = S.get_metric(result_a, S.FP_PER_1000_NAMES, required=False)
    fp_b = S.get_metric(result_b, S.FP_PER_1000_NAMES, required=False)
    if fp_a is not None and fp_b is not None:
        assert float(fp_a) == float(fp_b)


def test_metrics_are_a_pure_function_of_predictions(units):
    """Scoring the very same predictions twice is stable (no hidden state)."""
    metrics_mod = S.import_module_or_fail("eval.metrics")
    score = S.resolve_score_fn(metrics_mod)

    # A fixed, detector-independent prediction set: mark every unit benign.
    preds = [S.make_prediction(u.unit_id, False) for u in units]

    first = S.score_run(score, preds, units)
    second = S.score_run(score, preds, units)

    for names in (S.PRECISION_NAMES, S.RECALL_NAMES, S.F1_NAMES):
        a = S.get_metric(first, names, required=False)
        b = S.get_metric(second, names, required=False)
        if a is not None and b is not None:
            assert float(a) == float(b)
