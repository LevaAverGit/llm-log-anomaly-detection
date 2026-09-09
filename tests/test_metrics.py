"""Metrics wiring: a toy set with a hand-computed answer verifies that the
sklearn-backed scoring produces the right precision / recall / F1 and a sane
false-positives-per-1000 figure.

The toy corpus (10 units) has a known confusion matrix for the incident class:

    TP = 3, FP = 1, FN = 2, TN = 4

    precision = TP / (TP + FP) = 3 / 4 = 0.75
    recall    = TP / (TP + FN) = 3 / 5 = 0.60
    f1        = 2 * P * R / (P + R)   = 0.6666...

Expected interface (``eval.metrics``): a scoring function that takes the
predictions and the ground-truth labeled units (joined by ``unit_id``) and
returns precision / recall / f1 / false-positives-per-1000. The exact function
name and argument order are resolved in ``tests/_support`` and confirmed here
against the known precision value.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _support as S  # noqa: E402

# Toy set: (unit_id, truth_is_incident, pred_is_incident)
_TOY = [
    ("u01", True, True),    # TP
    ("u02", True, True),    # TP
    ("u03", True, True),    # TP
    ("u04", True, False),   # FN
    ("u05", True, False),   # FN
    ("u06", False, True),   # FP
    ("u07", False, False),  # TN
    ("u08", False, False),  # TN
    ("u09", False, False),  # TN
    ("u10", False, False),  # TN
]

EXPECTED_PRECISION = 0.75
EXPECTED_RECALL = 0.60
EXPECTED_F1 = 2 * 0.75 * 0.60 / (0.75 + 0.60)
EXPECTED_FP = 1
TOTAL = len(_TOY)
BENIGN = sum(1 for _, truth, _ in _TOY if not truth)


@pytest.fixture(scope="module")
def metrics_mod():
    return S.import_module_or_fail("eval.metrics")


@pytest.fixture(scope="module")
def toy_labels():
    return [
        S.make_labeled_unit(uid, truth, technique="T1110" if truth else None)
        for uid, truth, _ in _TOY
    ]


@pytest.fixture(scope="module")
def toy_predictions():
    return [S.make_prediction(uid, pred) for uid, _, pred in _TOY]


@pytest.fixture(scope="module")
def scored(metrics_mod, toy_predictions, toy_labels):
    """Score the toy set, resolving the correct (predictions, labels) order.

    The scoring function's argument order is confirmed by requiring the known
    precision of 0.75; whichever order reproduces it is the correct one.
    """
    fn = S.resolve_score_fn(metrics_mod)
    observed = []
    for order in ((toy_predictions, toy_labels), (toy_labels, toy_predictions)):
        try:
            result = fn(*order)
        except TypeError:
            continue
        except Exception:  # noqa: BLE001 - try the other order before giving up
            continue
        try:
            precision = float(S.get_metric(result, S.PRECISION_NAMES))
        except Exception:  # noqa: BLE001
            continue
        observed.append(precision)
        if math.isclose(precision, EXPECTED_PRECISION, abs_tol=1e-6):
            return result
    raise AssertionError(
        f"scoring never reproduced the known precision {EXPECTED_PRECISION}; "
        f"observed precisions across argument orders: {observed}"
    )


def test_precision(scored):
    assert float(S.get_metric(scored, S.PRECISION_NAMES)) == pytest.approx(EXPECTED_PRECISION)


def test_recall(scored):
    assert float(S.get_metric(scored, S.RECALL_NAMES)) == pytest.approx(EXPECTED_RECALL)


def test_f1(scored):
    assert float(S.get_metric(scored, S.F1_NAMES)) == pytest.approx(EXPECTED_F1)


def test_false_positive_count_if_exposed(scored):
    fp = S.get_metric(scored, S.FP_COUNT_NAMES, required=False)
    if fp is not None:
        assert int(fp) == EXPECTED_FP


def test_false_positives_per_1000(scored):
    """FP/1000 must match one of the two standard denominators (all units or
    benign units): 1/10*1000 = 100, or 1/5*1000 = 200."""
    fp_per_1000 = S.get_metric(scored, S.FP_PER_1000_NAMES)
    value = float(fp_per_1000)
    per_total = EXPECTED_FP / TOTAL * 1000.0
    per_benign = EXPECTED_FP / BENIGN * 1000.0
    assert value == pytest.approx(per_total) or value == pytest.approx(per_benign), (
        f"fp_per_1000={value}; expected {per_total} (per unit) or {per_benign} (per benign)"
    )


def test_perfect_predictions_score_one(metrics_mod, toy_labels):
    """Sanity check on the wiring: predicting the truth exactly gives P=R=F1=1."""
    fn = S.resolve_score_fn(metrics_mod)
    perfect = [S.make_prediction(u.unit_id, u.is_incident) for u in toy_labels]
    for order in ((perfect, toy_labels), (toy_labels, perfect)):
        try:
            result = fn(*order)
        except (TypeError, Exception):  # noqa: BLE001
            continue
        try:
            precision = float(S.get_metric(result, S.PRECISION_NAMES))
            recall = float(S.get_metric(result, S.RECALL_NAMES))
            f1 = float(S.get_metric(result, S.F1_NAMES))
        except Exception:  # noqa: BLE001
            continue
        if math.isclose(precision, 1.0, abs_tol=1e-6):
            assert recall == pytest.approx(1.0)
            assert f1 == pytest.approx(1.0)
            return
    raise AssertionError("perfect predictions did not score precision 1.0 in either argument order")
