"""Scoring primitives: precision / recall / F1, confusion matrix, FP-per-1000.

Everything here is a pure function of (ground-truth labels, detector
predictions). It deliberately knows nothing about how the predictions were
produced (rules, LLM or hybrid) or how good they are, so the rules baseline and
the LLM detector are scored by *exactly* the same code path -- the only honest
way to compare them.

Prediction files: the contract with runner/run.py
--------------------------------------------------
The runner writes one JSONL file per scored run into ``reports/predictions/``::

    reports/predictions/<run_key>.jsonl

Every line is a :class:`detector.schema.Prediction` serialised with
``model_dump(mode="json")`` -- an object with ``unit_id``, ``is_incident`` and
``technique``. ``<run_key>`` names the run and matches
``eval.compare.RUN_REGISTRY`` (``rules``, ``llm_v1``, ``llm_v3``, ``hybrid``,
...). The join key between a prediction and its ground-truth unit is
``unit_id``.

A labeled unit that has no matching prediction is scored as "not flagged"
(predicted benign) and counted in :attr:`RunMetrics.n_missing`, so an incomplete
run shows up as a visible gap instead of silently distorting the numbers. Pass
``strict=True`` to :func:`score` to turn a missing prediction into an error
instead.

Positive class
--------------
The positive class is ``incident`` (``is_incident is True``). Precision, recall
and F1 are reported for that class; false-positives-per-1000 is the number of
benign units wrongly flagged, scaled to 1000 units so runs of different sizes
stay comparable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Optional

from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from detector.schema import LabeledUnit, Prediction

# --------------------------------------------------------------------------- #
# Canonical locations (relative to the repository root).
# --------------------------------------------------------------------------- #
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LABELS_PATH = _REPO_ROOT / "corpus" / "labels.jsonl"
DEFAULT_PREDICTIONS_DIR = _REPO_ROOT / "reports" / "predictions"
DEFAULT_REPORTS_DIR = _REPO_ROOT / "reports"


# --------------------------------------------------------------------------- #
# Loaders.
# --------------------------------------------------------------------------- #
def _iter_jsonl(path: Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_labels(path: Path | str = DEFAULT_LABELS_PATH) -> list[LabeledUnit]:
    """Load and validate the ground-truth corpus (``corpus/labels.jsonl``)."""
    return [LabeledUnit.model_validate(row) for row in _iter_jsonl(Path(path))]


def load_predictions(path: Path | str) -> list[Prediction]:
    """Load and validate one run's predictions (``reports/predictions/<run>.jsonl``)."""
    return [Prediction.model_validate(row) for row in _iter_jsonl(Path(path))]


def predictions_path(run_key: str, pred_dir: Path | str = DEFAULT_PREDICTIONS_DIR) -> Path:
    """Return the conventional path of a run's prediction file."""
    return Path(pred_dir) / f"{run_key}.jsonl"


# --------------------------------------------------------------------------- #
# The result of scoring one run.
# --------------------------------------------------------------------------- #
class RunMetrics(BaseModel):
    """All the numbers for one scored run, ready to tabulate or serialise."""

    model_config = ConfigDict(extra="forbid")

    run: str = Field(..., description="Run key, e.g. 'rules' or 'llm_v3'.")
    display_name: str = Field(..., description="Human-readable name for the table.")

    n_units: int = Field(..., description="Units scored (denominator).")
    n_incidents: int = Field(..., description="Ground-truth incidents among them.")
    n_benign: int = Field(..., description="Ground-truth benign among them.")
    n_predicted: int = Field(..., description="Units that had a prediction.")
    n_missing: int = Field(..., description="Labeled units with no prediction (scored benign).")

    tp: int = Field(..., description="Incidents correctly flagged.")
    fp: int = Field(..., description="Benign units wrongly flagged.")
    fn: int = Field(..., description="Incidents missed.")
    tn: int = Field(..., description="Benign units correctly left alone.")

    precision: float = Field(..., description="TP / (TP + FP) for the incident class.")
    recall: float = Field(..., description="TP / (TP + FN) for the incident class.")
    f1: float = Field(..., description="Harmonic mean of precision and recall.")
    fp_per_1000: float = Field(..., description="False positives scaled to 1000 units.")

    technique_evaluable: int = Field(
        ..., description="True positives whose ground-truth technique is known."
    )
    technique_correct: int = Field(
        ..., description="Of those, how many got the ATT&CK technique right."
    )
    technique_accuracy: Optional[float] = Field(
        None, description="technique_correct / technique_evaluable, or None if 0."
    )

    def confusion_matrix(self) -> list[list[int]]:
        """Return ``[[TN, FP], [FN, TP]]`` (sklearn row/column order)."""
        return [[self.tn, self.fp], [self.fn, self.tp]]

    def format_confusion(self) -> str:
        """A compact text rendering of the confusion matrix."""
        return (
            "                 pred benign  pred incident\n"
            f"  true benign    {self.tn:>11}  {self.fp:>13}\n"
            f"  true incident  {self.fn:>11}  {self.tp:>13}"
        )

    def table_row(self) -> dict[str, object]:
        """One row for the README/compare table (the four headline columns)."""
        return {
            "Approach": self.display_name,
            "Precision": round(self.precision, 3),
            "Recall": round(self.recall, 3),
            "F1": round(self.f1, 3),
            "FP/1000": round(self.fp_per_1000, 1),
        }


# --------------------------------------------------------------------------- #
# Scoring.
# --------------------------------------------------------------------------- #
def score(
    labels: Iterable[LabeledUnit],
    predictions: Iterable[Prediction],
    run: str = "run",
    display_name: str = "",
    strict: bool = False,
) -> RunMetrics:
    """Score one run's predictions against the ground-truth labels.

    Units are joined by ``unit_id``. A labeled unit with no prediction is scored
    as predicted-benign and counted in ``n_missing`` (or raises when
    ``strict=True``). Predictions for unknown unit ids are ignored.
    """
    labels = list(labels)
    pred_by_id: dict[str, Prediction] = {p.unit_id: p for p in predictions}

    y_true: list[bool] = []
    y_pred: list[bool] = []
    n_missing = 0
    technique_evaluable = 0
    technique_correct = 0

    for unit in labels:
        truth = unit.is_incident
        pred = pred_by_id.get(unit.unit_id)
        if pred is None:
            if strict:
                raise KeyError(f"no prediction for unit {unit.unit_id!r} in run {run!r}")
            predicted = False
            n_missing += 1
        else:
            predicted = pred.is_incident

        y_true.append(truth)
        y_pred.append(predicted)

        # Technique accuracy is only meaningful on correctly-caught incidents.
        if truth and predicted and unit.technique is not None:
            technique_evaluable += 1
            if pred is not None and pred.technique == unit.technique:
                technique_correct += 1

    if not y_true:
        raise ValueError("cannot score an empty label set")

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[True],
        average="binary",
        pos_label=True,
        zero_division=0.0,
    )
    tn, fp, fn, tp = (int(x) for x in confusion_matrix(
        y_true, y_pred, labels=[False, True]
    ).ravel())

    n_units = len(y_true)
    n_incidents = sum(y_true)
    technique_accuracy = (
        technique_correct / technique_evaluable if technique_evaluable else None
    )

    return RunMetrics(
        run=run,
        display_name=display_name or run,
        n_units=n_units,
        n_incidents=n_incidents,
        n_benign=n_units - n_incidents,
        n_predicted=n_units - n_missing,
        n_missing=n_missing,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        fp_per_1000=(fp / n_units * 1000.0) if n_units else 0.0,
        technique_evaluable=technique_evaluable,
        technique_correct=technique_correct,
        technique_accuracy=technique_accuracy,
    )


def score_file(
    run_key: str,
    display_name: str = "",
    labels_path: Path | str = DEFAULT_LABELS_PATH,
    pred_dir: Path | str = DEFAULT_PREDICTIONS_DIR,
    strict: bool = False,
) -> RunMetrics:
    """Convenience: load the label file and one prediction file, then score."""
    labels = load_labels(labels_path)
    predictions = load_predictions(predictions_path(run_key, pred_dir))
    return score(labels, predictions, run=run_key, display_name=display_name, strict=strict)


def main() -> None:
    """Score every prediction file found under ``reports/predictions/`` and print it."""
    import argparse

    parser = argparse.ArgumentParser(description="Score detector predictions against the corpus.")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS_PATH))
    parser.add_argument("--pred-dir", default=str(DEFAULT_PREDICTIONS_DIR))
    parser.add_argument("--run", default=None, help="Score only this run key (default: all found).")
    args = parser.parse_args()

    labels = load_labels(args.labels)
    pred_dir = Path(args.pred_dir)
    if args.run:
        run_keys = [args.run]
    else:
        run_keys = sorted(p.stem for p in pred_dir.glob("*.jsonl"))

    if not run_keys:
        print(f"No prediction files found in {pred_dir}. Run the detector first (e.g. `make run`).")
        return

    for run_key in run_keys:
        path = predictions_path(run_key, pred_dir)
        if not path.exists():
            print(f"[skip] {run_key}: {path} not found")
            continue
        metrics = score(labels, load_predictions(path), run=run_key, display_name=run_key)
        print(
            f"{run_key:>10}  P={metrics.precision:.3f}  R={metrics.recall:.3f}  "
            f"F1={metrics.f1:.3f}  FP/1000={metrics.fp_per_1000:.1f}  "
            f"(TP={metrics.tp} FP={metrics.fp} FN={metrics.fn} TN={metrics.tn}"
            + (f", {metrics.n_missing} missing" if metrics.n_missing else "")
            + ")"
        )


if __name__ == "__main__":
    main()
