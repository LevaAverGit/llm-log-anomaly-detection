"""Evaluation layer: scoring, cross-approach comparison, error analysis and README rendering.

The four modules here turn detector predictions into the numbers and examples a
reviewer sees:

- ``metrics``       scikit-learn precision / recall / F1, confusion matrix and
                    false-positives-per-1000 for a single run.
- ``compare``       a pandas summary across approaches and prompt versions.
- ``errors``        auto-selected 3 + 3 disagreement examples between the rules
                    baseline and the LLM detector, each with its raw log.
- ``render_readme`` writes the results table and the error examples into
                    README.md between explicit markers.

All of them read the ground truth from ``corpus/labels.jsonl`` and the detector
outputs from ``reports/predictions/<run>.jsonl`` (the contract documented in
``metrics``), so nothing here depends on how a prediction was produced.
"""

from eval.metrics import RunMetrics, load_labels, load_predictions, score

__all__ = ["RunMetrics", "load_labels", "load_predictions", "score"]
