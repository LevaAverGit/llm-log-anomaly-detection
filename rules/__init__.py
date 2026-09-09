"""Deterministic Sigma-style rules baseline.

The public entry point is :func:`rules.engine.run_rules`, which turns a
collection of normalized :class:`detector.schema.Event` objects into one
:class:`detector.schema.Prediction` per unit without ever looking at the
ground-truth labels.
"""

from rules.engine import evaluate, load_sigma_rules, run_rules

__all__ = ["run_rules", "evaluate", "load_sigma_rules"]
