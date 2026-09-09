"""Detector package: shared schema plus normalization, rules and LLM detectors."""

from detector.schema import (
    Event,
    Label,
    LabeledUnit,
    Prediction,
    SourceType,
    Verdict,
)

__all__ = [
    "Event",
    "Verdict",
    "Prediction",
    "LabeledUnit",
    "Label",
    "SourceType",
]
