"""Shared data contracts for the rules-vs-LLM detection comparison.

Every component in this repository conforms to the models defined here:

- ``normalize.py`` turns the four raw log formats into :class:`Event` objects.
- ``rules/engine.py`` and ``detector/llm.py`` both consume :class:`Event`
  objects and return a :class:`Verdict`.
- The runner records a :class:`Prediction` per analysis unit, which the eval
  code compares against the ground-truth labels stored in
  ``corpus/labels.jsonl`` (each row is a :class:`LabeledUnit`).

Keeping these three models (Event, Verdict, Prediction) stable is what lets the
rules baseline and the LLM detector be scored on exactly the same units.

The design rule is deliberate: an *analysis unit* is not always a single log
line. A brute-force burst or a scan is a coherent activity that only makes
sense as one unit (a count-based rule and an LLM both need the whole burst to
judge it), so those are aggregated into a single :class:`Event` whose ``raw``
field carries the count, the time window and representative sample lines.
Discrete events (one login, one exploit request, one cloud action) stay as one
unit each.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# ATT&CK technique id, e.g. ``T1110`` or ``T1110.001``.
_TECHNIQUE_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$")


class SourceType(str, Enum):
    """The four log sources the corpus is built from."""

    linux_auth = "linux_auth"
    nginx_access = "nginx_access"
    windows_security = "windows_security"
    cloud_audit = "cloud_audit"


class Label(str, Enum):
    """Ground-truth label for a unit of analysis."""

    incident = "incident"
    benign = "benign"


def _clean_technique(value: Optional[str]) -> Optional[str]:
    """Normalise a technique string: empty / ``none`` / ``n/a`` become ``None``."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "n/a", "na", "-"}:
        return None
    return text.upper()


class Event(BaseModel):
    """A single normalised unit of analysis.

    One :class:`Event` is one thing that gets a verdict. For point activity it
    maps to one raw log line; for a brute-force burst or a scan it represents
    the whole aggregated activity, with the detail preserved under ``raw``.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Stable unit identifier, e.g. 'lx-0007'.")
    source: SourceType = Field(..., description="Which of the four log sources this came from.")
    timestamp: datetime = Field(..., description="Event time (aggregates use the window start).")
    actor: Optional[str] = Field(
        None,
        description="The acting entity: source IP or username. None when unknown.",
    )
    action: str = Field(..., description="Normalised action, e.g. 'ssh_failed_password'.")
    target: Optional[str] = Field(
        None,
        description="What was acted on: target user, request path, host or resource.",
    )
    status: Optional[str] = Field(
        None,
        description="Outcome, e.g. 'success', 'failure' or an HTTP status code.",
    )
    raw: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Original fields plus, for aggregated units, keys such as 'count', "
            "'window_end', 'distinct_targets' and 'sample_lines'."
        ),
    )

    @field_validator("action")
    @classmethod
    def _action_not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("action must be a non-empty string")
        return value


class Verdict(BaseModel):
    """A detector's judgement about one :class:`Event`.

    Returned by both the rules engine and the LLM detector so the two can be
    compared with identical downstream code.
    """

    model_config = ConfigDict(extra="forbid")

    is_incident: bool = Field(..., description="True if the detector calls this an incident.")
    technique: Optional[str] = Field(
        None, description="MITRE ATT&CK technique id (e.g. 'T1110.001') or None."
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detector confidence in [0, 1]."
    )
    reason: str = Field(..., description="Short human-readable justification.")

    @field_validator("technique", mode="before")
    @classmethod
    def _normalise_technique(cls, value: Optional[str]) -> Optional[str]:
        cleaned = _clean_technique(value)
        if cleaned is not None and not _TECHNIQUE_RE.match(cleaned):
            # Keep the raw string rather than crashing on a malformed LLM answer;
            # downstream scoring treats it as "some technique was named".
            return cleaned
        return cleaned


class Prediction(BaseModel):
    """One detector output for one unit, ready to be scored against the labels.

    This is the minimal projection of a :class:`Verdict` that the metrics code
    needs, joined to the unit it belongs to by ``unit_id``.
    """

    model_config = ConfigDict(extra="forbid")

    unit_id: str = Field(..., description="Matches Event.id and LabeledUnit.unit_id.")
    is_incident: bool = Field(..., description="Predicted incident / benign.")
    technique: Optional[str] = Field(
        None, description="Predicted MITRE ATT&CK technique id or None."
    )

    @field_validator("technique", mode="before")
    @classmethod
    def _normalise_technique(cls, value: Optional[str]) -> Optional[str]:
        return _clean_technique(value)

    @classmethod
    def from_verdict(cls, unit_id: str, verdict: Verdict) -> "Prediction":
        """Build a :class:`Prediction` from a :class:`Verdict`."""
        return cls(
            unit_id=unit_id,
            is_incident=verdict.is_incident,
            technique=verdict.technique,
        )


class LabeledUnit(BaseModel):
    """One row of ``corpus/labels.jsonl``: a unit plus its ground-truth label.

    The label is assigned from knowledge of the injected attack scenarios and is
    independent of the detection rules being evaluated (see ``corpus/README.md``).
    """

    model_config = ConfigDict(extra="forbid")

    unit_id: str = Field(..., description="Stable id, equal to the embedded Event.id.")
    event: Event = Field(..., description="The normalised unit of analysis.")
    label: Label = Field(..., description="Ground truth: incident or benign.")
    technique: Optional[str] = Field(
        None, description="Ground-truth MITRE ATT&CK technique for incidents; None for benign."
    )
    rationale: str = Field(
        ..., description="Why this label was assigned, tied to the labeling criterion."
    )

    @property
    def is_incident(self) -> bool:
        return self.label is Label.incident

    @field_validator("technique", mode="before")
    @classmethod
    def _normalise_technique(cls, value: Optional[str]) -> Optional[str]:
        return _clean_technique(value)
