"""Hybrid detector: rules first, LLM only on what the rules do not catch.

The policy is intentionally simple and asymmetric:

1. Run the rules engine on the unit. If it fires (calls the unit an incident),
   trust it and return that verdict immediately. Rules are cheap, deterministic
   and high-precision on the patterns they encode, so there is no reason to
   second-guess a positive with a slower, noisier model.
2. Only when the rules stay silent (benign) do we pay for an LLM call and let
   the model decide.

Because a rules-positive is never overturned, the hybrid can only *add*
detections on top of the rules baseline: its recall is at least the rules'
recall, and any extra false positives it introduces come solely from the LLM
arm acting on units the rules left alone. That is exactly the trade the
comparison in this repository is meant to measure.

The class is deliberately decoupled from the concrete rules engine and LLM
detector: it is constructed from two callables, each ``Event -> Verdict``. The
runner wires the real components together; tests pass in stubs. Nothing here
imports :mod:`rules.engine` or :mod:`detector.llm`, so this module stays
trivially testable and free of heavy dependencies.
"""

from __future__ import annotations

from typing import Callable

from detector.schema import Event, Prediction, Verdict

# A detector arm: given one normalised unit, return a verdict for it.
VerdictFn = Callable[[Event], Verdict]

# Marks in a verdict's reason so downstream error analysis can tell which arm
# produced a hybrid decision without changing the Verdict schema.
_RULES_TAG = "[hybrid:rules]"
_LLM_TAG = "[hybrid:llm]"


class HybridDetector:
    """Combine a rules engine and an LLM detector under a rules-first policy.

    Parameters
    ----------
    rules_fn:
        Callable that judges one :class:`~detector.schema.Event` with the rules
        engine and returns a :class:`~detector.schema.Verdict`.
    llm_fn:
        Callable that judges one :class:`~detector.schema.Event` with the LLM
        detector and returns a :class:`~detector.schema.Verdict`.
    annotate_source:
        When true (the default) the returned verdict's ``reason`` is prefixed
        with a small tag recording which arm decided the unit. The verdict is
        copied, never mutated in place, so the underlying arms' outputs are left
        untouched.
    """

    def __init__(
        self,
        rules_fn: VerdictFn,
        llm_fn: VerdictFn,
        *,
        annotate_source: bool = True,
    ) -> None:
        self._rules_fn = rules_fn
        self._llm_fn = llm_fn
        self._annotate_source = annotate_source

    def judge(self, event: Event) -> Verdict:
        """Return the hybrid :class:`Verdict` for one unit.

        If the rules engine calls it an incident, that verdict wins; otherwise
        the LLM's verdict is returned.
        """
        rules_verdict = self._rules_fn(event)
        if rules_verdict.is_incident:
            return self._tag(rules_verdict, _RULES_TAG)
        llm_verdict = self._llm_fn(event)
        return self._tag(llm_verdict, _LLM_TAG)

    def predict(self, event: Event) -> Prediction:
        """Return the hybrid :class:`Prediction` for one unit."""
        return Prediction.from_verdict(event.id, self.judge(event))

    def __call__(self, event: Event) -> Verdict:
        return self.judge(event)

    def _tag(self, verdict: Verdict, tag: str) -> Verdict:
        if not self._annotate_source:
            return verdict
        reason = verdict.reason.strip()
        reason = f"{tag} {reason}" if reason else tag
        # model_copy keeps every other field (including the validated
        # confidence and normalised technique) exactly as the arm produced it.
        return verdict.model_copy(update={"reason": reason})
