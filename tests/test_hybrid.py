"""Hybrid policy: rules first, LLM only on what the rules leave benign.

The module is decoupled from the real engine and detector — it takes two
``Event -> Verdict`` callables — so these tests wire in stubs and assert the
policy itself: a rules-positive is trusted and never overturned by the LLM, a
rules-negative defers to the LLM, and the arm that decided is recorded in the
reason without mutating the arms' own outputs. Because a positive is never
overturned, the hybrid's recall can only meet or beat the rules'. No model.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detector.hybrid import HybridDetector  # noqa: E402
from detector.schema import Event, SourceType, Verdict  # noqa: E402


def _event(uid: str = "u-0001") -> Event:
    return Event(
        id=uid,
        source=SourceType.linux_auth,
        timestamp=datetime(2026, 1, 10, 9, 0, 0, tzinfo=timezone.utc),
        actor="203.0.113.10",
        action="ssh_failed_password",
        target="root",
        status="failure",
        raw={"count": 13},
    )


def _incident(reason: str = "hit") -> Verdict:
    return Verdict(is_incident=True, technique="T1110.001", confidence=0.9, reason=reason)


def _benign(reason: str = "clean") -> Verdict:
    return Verdict(is_incident=False, technique=None, confidence=0.5, reason=reason)


def test_rules_positive_is_trusted_and_not_overturned():
    """When the rules fire, the hybrid returns their verdict and never consults
    the LLM (even one that would disagree)."""
    llm_called = False

    def llm_fn(_event):
        nonlocal llm_called
        llm_called = True
        return _benign("llm says benign")

    hybrid = HybridDetector(lambda e: _incident("rules matched"), llm_fn)
    verdict = hybrid.judge(_event())

    assert verdict.is_incident is True
    assert verdict.technique == "T1110.001"
    assert llm_called is False, "the LLM must not be called once the rules fire"


def test_rules_negative_defers_to_llm():
    """When the rules stay silent, the LLM's verdict decides the unit."""
    hybrid = HybridDetector(lambda e: _benign("no rule matched"), lambda e: _incident("llm caught it"))
    verdict = hybrid.judge(_event())
    assert verdict.is_incident is True


def test_reason_records_the_deciding_arm_without_mutation():
    rules_out = _incident("rules matched")
    hybrid = HybridDetector(lambda e: rules_out, lambda e: _benign())
    verdict = hybrid.judge(_event())

    assert "[hybrid:rules]" in verdict.reason
    assert "rules matched" in verdict.reason
    # The underlying arm's verdict is copied, never mutated in place.
    assert rules_out.reason == "rules matched"

    hybrid_llm = HybridDetector(lambda e: _benign("silent"), lambda e: _incident("llm caught it"))
    assert "[hybrid:llm]" in hybrid_llm.judge(_event()).reason


def test_annotate_source_can_be_disabled():
    hybrid = HybridDetector(lambda e: _incident("rules matched"), lambda e: _benign(), annotate_source=False)
    assert hybrid.judge(_event()).reason == "rules matched"


def test_hybrid_recall_is_at_least_rules_recall():
    """On a small set of incident units, the hybrid catches everything the rules
    catch plus anything the LLM adds on rules-negative units — so its recall
    cannot fall below the rules'."""
    events = [_event(f"u-{i:04d}") for i in range(4)]
    # Rules catch only the first two; the LLM catches all four.
    rules = {events[0].id, events[1].id}

    def rules_fn(e):
        return _incident() if e.id in rules else _benign()

    hybrid = HybridDetector(rules_fn, lambda e: _incident("llm"))
    caught = sum(1 for e in events if hybrid.judge(e).is_incident)
    rules_caught = sum(1 for e in events if rules_fn(e).is_incident)

    assert caught >= rules_caught
    assert caught == 4  # the LLM lifts the two the rules missed
