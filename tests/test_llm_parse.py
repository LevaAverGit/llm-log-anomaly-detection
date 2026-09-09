"""LLM response parsing: valid / broken / partial JSON must degrade gracefully.

The LLM detector cannot trust a local model to always emit clean JSON, so the
response parser must:

- parse well-formed JSON into a :class:`Verdict`;
- find JSON embedded in prose or Markdown fences;
- fill sane defaults for missing fields instead of raising;
- never raise on broken / empty output, and always return a confidence in
  ``[0, 1]`` (a raw ``confidence`` of 1.7 cannot reach the Verdict unclamped).

Expected interface (``detector.llm``): a callable ``parse_verdict(text) ->
Verdict`` (name resolved from a small candidate list in ``tests/_support``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _support as S  # noqa: E402
from detector.schema import Verdict  # noqa: E402


@pytest.fixture(scope="module")
def parse():
    mod = S.import_module_or_fail("detector.llm")
    return S.resolve_parser(mod)


def _parse_verdict(parse, text: str) -> Verdict:
    """Call the parser and coerce the result into a Verdict, failing clearly."""
    try:
        raw = parse(text)
    except Exception as exc:  # noqa: BLE001 - the whole point is no crashes
        pytest.fail(f"parser raised on input {text!r}: {exc!r}")
    try:
        return S.as_verdict(raw)
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"parser returned an un-Verdict-able value for {text!r}: {exc!r}")


def test_valid_incident_json(parse):
    text = (
        '{"is_incident": true, "technique": "T1110.001", '
        '"confidence": 0.91, "reason": "Sustained SSH password guessing"}'
    )
    v = _parse_verdict(parse, text)
    assert v.is_incident is True
    assert v.technique == "T1110.001"
    assert v.confidence == pytest.approx(0.91)
    assert v.reason


def test_valid_benign_json(parse):
    text = '{"is_incident": false, "technique": null, "confidence": 0.15, "reason": "normal login"}'
    v = _parse_verdict(parse, text)
    assert v.is_incident is False
    assert v.technique is None
    assert 0.0 <= v.confidence <= 1.0


def test_technique_is_upper_cased(parse):
    text = '{"is_incident": true, "technique": "t1190", "confidence": 0.8, "reason": "sqli"}'
    v = _parse_verdict(parse, text)
    assert v.technique == "T1190"


def test_json_embedded_in_prose_and_fences(parse):
    text = (
        "Sure, here is my assessment:\n\n"
        "```json\n"
        '{"is_incident": true, "technique": "T1190", "confidence": 0.7, '
        '"reason": "path traversal against /etc/passwd"}\n'
        "```\n\n"
        "Let me know if you need more detail."
    )
    v = _parse_verdict(parse, text)
    assert v.is_incident is True
    assert v.technique == "T1190"
    assert 0.0 <= v.confidence <= 1.0


def test_partial_json_missing_confidence(parse):
    """Missing ``confidence`` must be defaulted, not crash the Verdict build."""
    text = '{"is_incident": true, "reason": "looks like brute force"}'
    v = _parse_verdict(parse, text)
    assert v.is_incident is True
    assert v.technique is None
    assert 0.0 <= v.confidence <= 1.0


def test_partial_json_only_flag(parse):
    text = '{"is_incident": false}'
    v = _parse_verdict(parse, text)
    assert v.is_incident is False
    assert isinstance(v.reason, str)
    assert 0.0 <= v.confidence <= 1.0


def test_confidence_out_of_range_is_clamped(parse):
    """A confidence above 1.0 must be pulled back into range, not raised."""
    text = '{"is_incident": true, "confidence": 1.7, "reason": "over-confident model"}'
    v = _parse_verdict(parse, text)
    assert 0.0 <= v.confidence <= 1.0


@pytest.mark.parametrize(
    "text",
    [
        "I think this is definitely an attack!!!",   # prose, no JSON at all
        "",                                            # empty output
        "   ",                                         # whitespace only
        "{not valid json, missing quotes}",           # broken JSON
        '{"is_incident": true, "confidence":',        # truncated JSON
        "null",                                        # valid JSON but not an object
    ],
)
def test_broken_output_degrades_gracefully(parse, text):
    """No exception, always a usable Verdict with a valid confidence."""
    v = _parse_verdict(parse, text)
    assert isinstance(v, Verdict)
    assert isinstance(v.is_incident, bool)
    assert 0.0 <= v.confidence <= 1.0
    assert isinstance(v.reason, str)
