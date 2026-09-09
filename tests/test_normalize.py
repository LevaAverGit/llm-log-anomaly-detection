"""Normalization tests: each of the four raw log formats must turn into valid
:class:`Event` objects that satisfy the shared schema.

Expected interface (``detector.normalize``): a callable that takes a raw log
file path and its :class:`SourceType` and returns a list of ``Event`` (or of
dicts that validate as ``Event``). Either a generic dispatcher
(``normalize_file(path, source)`` / ``normalize(path, source)``) or per-source
functions (``normalize_linux_auth(path)`` ...) are accepted; see
``tests/_support.run_normalize``.

Key contract exercised here: a brute-force / scan burst is aggregated into ONE
event whose ``raw`` carries ``count`` and ``sample_lines`` (so a count-based
rule reads ``raw['count']`` instead of re-counting log lines).
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _support as S  # noqa: E402
from detector.schema import Event, SourceType  # noqa: E402

ALL_SOURCES = [
    SourceType.linux_auth,
    SourceType.nginx_access,
    SourceType.windows_security,
    SourceType.cloud_audit,
]


@pytest.fixture(scope="module")
def normalize_mod():
    return S.import_module_or_fail("detector.normalize")


def _normalized(normalize_mod, source: SourceType) -> list[Event]:
    path = S.raw_path(source)
    assert path.exists(), f"raw log for {source.value} missing at {path}"
    events = S.run_normalize(normalize_mod, source, path)
    assert isinstance(events, list) and events, f"no events produced for {source.value}"
    return events


@pytest.mark.parametrize("source", ALL_SOURCES, ids=lambda s: s.value)
def test_each_format_produces_valid_events(normalize_mod, source):
    """Every source parses into schema-valid Events tagged with that source."""
    events = _normalized(normalize_mod, source)

    for ev in events:
        assert isinstance(ev, Event)
        # ``as_event`` already validated the model; re-assert the invariants
        # that downstream detectors and scoring rely on.
        assert ev.source == source, f"{ev.id}: source {ev.source} != {source}"
        assert isinstance(ev.action, str) and ev.action.strip(), f"{ev.id}: empty action"
        assert isinstance(ev.timestamp, datetime), f"{ev.id}: timestamp not a datetime"
        assert isinstance(ev.id, str) and ev.id.strip(), "event id must be a non-empty string"
        assert isinstance(ev.raw, dict)

    ids = [ev.id for ev in events]
    assert len(ids) == len(set(ids)), f"{source.value}: event ids are not unique"


def test_all_four_sources_covered(normalize_mod):
    """The four formats normalize to four disjoint, non-empty source sets."""
    per_source = {src: _normalized(normalize_mod, src) for src in ALL_SOURCES}
    assert set(per_source) == set(ALL_SOURCES)
    for src, events in per_source.items():
        assert all(ev.source == src for ev in events)


def test_linux_bruteforce_is_aggregated(normalize_mod):
    """The SSH failure burst collapses into a single aggregated Event.

    ``corpus/raw/linux_auth.log`` contains 13 failed passwords from a single IP
    (192.0.2.100). The contract requires that this becomes ONE event whose
    ``raw`` carries the count and representative sample lines.
    """
    events = _normalized(normalize_mod, SourceType.linux_auth)
    aggregated = [ev for ev in events if isinstance(ev.raw.get("count"), int) and ev.raw["count"] >= 2]
    assert aggregated, "expected at least one aggregated (count>=2) linux event"

    max_burst = max(ev.raw["count"] for ev in aggregated)
    assert max_burst >= 10, (
        f"largest aggregated burst is {max_burst}; the 13-line SSH brute force "
        "from 192.0.2.100 should aggregate into one unit"
    )

    for ev in aggregated:
        samples = ev.raw.get("sample_lines")
        assert isinstance(samples, list) and samples, (
            f"{ev.id}: aggregated unit must keep non-empty 'sample_lines' as evidence"
        )


def test_nginx_actions_are_http(normalize_mod):
    """nginx access units expose HTTP-shaped actions and an HTTP-ish status."""
    events = _normalized(normalize_mod, SourceType.nginx_access)
    assert any(ev.action.lower().startswith("http") for ev in events), (
        "expected nginx actions to be normalized to http_* verbs"
    )
    for ev in events:
        if ev.status is not None:
            assert isinstance(ev.status, str)


def test_windows_and_cloud_actors_present(normalize_mod):
    """Windows and cloud units normalize with meaningful actions/actors."""
    windows = _normalized(normalize_mod, SourceType.windows_security)
    assert any("logon" in ev.action.lower() for ev in windows), (
        "expected at least one Windows logon-related action"
    )

    cloud = _normalized(normalize_mod, SourceType.cloud_audit)
    assert any(ev.action.lower() == "console_login" or "login" in ev.action.lower()
               or "iam" in ev.action.lower() or "security_group" in ev.action.lower()
               for ev in cloud), "expected recognizable cloud audit actions"


def test_aggregated_units_carry_window(normalize_mod):
    """Where a source aggregates, the window bounds travel with the unit."""
    seen_aggregated = False
    for source in ALL_SOURCES:
        for ev in _normalized(normalize_mod, source):
            if isinstance(ev.raw.get("count"), int) and ev.raw["count"] >= 2:
                seen_aggregated = True
                if "window_start" in ev.raw:
                    # window_start must be parseable / consistent when present.
                    assert ev.raw["window_start"], f"{ev.id}: empty window_start"
    assert seen_aggregated, "expected at least one aggregated unit across the four sources"
