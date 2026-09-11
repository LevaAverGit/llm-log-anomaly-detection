"""Rules baseline: the deterministic Sigma engine that produces the headline
F1 must fire on each attack pattern it encodes, stay silent on look-alikes, and
reproduce the committed predictions exactly.

The engine is the reference the whole comparison is measured against, so these
tests pin its per-rule behaviour on small synthetic units (thresholds, payloads,
the two cross-unit correlations) and its honesty (it never reads a label), then
assert that running it over the real corpus reproduces
``reports/predictions/rules.jsonl`` byte-for-byte — the offline reproducibility
guarantee behind the README's rules row. No model, no network.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detector.schema import Event, Label, LabeledUnit, SourceType  # noqa: E402
from rules.engine import RuleContext, evaluate, run_rules  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
_T0 = datetime(2026, 1, 10, 9, 0, 0, tzinfo=timezone.utc)


def _event(**kwargs) -> Event:
    base = dict(
        id="t-0001",
        source=SourceType.linux_auth,
        timestamp=_T0,
        actor="203.0.113.10",
        action="noop",
        target=None,
        status=None,
        raw={},
    )
    base.update(kwargs)
    return Event(**base)


def _verdict(event: Event, *context: Event):
    """Evaluate one event with a context built from the given events (self only
    when none are passed)."""
    ctx = RuleContext.build(list(context) or [event])
    return evaluate(event, ctx)


# --------------------------------------------------------------------------- #
# Single-unit rules: threshold, payload, scanner, cloud opening.
# --------------------------------------------------------------------------- #
def test_ssh_burst_over_threshold_is_incident():
    e = _event(action="ssh_failed_password", status="failure", raw={"count": 13})
    v = _verdict(e)
    assert v.is_incident is True
    assert v.technique == "T1110.001"


def test_ssh_burst_below_threshold_is_benign():
    """Two failures are noise: the count-based rule must not fire (the corpus
    plants exactly this benign look-alike, lx-0041)."""
    e = _event(action="ssh_failed_password", status="failure", raw={"count": 2})
    assert _verdict(e).is_incident is False


def test_web_payload_is_exploit():
    e = _event(
        source=SourceType.nginx_access,
        action="http_get",
        target="/search",
        raw={"path": "/search?q=union select 1 from users", "user_agent": "curl/8"},
    )
    v = _verdict(e)
    assert v.is_incident is True
    assert v.technique == "T1190"


def test_scanner_user_agent_is_recon():
    e = _event(
        source=SourceType.nginx_access,
        action="http_get",
        status="404",
        raw={"count": 50, "user_agents": ["gobuster/3.1"]},
    )
    v = _verdict(e)
    assert v.is_incident is True
    assert v.technique == "T1595.002"


def test_security_group_opened_to_internet_on_sensitive_port():
    e = _event(
        source=SourceType.cloud_audit,
        action="security_group_rule_added",
        raw={"event": {"cidr": "0.0.0.0/0", "port": 22, "username": "svc"}},
    )
    v = _verdict(e)
    assert v.is_incident is True
    assert v.technique == "T1562.007"


def test_security_group_on_web_port_is_benign():
    """0.0.0.0/0 on 443 is not a sensitive-port opening; the rule must not fire."""
    e = _event(
        source=SourceType.cloud_audit,
        action="security_group_rule_added",
        raw={"event": {"cidr": "0.0.0.0/0", "port": 443, "username": "svc"}},
    )
    assert _verdict(e).is_incident is False


def test_routine_login_is_benign():
    e = _event(
        source=SourceType.cloud_audit,
        action="console_login",
        status="success",
        raw={"event": {"username": "sre@example.com"}},
    )
    assert _verdict(e).is_incident is False


# --------------------------------------------------------------------------- #
# Cross-unit correlation rules: they need the whole batch, and must not fire
# without the correlated partner.
# --------------------------------------------------------------------------- #
def test_windows_account_after_failure_burst_correlates():
    burst = _event(
        id="w-0001",
        source=SourceType.windows_security,
        timestamp=_T0,
        action="logon_failure",
        status="failure",
        raw={"count": 6, "event_code": "4625", "host": "WIN-01"},
    )
    account = _event(
        id="w-0002",
        source=SourceType.windows_security,
        timestamp=_T0 + timedelta(minutes=10),
        action="user_account_created",
        raw={"event": {"event_code": "4720", "host": "WIN-01", "new_account": "backdoor"}},
    )
    assert _verdict(account, burst, account).technique == "T1136.001"
    # Same account creation, no preceding burst in context -> benign.
    assert _verdict(account, account).is_incident is False


def test_cloud_iam_change_after_failure_correlates():
    fail = _event(
        id="c-0001",
        source=SourceType.cloud_audit,
        timestamp=_T0,
        action="console_login",
        status="failure",
        raw={"event": {"username": "bob", "source_ip": "9.9.9.9"}},
    )
    iam = _event(
        id="c-0002",
        source=SourceType.cloud_audit,
        timestamp=_T0 + timedelta(minutes=5),
        action="iam_policy_changed",
        raw={"event": {"username": "bob", "source_ip": "9.9.9.9"}},
    )
    assert _verdict(iam, fail, iam).technique == "T1098"
    # No preceding failure in context -> routine administration, benign.
    assert _verdict(iam, iam).is_incident is False


# --------------------------------------------------------------------------- #
# Honesty and reproducibility.
# --------------------------------------------------------------------------- #
def test_run_rules_never_reads_the_label():
    """A LabeledUnit whose label lies (benign event marked incident) must still
    be judged benign: the detector never sees ground truth."""
    benign = _event(
        id="h-0001",
        source=SourceType.cloud_audit,
        action="console_login",
        status="success",
        raw={"event": {"username": "a"}},
    )
    mislabeled = LabeledUnit(
        unit_id="h-0001",
        event=benign,
        label=Label.incident,
        technique="T1078",
        rationale="deliberately mislabeled to prove the engine ignores it",
    )
    assert run_rules([mislabeled])[0].is_incident is False


def test_engine_reproduces_committed_rules_predictions():
    """run_rules over the real corpus must match reports/predictions/rules.jsonl
    exactly — the offline guarantee behind the README's deterministic rules row."""
    labels_path = _REPO_ROOT / "corpus" / "labels.jsonl"
    committed_path = _REPO_ROOT / "reports" / "predictions" / "rules.jsonl"
    if not committed_path.exists():
        pytest.skip("committed rules predictions not present")

    units = [
        LabeledUnit.model_validate(json.loads(line))
        for line in labels_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    produced = {p.unit_id: p for p in run_rules(units)}
    committed = {
        row["unit_id"]: row
        for row in (json.loads(l) for l in committed_path.read_text(encoding="utf-8").splitlines() if l.strip())
    }

    assert set(produced) == set(committed)
    for unit_id, row in committed.items():
        pred = produced[unit_id]
        assert pred.is_incident == row["is_incident"], f"{unit_id}: incident flag drifted"
        assert (pred.technique or None) == (row.get("technique") or None), f"{unit_id}: technique drifted"
