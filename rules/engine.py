"""Deterministic rules baseline for the rules-vs-LLM comparison.

This module is the rule-based detector. It loads the Sigma-style rules in
``rules/sigma/*.yml`` (their titles, ATT&CK tags and, where the condition is a
simple threshold, their numeric thresholds and value lists) and applies an
equivalent detection routine to normalized :class:`Event` objects, returning one
:class:`Prediction` per unit.

The detection logic is a faithful re-implementation of the mini-siem detection
engine:

* count-based bursts (SSH / web-login brute force, 404 scan floods) read
  ``raw["count"]`` from the aggregated unit rather than re-counting log lines;
* web exploitation is matched on the URL-decoded request path and the
  User-Agent (SQLi, path traversal, OS command injection, Log4Shell/JNDI, XSS);
* scanning is matched on known scanner User-Agents, 404 floods and access to
  sensitive paths;
* two rules are correlations across units — a new Windows account created on a
  host that just saw a failed-logon burst, and a cloud IAM change by an identity
  that had recent console-login failures.

Design contract:

* ``run_rules(units) -> list[Prediction]`` is the public entry point. ``units``
  may be :class:`Event` objects, :class:`LabeledUnit` objects, or plain dicts of
  either; the engine extracts the :class:`Event` and *never* reads the label,
  so the baseline is scored honestly against ground truth it did not see.
* ``evaluate(event, context) -> Verdict`` applies the rules to a single event.

The engine is fully deterministic: no randomness, no network, no clock reads.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

from detector.schema import Event, LabeledUnit, Prediction, SourceType, Verdict

# Directory holding the Sigma rule definitions that back this engine.
SIGMA_DIR = Path(__file__).resolve().parent / "sigma"

# Fallback thresholds / catalogs, used only if a Sigma file cannot be parsed for
# them. The Sigma files are the source of truth when they load cleanly.
_DEFAULT_SSH_THRESHOLD = 10
_DEFAULT_WEB_AUTH_THRESHOLD = 20
_DEFAULT_SCAN_404_THRESHOLD = 30
_DEFAULT_WIN_FAILURE_THRESHOLD = 3
_DEFAULT_WIN_CORRELATION_MINUTES = 60
_DEFAULT_SENSITIVE_PORTS = frozenset({22, 3389, 3306, 5432, 27017})
_DEFAULT_OPEN_CIDR = "0.0.0.0/0"
_DEFAULT_LOGIN_PATHS = ("/login", "/api/login", "/admin/login", "/signin", "/auth")
_DEFAULT_AUTH_FAIL_STATUSES = frozenset({"401", "403"})
_DEFAULT_SENSITIVE_PATHS = (
    "/.env",
    "/.git",
    "/admin",
    "/phpmyadmin",
    "/backup",
    "/wp-admin",
    "/config",
)
_DEFAULT_SCANNER_UAS = ("sqlmap", "nikto", "nmap", "masscan", "gobuster", "dirbuster")
_DEFAULT_FOUND_STATUSES = frozenset({"200", "302"})
_DEFAULT_SQLI = ("union select", "or 1=1", "' or '1'='1", "information_schema", "sleep(")
_DEFAULT_TRAVERSAL = ("../", "/etc/passwd", "/etc/shadow")
_DEFAULT_XSS = ("<script", "onerror=", "javascript:")
_DEFAULT_CMD_RE = r"[;|]\s*(cat|ls|id|whoami|uname|curl|wget|nc|bash|sh)\b"

# Database ports are the highest-severity remote exposure.
_DB_PORTS = frozenset({3306, 5432, 27017})

_TAG_TECHNIQUE_RE = re.compile(r"attack\.(t\d{4}(?:\.\d{3})?)", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Sigma rule loading
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SigmaRule:
    """A parsed Sigma rule: its metadata plus the raw ``detection`` block."""

    stem: str
    title: str
    description: str
    level: str
    techniques: tuple[str, ...]
    detection: dict[str, Any]

    @property
    def primary_technique(self) -> Optional[str]:
        return self.techniques[0] if self.techniques else None


def _techniques_from_tags(tags: Iterable[str]) -> tuple[str, ...]:
    """Extract ATT&CK technique ids from Sigma ``tags``, most specific first."""
    found: list[str] = []
    for tag in tags or []:
        match = _TAG_TECHNIQUE_RE.search(str(tag))
        if match:
            found.append(match.group(1).upper())
    # Prefer sub-techniques (those containing a dot) but keep every id.
    found.sort(key=lambda t: (0 if "." in t else 1, t))
    # Deduplicate, preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for tech in found:
        if tech not in seen:
            seen.add(tech)
            unique.append(tech)
    return tuple(unique)


def load_sigma_rules(sigma_dir: Path | str = SIGMA_DIR) -> dict[str, SigmaRule]:
    """Load every ``*.yml`` rule under ``sigma_dir`` keyed by file stem."""
    directory = Path(sigma_dir)
    rules: dict[str, SigmaRule] = {}
    for path in sorted(directory.glob("*.yml")):
        with path.open("r", encoding="utf-8") as handle:
            doc = yaml.safe_load(handle) or {}
        rules[path.stem] = SigmaRule(
            stem=path.stem,
            title=str(doc.get("title", path.stem)),
            description=str(doc.get("description", "")).strip(),
            level=str(doc.get("level", "medium")),
            techniques=_techniques_from_tags(doc.get("tags", [])),
            detection=dict(doc.get("detection", {}) or {}),
        )
    return rules


def _threshold_from_condition(condition: Any, default: int) -> int:
    """Pull the numeric threshold out of a ``count() ... > N`` Sigma condition."""
    if isinstance(condition, str):
        match = re.search(r">\s*(\d+)", condition)
        if match:
            return int(match.group(1))
    return default


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _statuses_in_condition(condition: Any, default: frozenset[str]) -> frozenset[str]:
    if isinstance(condition, str):
        match = re.search(r"in\s*\[([\d,\s]+)\]", condition)
        if match:
            return frozenset(s.strip() for s in match.group(1).split(",") if s.strip())
    return default


# --------------------------------------------------------------------------- #
# Rule context (parsed thresholds + cross-unit correlation indexes)
# --------------------------------------------------------------------------- #
@dataclass
class RuleContext:
    """Everything the rules need beyond a single event.

    Holds the thresholds and value lists parsed from the Sigma files plus the
    correlation indexes (cloud login failures, Windows failed-logon bursts) built
    once from the whole batch of events.
    """

    rules: dict[str, SigmaRule]

    ssh_threshold: int = _DEFAULT_SSH_THRESHOLD
    web_auth_threshold: int = _DEFAULT_WEB_AUTH_THRESHOLD
    scan_404_threshold: int = _DEFAULT_SCAN_404_THRESHOLD
    win_failure_threshold: int = _DEFAULT_WIN_FAILURE_THRESHOLD
    win_correlation_minutes: int = _DEFAULT_WIN_CORRELATION_MINUTES

    sensitive_ports: frozenset[int] = _DEFAULT_SENSITIVE_PORTS
    open_cidr: str = _DEFAULT_OPEN_CIDR
    login_paths: tuple[str, ...] = _DEFAULT_LOGIN_PATHS
    auth_fail_statuses: frozenset[str] = _DEFAULT_AUTH_FAIL_STATUSES
    sensitive_paths: tuple[str, ...] = _DEFAULT_SENSITIVE_PATHS
    scanner_uas: tuple[str, ...] = _DEFAULT_SCANNER_UAS
    found_statuses: frozenset[str] = _DEFAULT_FOUND_STATUSES

    exploit_substrings: tuple[str, ...] = _DEFAULT_SQLI + _DEFAULT_TRAVERSAL + _DEFAULT_XSS
    jndi_markers: tuple[str, ...] = ("${jndi:",)
    cmd_injection_re: re.Pattern[str] = field(
        default_factory=lambda: re.compile(_DEFAULT_CMD_RE, re.IGNORECASE)
    )

    # Correlation indexes: sorted lists of (timestamp, key-fields).
    cloud_login_failures: list[tuple[datetime, Optional[str], Optional[str]]] = field(
        default_factory=list
    )
    win_failure_bursts: list[tuple[datetime, Optional[str]]] = field(default_factory=list)

    @classmethod
    def build(cls, events: list[Event], sigma_dir: Path | str = SIGMA_DIR) -> "RuleContext":
        rules = load_sigma_rules(sigma_dir)
        ctx = cls(rules=rules)
        ctx._load_from_sigma()
        ctx._build_correlation_indexes(events)
        return ctx

    # -- parse thresholds / catalogs out of the Sigma detection blocks -------- #
    def _load_from_sigma(self) -> None:
        ssh = self.rules.get("ssh_brute_force")
        if ssh:
            self.ssh_threshold = _threshold_from_condition(
                ssh.detection.get("condition"), _DEFAULT_SSH_THRESHOLD
            )

        web_auth = self.rules.get("web_auth_brute_force")
        if web_auth:
            self.web_auth_threshold = _threshold_from_condition(
                web_auth.detection.get("condition"), _DEFAULT_WEB_AUTH_THRESHOLD
            )
            selection = web_auth.detection.get("selection", {}) or {}
            paths = _as_list(selection.get("c-uri|contains"))
            if paths:
                self.login_paths = tuple(str(p).lower() for p in paths)
            statuses = _as_list(selection.get("sc-status"))
            if statuses:
                self.auth_fail_statuses = frozenset(str(s) for s in statuses)

        scan = self.rules.get("web_path_traversal_scan")
        if scan:
            flood = scan.detection.get("selection_404_flood", {}) or {}
            self.scan_404_threshold = _threshold_from_condition(
                flood.get("condition"), _DEFAULT_SCAN_404_THRESHOLD
            )
            sens = scan.detection.get("selection_sensitive_paths", {}) or {}
            paths = _as_list(sens.get("request_path|contains"))
            if paths:
                self.sensitive_paths = tuple(str(p).lower() for p in paths)
            uas = _as_list(
                (scan.detection.get("selection_scanner_ua", {}) or {}).get(
                    "user_agent|contains"
                )
            )
            if uas:
                self.scanner_uas = tuple(str(u).lower() for u in uas)
            self.found_statuses = _statuses_in_condition(
                scan.detection.get("condition"), _DEFAULT_FOUND_STATUSES
            )

        exploit = self.rules.get("web_exploit_attempt")
        if exploit:
            det = exploit.detection
            substrings: list[str] = []
            for key in ("selection_sqli", "selection_traversal", "selection_xss"):
                substrings.extend(
                    str(s).lower()
                    for s in _as_list((det.get(key, {}) or {}).get("request_path|contains"))
                )
            if substrings:
                self.exploit_substrings = tuple(substrings)
            log4 = det.get("selection_log4shell", {}) or {}
            markers = []
            markers.extend(str(m).lower() for m in _as_list(log4.get("request_path|contains")))
            markers.extend(str(m).lower() for m in _as_list(log4.get("user_agent|contains")))
            if markers:
                # Deduplicate while keeping order.
                self.jndi_markers = tuple(dict.fromkeys(markers))
            cmd = (det.get("selection_cmdinjection", {}) or {}).get("request_path|re")
            if cmd:
                self.cmd_injection_re = re.compile(str(cmd), re.IGNORECASE)

        windows = self.rules.get("windows_failed_logons_account_creation")
        if windows:
            failures = windows.detection.get("selection_failures", {}) or {}
            self.win_failure_threshold = _threshold_from_condition(
                failures.get("condition"), _DEFAULT_WIN_FAILURE_THRESHOLD
            )
            within = re.search(r"within\s+(\d+)m", str(windows.detection.get("condition", "")))
            if within:
                self.win_correlation_minutes = int(within.group(1))

        sg = self.rules.get("cloud_security_group_open")
        if sg:
            open_sel = sg.detection.get("selection_open_cidr", {}) or {}
            cidr = open_sel.get("cidr")
            if cidr:
                self.open_cidr = str(cidr)
            ports = _as_list(
                (sg.detection.get("selection_sensitive_ports", {}) or {}).get("port")
            )
            parsed_ports = {int(p) for p in ports if str(p).strip().isdigit()}
            if parsed_ports:
                self.sensitive_ports = frozenset(parsed_ports)

    # -- build cross-unit correlation indexes -------------------------------- #
    def _build_correlation_indexes(self, events: list[Event]) -> None:
        cloud_failures: list[tuple[datetime, Optional[str], Optional[str]]] = []
        win_bursts: list[tuple[datetime, Optional[str]]] = []
        for event in events:
            if event.source is SourceType.cloud_audit:
                if event.action == "console_login" and (event.status or "").lower() == "failure":
                    # A console-login-failure unit may be aggregated: its per-event
                    # identities live in raw["events"] (a list), not at the top
                    # level, so read each failure record to keep the username /
                    # source_ip the IAM-change correlation matches on. Fall back to
                    # the point shape (raw["event"] or the unit's own fields).
                    records = event.raw.get("events")
                    if isinstance(records, list) and records:
                        for rec in records:
                            if not isinstance(rec, dict):
                                continue
                            cloud_failures.append(
                                (
                                    _parse_ts(rec.get("timestamp"), event.timestamp),
                                    rec.get("username"),
                                    rec.get("source_ip"),
                                )
                            )
                    else:
                        nested = _cloud_fields(event)
                        cloud_failures.append(
                            (
                                _aware(event.timestamp),
                                nested.get("username") or event.target,
                                nested.get("source_ip") or event.actor,
                            )
                        )
            elif event.source is SourceType.windows_security:
                if _win_field(event, "event_code") == "4625":
                    count = _int(event.raw.get("count"), 1)
                    if count > self.win_failure_threshold:
                        win_bursts.append((_aware(event.timestamp), _win_field(event, "host")))
        self.cloud_login_failures = sorted(cloud_failures, key=lambda item: item[0])
        self.win_failure_bursts = sorted(win_bursts, key=lambda item: item[0])


# --------------------------------------------------------------------------- #
# Field accessors (tolerant to point vs aggregated unit shapes)
# --------------------------------------------------------------------------- #
def _aware(value: datetime) -> datetime:
    """Return a timezone-aware datetime (assume UTC for naive inputs)."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _parse_ts(value: Any, fallback: datetime) -> datetime:
    """Parse an ISO-8601 timestamp string to an aware datetime, else the fallback."""
    if isinstance(value, str):
        try:
            return _aware(datetime.fromisoformat(value))
        except ValueError:
            pass
    return _aware(fallback)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _cloud_fields(event: Event) -> dict[str, Any]:
    nested = event.raw.get("event")
    return nested if isinstance(nested, dict) else event.raw


def _win_field(event: Event, key: str) -> Optional[str]:
    """Read a Windows field from either the aggregated (flat) or point shape."""
    if key in event.raw and not isinstance(event.raw.get(key), dict):
        value = event.raw.get(key)
        return None if value is None else str(value)
    nested = event.raw.get("event")
    if isinstance(nested, dict) and nested.get(key) is not None:
        return str(nested.get(key))
    return None


def _decode(text: str) -> str:
    return urllib.parse.unquote(str(text))


def _nginx_paths(event: Event) -> list[str]:
    """Candidate request paths for an nginx unit, URL-decoded and lowercased."""
    candidates: list[str] = []
    path = event.raw.get("path")
    if path:
        candidates.append(str(path))
    for sample in _as_list(event.raw.get("sample_paths")):
        candidates.append(str(sample))
    if event.target and not event.target.endswith("paths"):
        candidates.append(event.target)
    return [_decode(c).lower() for c in candidates if c]


def _nginx_user_agents(event: Event) -> list[str]:
    agents: list[str] = []
    ua = event.raw.get("user_agent")
    if ua:
        agents.append(str(ua))
    for sample in _as_list(event.raw.get("user_agents")):
        agents.append(str(sample))
    return [a.lower() for a in agents if a]


def _nginx_statuses(event: Event) -> set[str]:
    """All HTTP statuses seen in an nginx unit (point status plus aggregates)."""
    statuses: set[str] = set()
    if event.status:
        statuses.add(str(event.status))
    if event.raw.get("final_status"):
        statuses.add(str(event.raw["final_status"]))
    for status in _as_list(event.raw.get("status_sequence")):
        statuses.add(str(status))
    for line in _as_list(event.raw.get("sample_lines")):
        match = re.search(r"\"\s+(\d{3})\s+\d+", str(line))
        if match:
            statuses.add(match.group(1))
    return statuses


def _cmd_injection_source(event: Event) -> list[str]:
    """Raw (not lowercased, still decoded) request paths for regex matching."""
    candidates: list[str] = []
    path = event.raw.get("path")
    if path:
        candidates.append(str(path))
    for sample in _as_list(event.raw.get("sample_paths")):
        candidates.append(str(sample))
    if event.target and not event.target.endswith("paths"):
        candidates.append(event.target)
    return [_decode(c) for c in candidates if c]


# --------------------------------------------------------------------------- #
# Individual rules — each returns a Verdict on a match, else None
# --------------------------------------------------------------------------- #
def _confidence_for_count(count: int, low: float, mid: float, high: float) -> float:
    if count >= 100:
        return high
    if count >= 30:
        return mid
    return low


def _rule_ssh_brute_force(event: Event, ctx: RuleContext) -> Optional[Verdict]:
    if event.source is not SourceType.linux_auth or event.action != "ssh_failed_password":
        return None
    count = _int(event.raw.get("count"), 1)
    if count <= ctx.ssh_threshold:
        return None
    confidence = _confidence_for_count(count, 0.75, 0.9, 0.99)
    return Verdict(
        is_incident=True,
        technique="T1110.001",
        confidence=confidence,
        reason=(
            f"SSH Brute Force: {count} failed SSH password attempts from {event.actor} "
            f"(> {ctx.ssh_threshold} in window); target users {event.target}."
        ),
    )


def _rule_web_auth_brute_force(event: Event, ctx: RuleContext) -> Optional[Verdict]:
    if event.source is not SourceType.nginx_access:
        return None
    paths = _nginx_paths(event)
    if not any(login in p for p in paths for login in ctx.login_paths):
        return None
    if not (_nginx_statuses(event) & ctx.auth_fail_statuses):
        return None
    count = _int(event.raw.get("count"), 1)
    if count <= ctx.web_auth_threshold:
        return None
    confidence = _confidence_for_count(count, 0.8, 0.9, 0.97)
    return Verdict(
        is_incident=True,
        technique="T1110",
        confidence=confidence,
        reason=(
            f"Web Login Brute Force: {count} failed web logins from {event.actor} "
            f"to {event.target} (> {ctx.web_auth_threshold} in window)."
        ),
    )


def _rule_web_exploit(event: Event, ctx: RuleContext) -> Optional[Verdict]:
    if event.source is not SourceType.nginx_access:
        return None
    paths = _nginx_paths(event)
    agents = _nginx_user_agents(event)

    for path in paths:
        for signature in ctx.exploit_substrings:
            if signature in path:
                return _exploit_verdict(event, f"payload signature '{signature}' in path")

    for haystack in paths + agents:
        for marker in ctx.jndi_markers:
            if marker in haystack:
                return _exploit_verdict(event, "Log4Shell/JNDI marker")

    for path in _cmd_injection_source(event):
        if ctx.cmd_injection_re.search(path):
            return _exploit_verdict(event, "OS command-injection pattern in path")

    return None


def _exploit_verdict(event: Event, detail: str) -> Verdict:
    return Verdict(
        is_incident=True,
        technique="T1190",
        confidence=0.95,
        reason=(
            f"Web Exploit Attempt: {detail} from {event.actor} against {event.target}."
        ),
    )


def _rule_web_scan(event: Event, ctx: RuleContext) -> Optional[Verdict]:
    if event.source is not SourceType.nginx_access:
        return None
    agents = _nginx_user_agents(event)
    for agent in agents:
        for scanner in ctx.scanner_uas:
            if scanner in agent:
                return Verdict(
                    is_incident=True,
                    technique="T1595.002",
                    confidence=0.9,
                    reason=(
                        f"Web Scanning: known scanner User-Agent '{scanner}' from "
                        f"{event.actor}."
                    ),
                )

    count = _int(event.raw.get("count"), 1)
    if str(event.status) == "404" and count > ctx.scan_404_threshold:
        return Verdict(
            is_incident=True,
            technique="T1595.002",
            confidence=_confidence_for_count(count, 0.7, 0.85, 0.95),
            reason=(
                f"Web Scanning: {count} HTTP 404 responses from {event.actor} "
                f"(> {ctx.scan_404_threshold} in window), consistent with directory brute force."
            ),
        )

    paths = _nginx_paths(event)
    hit = next(
        (sp for p in paths for sp in ctx.sensitive_paths if sp in p),
        None,
    )
    if hit and (_nginx_statuses(event) & ctx.found_statuses):
        return Verdict(
            is_incident=True,
            technique="T1083",
            confidence=0.75,
            reason=(
                f"Sensitive Path Access: request to '{hit}' returned a found status "
                f"from {event.actor}, consistent with file/directory discovery."
            ),
        )
    return None


def _rule_windows_account_after_failures(event: Event, ctx: RuleContext) -> Optional[Verdict]:
    if event.source is not SourceType.windows_security:
        return None
    if _win_field(event, "event_code") != "4720":
        return None
    host = _win_field(event, "host")
    when = _aware(event.timestamp)
    window = ctx.win_correlation_minutes * 60
    for burst_time, burst_host in ctx.win_failure_bursts:
        if burst_host == host and 0 <= (when - burst_time).total_seconds() <= window:
            new_account = _win_field(event, "new_account") or event.target
            return Verdict(
                is_incident=True,
                technique="T1136.001",
                confidence=0.9,
                reason=(
                    f"Account Created After Failed Logons: new account '{new_account}' on "
                    f"{host} within {ctx.win_correlation_minutes}m of a failed-logon burst."
                ),
            )
    return None


def _rule_cloud_security_group_open(event: Event, ctx: RuleContext) -> Optional[Verdict]:
    if event.source is not SourceType.cloud_audit:
        return None
    if "security_group_rule" not in event.action:
        return None
    fields = _cloud_fields(event)
    if str(fields.get("cidr")) != ctx.open_cidr:
        return None
    port = _int(fields.get("port"), -1)
    if port not in ctx.sensitive_ports:
        return None
    severity = "database" if port in _DB_PORTS else "remote-access"
    confidence = 0.95 if port in _DB_PORTS else 0.9
    return Verdict(
        is_incident=True,
        technique="T1562.007",
        confidence=confidence,
        reason=(
            f"Security Group Opened to Internet: {severity} port {port} exposed to "
            f"{ctx.open_cidr} by {fields.get('username') or event.actor}."
        ),
    )


def _rule_cloud_iam_change_after_failure(event: Event, ctx: RuleContext) -> Optional[Verdict]:
    if event.source is not SourceType.cloud_audit or event.action != "iam_policy_changed":
        return None
    fields = _cloud_fields(event)
    username = fields.get("username")
    source_ip = fields.get("source_ip")
    when = _aware(event.timestamp)
    for fail_time, fail_user, fail_ip in ctx.cloud_login_failures:
        if fail_time > when:
            continue
        if (username and fail_user == username) or (source_ip and fail_ip == source_ip):
            return Verdict(
                is_incident=True,
                technique="T1098",
                confidence=0.8,
                reason=(
                    f"IAM Change After Login Failure: {event.action} by "
                    f"{username or source_ip} which had recent console-login failures."
                ),
            )
    return None


# Priority order: the first rule to match wins. Web exploitation is placed above
# scanning so that a payload request from a scanner User-Agent (e.g. sqlmap) is
# classified as exploitation (T1190) rather than reconnaissance.
_RULE_ORDER = (
    _rule_web_exploit,
    _rule_web_auth_brute_force,
    _rule_ssh_brute_force,
    _rule_windows_account_after_failures,
    _rule_cloud_security_group_open,
    _rule_cloud_iam_change_after_failure,
    _rule_web_scan,
)

_BENIGN_VERDICT = Verdict(
    is_incident=False,
    technique=None,
    confidence=0.5,
    reason="No rule matched: no known adversary signature, burst or correlation.",
)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def evaluate(event: Event, context: Optional[RuleContext] = None) -> Verdict:
    """Apply the rules to a single event and return the first matching verdict.

    ``context`` carries the parsed thresholds and the cross-unit correlation
    indexes. When it is omitted, an empty context is built (correlation rules,
    which need other units, then simply do not fire).
    """
    ctx = context if context is not None else RuleContext.build([event])
    for rule in _RULE_ORDER:
        verdict = rule(event, ctx)
        if verdict is not None:
            return verdict
    return _BENIGN_VERDICT.model_copy()


def _to_event(unit: Any) -> Event:
    """Coerce a unit (Event / LabeledUnit / dict of either) into an Event.

    The label on a :class:`LabeledUnit` is deliberately ignored: the detector
    must not see ground truth.
    """
    if isinstance(unit, Event):
        return unit
    if isinstance(unit, LabeledUnit):
        return unit.event
    if isinstance(unit, dict):
        if "event" in unit and isinstance(unit["event"], dict):
            return Event.model_validate(unit["event"])
        return Event.model_validate(unit)
    raise TypeError(f"Cannot turn {type(unit).__name__} into an Event")


def run_rules(units: Iterable[Any]) -> list[Prediction]:
    """Run the rules baseline over ``units`` and return one Prediction each.

    ``units`` may be :class:`Event`, :class:`LabeledUnit` or dict objects. The
    engine builds correlation indexes from the whole batch first, then scores
    each unit independently. Output order matches input order.
    """
    events = [_to_event(unit) for unit in units]
    ctx = RuleContext.build(events)
    predictions: list[Prediction] = []
    for event in events:
        verdict = evaluate(event, ctx)
        predictions.append(Prediction.from_verdict(event.id, verdict))
    return predictions


def load_corpus(path: Path | str) -> list[LabeledUnit]:
    """Load ``corpus/labels.jsonl`` as :class:`LabeledUnit` rows (for the CLI)."""
    units: list[LabeledUnit] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                units.append(LabeledUnit.model_validate(json.loads(line)))
    return units


def _main() -> None:
    """Sanity check: run the rules over the labeled corpus and print a summary.

    The labels are used only to print accuracy here, never to make a decision.
    """
    corpus_path = Path(__file__).resolve().parent.parent / "corpus" / "labels.jsonl"
    units = load_corpus(corpus_path)
    predictions = run_rules(units)
    by_id = {p.unit_id: p for p in predictions}

    tp = fp = fn = tn = 0
    tech_hit = 0
    misses: list[str] = []
    false_alarms: list[str] = []
    for unit in units:
        pred = by_id[unit.unit_id]
        if pred.is_incident and unit.is_incident:
            tp += 1
            if pred.technique == unit.technique:
                tech_hit += 1
        elif pred.is_incident and not unit.is_incident:
            fp += 1
            false_alarms.append(unit.unit_id)
        elif not pred.is_incident and unit.is_incident:
            fn += 1
            misses.append(f"{unit.unit_id} ({unit.technique})")
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    print(f"units={len(units)}  incidents={tp + fn}  benign={tn + fp}")
    print(f"TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"precision={precision:.3f}  recall={recall:.3f}  f1={f1:.3f}")
    print(f"technique match on TP = {tech_hit}/{tp}")
    print(f"false positives : {false_alarms}")
    print(f"false negatives : {misses}")


if __name__ == "__main__":
    _main()
