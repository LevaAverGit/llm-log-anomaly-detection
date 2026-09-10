"""Turn the four raw log formats into normalized :class:`Event` units.

Each of the four mini-siem sources has its own on-disk format:

- ``linux_auth``       — syslog lines (sshd / sudo).
- ``nginx_access``     — combined access-log lines.
- ``windows_security`` — one JSON object per line (Security event log export).
- ``cloud_audit``      — one JSON object per line (cloud control-plane audit).

This module parses each format and shapes the records into *analysis units*,
following the same conventions as ``corpus/build_corpus.py``:

- A brute-force / scan burst from one acting entity collapses into a single
  aggregated unit whose ``raw`` carries ``count``, ``window_start`` /
  ``window_end`` and a few ``sample_lines`` (plus per-source extras such as
  ``distinct_users``, ``user_agents``, ``sample_paths``, ``status_sequence`` and
  ``final_status``). A count-based rule and an LLM both need the whole burst to
  judge it, so it is one unit.
- Discrete activity (one login, one request, one Windows event, one cloud
  action) stays one unit each, with the original record kept under ``raw``.

The aggregation here is *knowledge-free*: it groups by acting entity and by the
content of the requests, and it never consults the injected-attack map that the
corpus builder uses to label units. It therefore produces the *shape* of the
analysis units but assigns no labels — labeling is the corpus builder's job, and
the authoritative labeled units (with stable ids) live in
``corpus/labels.jsonl``. Use :func:`normalize_corpus` to see the same units this
module derives from ``corpus/raw/``.

The action vocabulary matches the rest of the repository:
``ssh_failed_password``, ``ssh_accepted_password``, ``sudo_command``,
``sudo_not_in_sudoers``, ``http_get`` / ``http_post`` / ``http_<method>``,
``logon_success``, ``logon_failure``, ``special_privileges_assigned``,
``process_creation``, ``user_account_created``, ``logoff`` and the cloud action
verbatim from the audit record (``console_login``, ``security_group_rule_added``,
``iam_policy_changed`` ...).
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote

from detector.schema import Event, SourceType

# Syslog lines carry no year; the mini-siem sample logs are from 2026.
YEAR = 2026
UTC = timezone.utc
_MAX_SAMPLE_LINES = 4

# Burst thresholds for knowledge-free aggregation. These shape units; they are
# not detection thresholds (labels come from corpus/build_corpus.py).
_SSH_BURST_MIN = 2          # >=2 same-IP SSH failures collapse into one unit
_WIN_FAILED_BURST_MIN = 5   # >=5 same host+IP 4625 events collapse into one unit
_WEB_LOGIN_BURST_MIN = 5    # >=5 POST /login 401 from one IP -> one auth-bf unit
_WEB_SCAN_BURST_MIN = 10    # >=10 requests from one IP with a scan signature

_ID_PREFIX = {
    SourceType.linux_auth: "lx",
    SourceType.nginx_access: "ng",
    SourceType.windows_security: "win",
    SourceType.cloud_audit: "cld",
}

_FILENAME_SOURCE = {
    "linux_auth": SourceType.linux_auth,
    "nginx_access": SourceType.nginx_access,
    "nginx_auth_bruteforce": SourceType.nginx_access,
    "windows_security": SourceType.windows_security,
    "cloud_audit": SourceType.cloud_audit,
}


# --------------------------------------------------------------------------- #
# Raw line parsing (regexes shared in spirit with corpus/build_corpus.py).
# --------------------------------------------------------------------------- #
_RE_SSH_FAILED = re.compile(
    r"^(\w+\s+\d+\s+[\d:]+)\s+(\S+)\s+sshd\[\d+\]:\s+Failed password for(?: invalid user)?\s+(\S+)\s+from\s+(\S+)"
)
_RE_SSH_INVALID = re.compile(
    r"^(\w+\s+\d+\s+[\d:]+)\s+(\S+)\s+sshd\[\d+\]:\s+Invalid user\s+(\S+)\s+from\s+(\S+)"
)
_RE_SSH_ACCEPTED = re.compile(
    r"^(\w+\s+\d+\s+[\d:]+)\s+(\S+)\s+sshd\[\d+\]:\s+Accepted password for\s+(\S+)\s+from\s+(\S+)"
)
_RE_SUDO_FAIL = re.compile(
    r"^(\w+\s+\d+\s+[\d:]+)\s+(\S+)\s+sudo\s*:\s+(\S+)\s+:.*NOT in sudoers"
)
_RE_SUDO = re.compile(
    r"^(\w+\s+\d+\s+[\d:]+)\s+(\S+)\s+sudo\s*:\s+(\S+)\s+:.*COMMAND=(.*)"
)
_RE_NGINX = re.compile(
    r'^(\S+)\s+-\s+(\S+)\s+\[([^\]]+)\]\s+"(\S+)\s+(\S+)\s+\S+"\s+(\d+)\s+\d+\s+"[^"]*"\s+"([^"]*)"'
)
_LINUX_TS_FMT = "%b %d %H:%M:%S"
_NGINX_TS_FMT = "%d/%b/%Y:%H:%M:%S %z"


def _linux_ts(ts_str: str) -> datetime:
    try:
        dt = datetime.strptime(ts_str.strip(), _LINUX_TS_FMT)
    except (ValueError, TypeError):
        # A malformed timestamp must not abort the whole batch (mirror _iso).
        return datetime.now(UTC)
    return dt.replace(year=YEAR, tzinfo=UTC)


def _nginx_ts(ts_str: str) -> datetime:
    try:
        return datetime.strptime(ts_str.strip(), _NGINX_TS_FMT).astimezone(UTC)
    except (ValueError, TypeError):
        return datetime.now(UTC)


def _iso(ts_raw: Any) -> datetime:
    try:
        return datetime.fromisoformat(str(ts_raw)).astimezone(UTC)
    except (ValueError, TypeError):
        return datetime.now(UTC)


# --------------------------------------------------------------------------- #
# Web payload classification (content-based, knowledge-free).
# --------------------------------------------------------------------------- #
_EXPLOIT_SIGS = [
    ("union select", "T1190"),
    ("or '1'='1", "T1190"),
    ("../", "T1190"),
    ("/etc/passwd", "T1190"),
    (";whoami", "T1190"),
    ("${jndi:", "T1190"),
]
_SCANNER_UA = ("sqlmap", "nikto", "nmap", "masscan", "dirbuster", "gobuster")


def classify_web_payload(path: str, user_agent: Optional[str]) -> Optional[str]:
    """Return an ATT&CK technique id if a request carries a known attack payload.

    Purely content-based (URL-decoded path and User-Agent). Returns ``None`` for
    ordinary traffic. Shared with the mock detector so both reason about the same
    signals.
    """
    decoded = unquote(path or "").lower()
    ua = (user_agent or "").lower()
    if "${jndi:" in ua:
        return "T1190"
    for needle, technique in _EXPLOIT_SIGS:
        if needle in decoded:
            return technique
    if any(s in ua for s in _SCANNER_UA):
        return "T1595.002"
    return None


def _looks_like_browser(ua: Optional[str]) -> bool:
    return (ua or "").lower().startswith("mozilla")


def _looks_like_login_path(path: str) -> bool:
    p = (path or "").lower()
    return any(k in p for k in ("login", "signin", "auth", "session"))


# --------------------------------------------------------------------------- #
# Sample-line helpers.
# --------------------------------------------------------------------------- #
def _samples(lines: list[str]) -> list[str]:
    if len(lines) <= _MAX_SAMPLE_LINES:
        return list(lines)
    kept = lines[: _MAX_SAMPLE_LINES - 1]
    return kept + [f"... (+{len(lines) - (_MAX_SAMPLE_LINES - 1)} more lines)"]


def _burst_window(times: list[datetime]) -> tuple[str, str]:
    return min(times).isoformat(), max(times).isoformat()


# --------------------------------------------------------------------------- #
# Event drafts (Event fields without id) -> finalized, id-numbered Events.
# --------------------------------------------------------------------------- #
def _finalize(source: SourceType, drafts: list[dict[str, Any]], id_start: int = 1) -> list[Event]:
    """Sort drafts deterministically and assign ``{prefix}-{NNNN}`` ids."""
    drafts.sort(key=lambda d: (d["timestamp"], d["action"], str(d.get("target"))))
    events: list[Event] = []
    n = id_start - 1
    for d in drafts:
        n += 1
        events.append(
            Event(
                id=f"{_ID_PREFIX[source]}-{n:04d}",
                source=source,
                timestamp=d["timestamp"],
                actor=d.get("actor"),
                action=d["action"],
                target=d.get("target"),
                status=d.get("status"),
                raw=d.get("raw", {}),
            )
        )
    return events


# --------------------------------------------------------------------------- #
# linux_auth
# --------------------------------------------------------------------------- #
def parse_linux_auth(text: str) -> list[Event]:
    """Parse a linux auth (syslog) blob into normalized Events."""
    fails_by_ip: dict[str, list[dict[str, Any]]] = {}
    point: list[dict[str, Any]] = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if (m := _RE_SSH_FAILED.match(line)) or (m := _RE_SSH_INVALID.match(line)):
            rec = {
                "ts": _linux_ts(m.group(1)),
                "host": m.group(2),
                "user": m.group(3),
                "ip": m.group(4),
                "line": line,
            }
            fails_by_ip.setdefault(rec["ip"], []).append(rec)
        elif m := _RE_SSH_ACCEPTED.match(line):
            point.append({
                "timestamp": _linux_ts(m.group(1)),
                "actor": m.group(4),
                "action": "ssh_accepted_password",
                "target": m.group(3),
                "status": "success",
                "raw": {"line": line, "source": "linux_auth"},
            })
        elif m := _RE_SUDO_FAIL.match(line):
            point.append({
                "timestamp": _linux_ts(m.group(1)),
                "actor": m.group(3),
                "action": "sudo_not_in_sudoers",
                "target": "root",
                "status": "failure",
                "raw": {"line": line, "source": "linux_auth"},
            })
        elif m := _RE_SUDO.match(line):
            cmd = m.group(4).strip()
            point.append({
                "timestamp": _linux_ts(m.group(1)),
                "actor": m.group(3),
                "action": "sudo_command",
                "target": cmd,
                "status": "success",
                "raw": {"line": line, "command": cmd, "source": "linux_auth"},
            })

    drafts: list[dict[str, Any]] = list(point)
    for ip, recs in fails_by_ip.items():
        recs.sort(key=lambda r: r["ts"])
        users = sorted({r["user"] for r in recs})
        start, end = _burst_window([r["ts"] for r in recs])
        drafts.append({
            "timestamp": recs[0]["ts"],
            "actor": ip,
            "action": "ssh_failed_password",
            "target": ", ".join(users[:5]) + (" ..." if len(users) > 5 else ""),
            "status": "failure",
            "raw": {
                "count": len(recs),
                "window_start": start,
                "window_end": end,
                "sample_lines": _samples([r["line"] for r in recs]),
                "distinct_users": users,
                "source": "linux_auth",
            },
        })
    return _finalize(SourceType.linux_auth, drafts)


# --------------------------------------------------------------------------- #
# nginx_access
# --------------------------------------------------------------------------- #
def _parse_nginx_records(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _RE_NGINX.match(line)
        if not m:
            continue
        records.append({
            "ip": m.group(1),
            "ts": _nginx_ts(m.group(3)),
            "method": m.group(4),
            "path": m.group(5),
            "status": m.group(6),
            "ua": m.group(7),
            "line": line,
        })
    return records


def _web_point(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp": r["ts"],
        "actor": r["ip"],
        "action": f"http_{r['method'].lower()}",
        "target": r["path"],
        "status": r["status"],
        "raw": {
            "line": r["line"],
            "path": r["path"],
            "user_agent": r["ua"],
            "source": "nginx_access",
        },
    }


def parse_nginx_access(text: str) -> list[Event]:
    """Parse an nginx access-log blob into normalized Events.

    Groups by client IP, then shapes each group by content:
    - an exploitation source (payload in path/UA) -> one unit per request, so
      each malicious request is judgeable on its own;
    - a login brute-force run (many POSTs to a login path) -> one aggregated unit
      with ``status_sequence`` / ``final_status``;
    - a scan / high-volume burst -> one aggregated unit with ``user_agents`` and
      ``sample_paths``;
    - everything else -> one unit per request.
    """
    records = _parse_nginx_records(text)
    by_ip: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_ip.setdefault(r["ip"], []).append(r)

    drafts: list[dict[str, Any]] = []
    for ip, recs in by_ip.items():
        recs.sort(key=lambda r: r["ts"])

        # Exploitation: any request carries a known payload -> per-request units.
        if any(classify_web_payload(r["path"], r["ua"]) for r in recs):
            drafts.extend(_web_point(r) for r in recs)
            continue

        # Web login brute force: a run of POSTs to a login path.
        login_posts = [
            r for r in recs
            if r["method"].upper() == "POST" and _looks_like_login_path(r["path"])
        ]
        if len(login_posts) >= _WEB_LOGIN_BURST_MIN:
            statuses = [r["status"] for r in login_posts]
            start, end = _burst_window([r["ts"] for r in login_posts])
            drafts.append({
                "timestamp": login_posts[0]["ts"],
                "actor": ip,
                "action": "http_post",
                "target": login_posts[0]["path"],
                "status": statuses[0],
                "raw": {
                    "count": len(login_posts),
                    "window_start": start,
                    "window_end": end,
                    "sample_lines": _samples([r["line"] for r in login_posts]),
                    "status_sequence": statuses,
                    "final_status": statuses[-1],
                    "source": "nginx_access",
                },
            })
            leftovers = [r for r in recs if r not in login_posts]
            drafts.extend(_web_point(r) for r in leftovers)
            continue

        # Scan / high-volume burst from a single client -> one aggregated unit.
        uas = sorted({r["ua"] for r in recs})
        scanner_ua = any(any(s in u.lower() for s in _SCANNER_UA) for u in uas)
        many_404 = len(recs) >= _WEB_SCAN_BURST_MIN and all(r["status"] == "404" for r in recs)
        if scanner_ua or many_404:
            paths = [r["path"] for r in recs]
            start, end = _burst_window([r["ts"] for r in recs])
            drafts.append({
                "timestamp": recs[0]["ts"],
                "actor": ip,
                "action": "http_get",
                "target": paths[0] if len(set(paths)) == 1 else f"{len(set(paths))} paths",
                "status": recs[-1]["status"],
                "raw": {
                    "count": len(recs),
                    "window_start": start,
                    "window_end": end,
                    "sample_lines": _samples([r["line"] for r in recs]),
                    "user_agents": uas,
                    "sample_paths": paths[:6],
                    "source": "nginx_access",
                },
            })
            continue

        # Ordinary browsing: one unit per request.
        drafts.extend(_web_point(r) for r in recs)

    return _finalize(SourceType.nginx_access, drafts)


# --------------------------------------------------------------------------- #
# windows_security
# --------------------------------------------------------------------------- #
def _parse_jsonl(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def parse_windows_security(text: str) -> list[Event]:
    """Parse a Windows Security event export (JSON lines) into Events.

    Event 4625 (failed logon) bursts from one host+source-IP collapse into a
    single aggregated unit; other events map one-to-one.
    """
    records = _parse_jsonl(text)
    drafts: list[dict[str, Any]] = []

    # 4625 failed logons, grouped by (host, source_ip).
    failed = [r for r in records if str(r.get("event_code")) == "4625"]
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in failed:
        by_key.setdefault((r.get("host", "?"), str(r.get("source_ip"))), []).append(r)
    for (host, ip), recs in by_key.items():
        if len(recs) >= _WIN_FAILED_BURST_MIN:
            users = sorted({r.get("username") for r in recs if r.get("username")})
            drafts.append({
                "timestamp": _iso(recs[0].get("timestamp")),
                "actor": recs[0].get("source_ip"),
                "action": "logon_failure",
                "target": ", ".join(users),
                "status": "failure",
                "raw": {
                    "count": len(recs),
                    "event_code": "4625",
                    "host": host,
                    "distinct_users": users,
                    "sample_lines": [json.dumps(r) for r in recs[:_MAX_SAMPLE_LINES]],
                    "source": "windows_security",
                },
            })
        else:
            for r in recs:
                drafts.append({
                    "timestamp": _iso(r.get("timestamp")),
                    "actor": r.get("source_ip") or r.get("username"),
                    "action": "logon_failure",
                    "target": r.get("host"),
                    "status": "failure",
                    "raw": {"event": r, "source": "windows_security"},
                })

    for r in records:
        code = str(r.get("event_code"))
        if code == "4624":
            drafts.append(_win_point(r, "logon_success", r.get("host"), "success"))
        elif code == "4634":
            drafts.append(_win_point(r, "logoff", r.get("host"), "success"))
        elif code == "4672":
            drafts.append(_win_point(r, "special_privileges_assigned", r.get("host"), "success"))
        elif code == "4688":
            target = r.get("process_name") or r.get("host")
            drafts.append(_win_point(r, "process_creation", target, "success"))
        elif code == "4720":
            target = r.get("new_account") or r.get("host")
            drafts.append(_win_point(r, "user_account_created", target, "success"))
    return _finalize(SourceType.windows_security, drafts)


def _win_point(r: dict[str, Any], action: str, target: Optional[str], status: str) -> dict[str, Any]:
    return {
        "timestamp": _iso(r.get("timestamp")),
        "actor": r.get("source_ip") or r.get("username"),
        "action": action,
        "target": target,
        "status": status,
        "raw": {"event": r, "source": "windows_security"},
    }


# --------------------------------------------------------------------------- #
# cloud_audit
# --------------------------------------------------------------------------- #
def parse_cloud_audit(text: str) -> list[Event]:
    """Parse a cloud control-plane audit blob (JSON lines) into Events.

    Console-login failures collapse into a single aggregated unit; every other
    control-plane action maps one-to-one, keeping its ``action`` verbatim.
    """
    records = _parse_jsonl(text)
    drafts: list[dict[str, Any]] = []

    login_failures = [
        r for r in records
        if r.get("action") == "console_login" and r.get("status") == "failure"
    ]
    if login_failures:
        drafts.append({
            "timestamp": _iso(login_failures[0].get("timestamp")),
            "actor": login_failures[0].get("source_ip"),
            "action": "console_login",
            "target": login_failures[0].get("username"),
            "status": "failure",
            "raw": {
                "count": len(login_failures),
                "events": login_failures,
                "source": "cloud_audit",
            },
        })

    for r in records:
        action = r.get("action")
        if action == "console_login" and r.get("status") == "failure":
            continue  # already aggregated above
        target: Optional[str]
        if action == "security_group_rule_added":
            target = f"port {r.get('port')} <- {r.get('cidr')}"
        elif action == "iam_policy_changed":
            target = r.get("target_user") or r.get("policy_name")
        else:
            target = r.get("username")
        drafts.append({
            "timestamp": _iso(r.get("timestamp")),
            "actor": r.get("source_ip") or r.get("username"),
            "action": action or "unknown",
            "target": target,
            "status": r.get("status", "unknown"),
            "raw": {"event": r, "source": "cloud_audit"},
        })
    return _finalize(SourceType.cloud_audit, drafts)


# --------------------------------------------------------------------------- #
# Dispatch and convenience loaders.
# --------------------------------------------------------------------------- #
_PARSERS = {
    SourceType.linux_auth: parse_linux_auth,
    SourceType.nginx_access: parse_nginx_access,
    SourceType.windows_security: parse_windows_security,
    SourceType.cloud_audit: parse_cloud_audit,
}


def normalize_source(source: SourceType, text: str) -> list[Event]:
    """Normalize a raw blob for a given :class:`SourceType` into Events."""
    source = SourceType(source)
    return _PARSERS[source](text)


def _infer_source(path: Path) -> SourceType:
    stem = path.stem
    if stem in _FILENAME_SOURCE:
        return _FILENAME_SOURCE[stem]
    for key, src in _FILENAME_SOURCE.items():
        if stem.startswith(key):
            return src
    raise ValueError(f"Cannot infer source type from filename: {path.name}")


def normalize_file(path: str | Path, source: Optional[SourceType] = None) -> list[Event]:
    """Normalize a single raw log file; infers the source from its name if omitted."""
    path = Path(path)
    if source is None:
        source = _infer_source(path)
    return normalize_source(source, path.read_text(encoding="utf-8"))


def normalize_corpus(raw_dir: str | Path) -> list[Event]:
    """Normalize every raw file under ``raw_dir`` into a flat list of Events.

    The two nginx files (``nginx_access.log`` and ``nginx_auth_bruteforce.log``)
    are both treated as the ``nginx_access`` source and their units are numbered
    together. Ids are local to this normalization and are not guaranteed to match
    the authoritative ids in ``corpus/labels.jsonl`` (whose units also include a
    synthetic benign background); use the corpus for scoring.
    """
    raw_dir = Path(raw_dir)
    events: list[Event] = []

    linux = raw_dir / "linux_auth.log"
    if linux.exists():
        events += normalize_source(SourceType.linux_auth, linux.read_text(encoding="utf-8"))

    nginx_text_parts = []
    for name in ("nginx_access.log", "nginx_auth_bruteforce.log"):
        f = raw_dir / name
        if f.exists():
            nginx_text_parts.append(f.read_text(encoding="utf-8"))
    if nginx_text_parts:
        events += normalize_source(SourceType.nginx_access, "\n".join(nginx_text_parts))

    win = raw_dir / "windows_security.jsonl"
    if win.exists():
        events += normalize_source(SourceType.windows_security, win.read_text(encoding="utf-8"))

    cloud = raw_dir / "cloud_audit.jsonl"
    if cloud.exists():
        events += normalize_source(SourceType.cloud_audit, cloud.read_text(encoding="utf-8"))

    return events
