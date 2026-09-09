"""Build the labeled corpus (``corpus/labels.jsonl``) from the raw logs.

Ground-truth labels come from *knowledge of the injected attack scenarios*, not
from running any detection rule. Concretely, this script contains an explicit
map of which source IPs / hosts / requests are attacks and which are benign
(``_INCIDENT_KNOWLEDGE`` below). It never imports ``rules/engine.py``. Counts are
recorded for context and may only decide how *benign* events are grouped; they
never turn a benign unit into an incident or vice versa. That independence is
what keeps the later rules-vs-LLM comparison honest: the rules cannot score a
fake-perfect result against labels they themselves produced.

Two things happen here:

1. The four raw mini-siem sample logs are parsed and collapsed into *analysis
   units*. A brute-force burst or a scan from one source becomes a single unit
   (its ``raw`` keeps the count, the time window and sample lines); discrete
   events stay one unit each. Each unit is labeled from the knowledge map.

2. A deterministic, obviously-benign background of everyday activity (normal
   logins, browsing, admin operations) is synthesised so the corpus has a
   realistic base rate: incidents are a small minority, as in a real SOC feed.
   A handful of benign "look-alikes" (a fat-finger login, an HR account
   creation with no preceding failures, a security group opened only on 443)
   are included so false positives can be measured.

Run:  ``python corpus/build_corpus.py``  (writes corpus/labels.jsonl).
The output is fully deterministic (fixed seed, fixed catalogs).
"""

from __future__ import annotations

import json
import random
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

# Make the repository root importable so `detector` resolves whether this file is
# run as a module (`python -m corpus.build_corpus`) or as a plain script
# (`python corpus/build_corpus.py`, which puts only corpus/ on sys.path).
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from detector.schema import Event, Label, LabeledUnit, SourceType

RAW_DIR = Path(__file__).resolve().parent / "raw"
OUT_PATH = Path(__file__).resolve().parent / "labels.jsonl"

YEAR = 2026
UTC = timezone.utc
_MAX_SAMPLE_LINES = 4


# --------------------------------------------------------------------------- #
# Ground-truth knowledge of the injected attacks (NOT derived from any rule).
# Keyed by the acting entity in each raw log. This is the human analyst's
# knowledge of the scenario that was injected into the mini-siem sample logs.
# --------------------------------------------------------------------------- #
_INCIDENT_KNOWLEDGE = {
    "linux_bruteforce_ips": {
        # source_ip -> (technique, rationale). Sustained password guessing.
        "192.0.2.100": (
            "T1110.001",
            "Sustained SSH password guessing: many consecutive failed logins "
            "against root/admin/oracle from one external IP.",
        ),
        "198.51.100.50": (
            "T1110.001",
            "Sustained SSH password guessing from one external IP cycling "
            "through root, admin and common service accounts.",
        ),
        "203.0.113.99": (
            "T1110.001",
            "High-volume SSH password guessing from one external IP; this IP "
            "then succeeds a login, so the burst is the run-up to compromise.",
        ),
    },
    # IPs whose eventual *successful* SSH login is the compromise itself.
    "linux_compromise_ips": {
        "203.0.113.99": (
            "T1078",
            "Successful SSH login from an IP that had just brute-forced the "
            "host: valid-account access obtained through credential guessing.",
        ),
    },
    "web_scan_ips": {
        "198.51.100.77": (
            "T1595.002",
            "Automated directory scan: high volume of 404s from a known "
            "scanner user-agent (gobuster).",
        ),
        "203.0.113.50": (
            "T1083",
            "Recon against sensitive paths (/.env, /.git, /phpmyadmin, /admin) "
            "from an automation user-agent (python-requests).",
        ),
    },
    "web_login_bruteforce_ips": {
        "203.0.113.77": (
            "T1110",
            "Web login brute force: many failed POST /login (HTTP 401) from one "
            "IP followed by a 200, i.e. credential guessing that succeeds.",
        ),
    },
    "windows_failed_logon_hosts": {
        # (host, source_ip) -> burst of 4625 that precedes account creation.
        ("WIN-SERVER01", "192.0.2.150"): (
            "T1110.001",
            "Burst of Windows failed logons (Event 4625) against "
            "administrator/admin from one source, preceding a backdoor account.",
        ),
    },
    "cloud_sensitive_ports": {22, 3389, 3306, 5432, 27017},
}


# --------------------------------------------------------------------------- #
# Raw log parsing.
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
_RE_SUDO = re.compile(
    r"^(\w+\s+\d+\s+[\d:]+)\s+(\S+)\s+sudo\s*:\s+(\S+)\s+:.*COMMAND=(.*)"
)
_RE_SUDO_FAIL = re.compile(
    r"^(\w+\s+\d+\s+[\d:]+)\s+(\S+)\s+sudo\s*:\s+(\S+)\s+:.*NOT in sudoers"
)
_RE_NGINX = re.compile(
    r'^(\S+)\s+-\s+(\S+)\s+\[([^\]]+)\]\s+"(\S+)\s+(\S+)\s+\S+"\s+(\d+)\s+\d+\s+"[^"]*"\s+"([^"]*)"'
)
_LINUX_TS_FMT = "%b %d %H:%M:%S"
_NGINX_TS_FMT = "%d/%b/%Y:%H:%M:%S %z"


def _linux_ts(ts_str: str) -> datetime:
    dt = datetime.strptime(ts_str.strip(), _LINUX_TS_FMT)
    return dt.replace(year=YEAR, tzinfo=UTC)


def _nginx_ts(ts_str: str) -> datetime:
    return datetime.strptime(ts_str.strip(), _NGINX_TS_FMT).astimezone(UTC)


def _parse_linux(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if (m := _RE_SSH_FAILED.match(line)) or (m := _RE_SSH_INVALID.match(line)):
            failed = bool(_RE_SSH_FAILED.match(line))
            records.append(
                {
                    "ts": _linux_ts(m.group(1)),
                    "host": m.group(2),
                    "user": m.group(3),
                    "ip": m.group(4),
                    "kind": "ssh_fail" if failed else "ssh_invalid",
                    "line": line,
                }
            )
        elif m := _RE_SSH_ACCEPTED.match(line):
            records.append(
                {
                    "ts": _linux_ts(m.group(1)),
                    "host": m.group(2),
                    "user": m.group(3),
                    "ip": m.group(4),
                    "kind": "ssh_accepted",
                    "line": line,
                }
            )
        elif m := _RE_SUDO_FAIL.match(line):
            records.append(
                {
                    "ts": _linux_ts(m.group(1)),
                    "host": m.group(2),
                    "user": m.group(3),
                    "ip": None,
                    "kind": "sudo_denied",
                    "line": line,
                }
            )
        elif m := _RE_SUDO.match(line):
            records.append(
                {
                    "ts": _linux_ts(m.group(1)),
                    "host": m.group(2),
                    "user": m.group(3),
                    "ip": None,
                    "kind": "sudo_command",
                    "cmd": m.group(4).strip(),
                    "line": line,
                }
            )
    return records


def _parse_nginx(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        m = _RE_NGINX.match(line)
        if not m:
            continue
        records.append(
            {
                "ip": m.group(1),
                "ts": _nginx_ts(m.group(3)),
                "method": m.group(4),
                "path": m.group(5),
                "status": m.group(6),
                "ua": m.group(7),
                "line": line,
            }
        )
    return records


def _parse_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


# --------------------------------------------------------------------------- #
# Unit drafts -> LabeledUnit. A draft is a plain dict; ids are assigned last so
# they are stable and ordered by (source, timestamp).
# --------------------------------------------------------------------------- #
def _draft(
    source: SourceType,
    ts: datetime,
    actor: Optional[str],
    action: str,
    target: Optional[str],
    status: Optional[str],
    raw: dict[str, Any],
    label: Label,
    technique: Optional[str],
    rationale: str,
) -> dict[str, Any]:
    return {
        "source": source,
        "ts": ts,
        "actor": actor,
        "action": action,
        "target": target,
        "status": status,
        "raw": raw,
        "label": label,
        "technique": technique,
        "rationale": rationale,
    }


def _samples(records: list[dict[str, Any]]) -> list[str]:
    lines = [r["line"] for r in records if "line" in r]
    if len(lines) <= _MAX_SAMPLE_LINES:
        return lines
    return lines[: _MAX_SAMPLE_LINES - 1] + [f"... (+{len(lines) - (_MAX_SAMPLE_LINES - 1)} more lines)"]


def _burst_raw(records: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    raw = {
        "count": len(records),
        "window_start": min(r["ts"] for r in records).isoformat(),
        "window_end": max(r["ts"] for r in records).isoformat(),
        "sample_lines": _samples(records),
    }
    raw.update(extra)
    return raw


# --------------------------------------------------------------------------- #
# Build units for each source from parsed records + knowledge map.
# --------------------------------------------------------------------------- #
def _build_linux(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    drafts: list[dict[str, Any]] = []
    bf_ips = _INCIDENT_KNOWLEDGE["linux_bruteforce_ips"]
    compromise_ips = _INCIDENT_KNOWLEDGE["linux_compromise_ips"]

    # Group SSH failure/invalid records by source IP.
    fails_by_ip: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        if r["kind"] in ("ssh_fail", "ssh_invalid"):
            fails_by_ip.setdefault(r["ip"], []).append(r)

    for ip, recs in fails_by_ip.items():
        recs.sort(key=lambda r: r["ts"])
        users = sorted({r["user"] for r in recs})
        if ip in bf_ips:
            technique, rationale = bf_ips[ip]
            drafts.append(
                _draft(
                    SourceType.linux_auth,
                    recs[0]["ts"],
                    ip,
                    "ssh_failed_password",
                    ", ".join(users[:5]) + (" ..." if len(users) > 5 else ""),
                    "failure",
                    _burst_raw(recs, distinct_users=users, source="linux_auth"),
                    Label.incident,
                    technique,
                    rationale,
                )
            )
        else:
            # Low-volume probing from an IP not in the knowledge map: benign
            # noise by our criterion (a couple of tries is not an incident).
            drafts.append(
                _draft(
                    SourceType.linux_auth,
                    recs[0]["ts"],
                    ip,
                    "ssh_failed_password",
                    ", ".join(users),
                    "failure",
                    _burst_raw(recs, distinct_users=users, source="linux_auth"),
                    Label.benign,
                    None,
                    f"Only {len(recs)} failed SSH attempts from this IP and it is "
                    "not a known attacker: below any reasonable incident bar "
                    "(a mistyped-username probe).",
                )
            )

    # Accepted logins.
    for r in records:
        if r["kind"] != "ssh_accepted":
            continue
        if r["ip"] in compromise_ips:
            technique, rationale = compromise_ips[r["ip"]]
            drafts.append(
                _draft(
                    SourceType.linux_auth,
                    r["ts"],
                    r["ip"],
                    "ssh_accepted_password",
                    r["user"],
                    "success",
                    {"line": r["line"], "source": "linux_auth"},
                    Label.incident,
                    technique,
                    rationale,
                )
            )
        else:
            drafts.append(
                _draft(
                    SourceType.linux_auth,
                    r["ts"],
                    r["ip"],
                    "ssh_accepted_password",
                    r["user"],
                    "success",
                    {"line": r["line"], "source": "linux_auth"},
                    Label.benign,
                    None,
                    "Successful interactive login by a known service account "
                    "from a trusted IP with no preceding guessing: routine.",
                )
            )

    # Sudo.
    for r in records:
        if r["kind"] == "sudo_command":
            drafts.append(
                _draft(
                    SourceType.linux_auth,
                    r["ts"],
                    r["user"],
                    "sudo_command",
                    r.get("cmd"),
                    "success",
                    {"line": r["line"], "command": r.get("cmd"), "source": "linux_auth"},
                    Label.benign,
                    None,
                    "Authorised sudo command by a normal operator account.",
                )
            )
        elif r["kind"] == "sudo_denied":
            drafts.append(
                _draft(
                    SourceType.linux_auth,
                    r["ts"],
                    r["user"],
                    "sudo_not_in_sudoers",
                    "root",
                    "failure",
                    {"line": r["line"], "source": "linux_auth"},
                    Label.benign,
                    None,
                    "A single denied sudo by an unknown user is suspicious but, "
                    "on its own, not enough evidence to call an incident "
                    "(documented boundary case).",
                )
            )
    return drafts


def _build_nginx_access(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    drafts: list[dict[str, Any]] = []
    scan_ips = _INCIDENT_KNOWLEDGE["web_scan_ips"]

    by_ip: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_ip.setdefault(r["ip"], []).append(r)

    for ip, recs in by_ip.items():
        recs.sort(key=lambda r: r["ts"])

        # Known scan / recon source: one aggregated incident unit.
        if ip in scan_ips:
            technique, rationale = scan_ips[ip]
            uas = sorted({r["ua"] for r in recs})
            paths = [r["path"] for r in recs]
            drafts.append(
                _draft(
                    SourceType.nginx_access,
                    recs[0]["ts"],
                    ip,
                    "http_get",
                    paths[0] if len(paths) == 1 else f"{len(set(paths))} paths",
                    recs[-1]["status"],
                    _burst_raw(
                        recs,
                        user_agents=uas,
                        sample_paths=paths[:6],
                        source="nginx_access",
                    ),
                    Label.incident,
                    technique,
                    rationale,
                )
            )
            continue

        # Known exploitation source (payloads in path/UA): one unit per request.
        exploit_units = [(r, _classify_exploit(r)) for r in recs]
        if any(kind for _, kind in exploit_units):
            for r, kind in exploit_units:
                if kind:
                    technique, rationale = kind
                    drafts.append(
                        _draft(
                            SourceType.nginx_access,
                            r["ts"],
                            ip,
                            f"http_{r['method'].lower()}",
                            r["path"],
                            r["status"],
                            {
                                "line": r["line"],
                                "path": r["path"],
                                "user_agent": r["ua"],
                                "source": "nginx_access",
                            },
                            Label.incident,
                            technique,
                            rationale,
                        )
                    )
                else:
                    drafts.append(_benign_web(r))
            continue

        # A large run of 404s from an ordinary browser user-agent on sequential
        # paths is a broken-link crawl, not a scan -> benign (documented). This
        # is deliberately independent of a naive "count 404s" rule.
        statuses = {r["status"] for r in recs}
        if len(recs) >= 20 and statuses == {"404"} and _looks_like_browser(recs[0]["ua"]):
            drafts.append(
                _draft(
                    SourceType.nginx_access,
                    recs[0]["ts"],
                    ip,
                    "http_get",
                    f"{len(recs)} sequential paths",
                    "404",
                    _burst_raw(recs, user_agents=sorted({r["ua"] for r in recs}), source="nginx_access"),
                    Label.benign,
                    None,
                    "High volume of 404s but from an ordinary browser user-agent "
                    "on sequential /pageN paths: a broken-link crawl, not a scan "
                    "(documented boundary case where a pure count rule over-fires).",
                )
            )
            continue

        # Everything else: normal browsing, one benign unit per request.
        for r in recs:
            drafts.append(_benign_web(r))

    return drafts


def _looks_like_browser(ua: str) -> bool:
    return ua.lower().startswith("mozilla")


_EXPLOIT_SIGS = [
    ("union select", "T1190", "SQL injection payload (UNION SELECT) in the request path."),
    ("or '1'='1", "T1190", "SQL injection payload (boolean OR 1=1) in the request path."),
    ("../", "T1190", "Path traversal payload (../ sequence targeting /etc/passwd)."),
    ("/etc/passwd", "T1190", "Path traversal targeting /etc/passwd."),
    (";whoami", "T1190", "OS command injection payload (; whoami) in the request path."),
    ("${jndi:", "T1190", "Log4Shell / JNDI lookup payload (CVE-2021-44228)."),
]


def _classify_exploit(r: dict[str, Any]) -> Optional[tuple[str, str]]:
    """Return (technique, rationale) if the request carries a known payload."""
    from urllib.parse import unquote

    decoded = unquote(r["path"]).lower()
    ua = (r["ua"] or "").lower()
    if "${jndi:" in ua:
        return ("T1190", "Log4Shell / JNDI lookup payload delivered via the User-Agent header.")
    for needle, technique, why in _EXPLOIT_SIGS:
        if needle in decoded:
            return (technique, why)
    if any(s in ua for s in ("sqlmap", "nikto", "nmap", "masscan", "dirbuster")):
        return ("T1595.002", f"Request from a known scanner user-agent ({r['ua']}).")
    return None


def _benign_web(r: dict[str, Any]) -> dict[str, Any]:
    return _draft(
        SourceType.nginx_access,
        r["ts"],
        r["ip"],
        f"http_{r['method'].lower()}",
        r["path"],
        r["status"],
        {"line": r["line"], "path": r["path"], "user_agent": r["ua"], "source": "nginx_access"},
        Label.benign,
        None,
        "Ordinary web request from a normal client: benign traffic.",
    )


def _build_nginx_bruteforce(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    drafts: list[dict[str, Any]] = []
    bf_ips = _INCIDENT_KNOWLEDGE["web_login_bruteforce_ips"]

    by_ip: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_ip.setdefault(r["ip"], []).append(r)

    for ip, recs in by_ip.items():
        recs.sort(key=lambda r: r["ts"])
        if ip in bf_ips:
            technique, rationale = bf_ips[ip]
            statuses = [r["status"] for r in recs]
            drafts.append(
                _draft(
                    SourceType.nginx_access,
                    recs[0]["ts"],
                    ip,
                    "http_post",
                    recs[0]["path"],
                    "401",
                    _burst_raw(
                        recs,
                        status_sequence=statuses,
                        final_status=statuses[-1],
                        source="nginx_access",
                    ),
                    Label.incident,
                    technique,
                    rationale,
                )
            )
        else:
            for r in recs:
                drafts.append(_benign_web(r))
    return drafts


def _build_windows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    drafts: list[dict[str, Any]] = []
    failed_hosts = _INCIDENT_KNOWLEDGE["windows_failed_logon_hosts"]

    failed = [r for r in records if str(r.get("event_code")) == "4625"]
    if failed:
        by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for r in failed:
            by_key.setdefault((r.get("host", "?"), r.get("source_ip", "?")), []).append(r)
        for (host, ip), recs in by_key.items():
            if (host, ip) in failed_hosts:
                technique, rationale = failed_hosts[(host, ip)]
                users = sorted({r.get("username") for r in recs if r.get("username")})
                drafts.append(
                    _draft(
                        SourceType.windows_security,
                        _iso(recs[0]["timestamp"]),
                        ip,
                        "logon_failure",
                        ", ".join(users),
                        "failure",
                        {
                            "count": len(recs),
                            "event_code": "4625",
                            "host": host,
                            "distinct_users": users,
                            "sample_lines": [json.dumps(r) for r in recs[:_MAX_SAMPLE_LINES]],
                            "source": "windows_security",
                        },
                        Label.incident,
                        technique,
                        rationale,
                    )
                )
            else:
                for r in recs:
                    drafts.append(_benign_windows(r, "logon_failure",
                                                   "Isolated Windows logon failure: a mistyped password, not an incident."))

    for r in records:
        code = str(r.get("event_code"))
        if code == "4624":
            drafts.append(_benign_windows(r, "logon_success",
                                          "Successful Windows logon by a normal user from a known host."))
        elif code == "4672":
            drafts.append(_benign_windows(r, "special_privileges_assigned",
                                          "Special privileges assigned to an administrative logon: routine for admins."))
        elif code == "4688":
            cmd = (r.get("command_line") or "").lower()
            if "-enc" in cmd or "-encodedcommand" in cmd:
                drafts.append(_incident_windows(
                    r, "process_creation", "T1059.001",
                    "Encoded PowerShell command line (-enc): obfuscated script execution."))
            elif "certutil" in cmd and ("urlcache" in cmd or "http" in cmd):
                drafts.append(_incident_windows(
                    r, "process_creation", "T1105",
                    "certutil used to download a remote executable: ingress tool transfer via a living-off-the-land binary."))
            else:
                drafts.append(_benign_windows(r, "process_creation",
                                              "Ordinary process creation."))
        elif code == "4720":
            drafts.append(_incident_windows(
                r, "user_account_created", "T1136.001",
                "New local account created on a host that had just seen a logon-failure burst: persistence backdoor."))
    return drafts


def _iso(ts_raw: str) -> datetime:
    try:
        return datetime.fromisoformat(ts_raw).astimezone(UTC)
    except (ValueError, TypeError):
        return datetime.now(UTC)


def _benign_windows(r: dict[str, Any], action: str, rationale: str) -> dict[str, Any]:
    return _draft(
        SourceType.windows_security,
        _iso(r.get("timestamp", "")),
        r.get("source_ip") or r.get("username"),
        action,
        r.get("host"),
        "success" if action != "logon_failure" else "failure",
        {"event": r, "source": "windows_security"},
        Label.benign,
        None,
        rationale,
    )


def _incident_windows(r: dict[str, Any], action: str, technique: str, rationale: str) -> dict[str, Any]:
    return _draft(
        SourceType.windows_security,
        _iso(r.get("timestamp", "")),
        r.get("source_ip") or r.get("username"),
        action,
        r.get("new_account") or r.get("process_name") or r.get("host"),
        "success",
        {"event": r, "source": "windows_security"},
        Label.incident,
        technique,
        rationale,
    )


def _build_cloud(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    drafts: list[dict[str, Any]] = []
    sensitive_ports = _INCIDENT_KNOWLEDGE["cloud_sensitive_ports"]

    # Identify identities with recent console-login failures (knowledge that an
    # IAM change from such an identity is the injected escalation scenario).
    failed_identities = {
        r.get("username")
        for r in records
        if r.get("action") == "console_login" and r.get("status") == "failure"
    }

    login_failures = [
        r for r in records
        if r.get("action") == "console_login" and r.get("status") == "failure"
    ]
    if login_failures:
        drafts.append(
            _draft(
                SourceType.cloud_audit,
                _iso(login_failures[0].get("timestamp", "")),
                login_failures[0].get("source_ip"),
                "console_login",
                login_failures[0].get("username"),
                "failure",
                {
                    "count": len(login_failures),
                    "events": login_failures,
                    "source": "cloud_audit",
                },
                Label.benign,
                None,
                f"Only {len(login_failures)} failed console logins: on their own, "
                "below the bar for an incident (documented boundary case).",
            )
        )

    for r in records:
        action = r.get("action")
        if action == "console_login" and r.get("status") == "success":
            drafts.append(_benign_cloud(r, "Successful console login by a known operator from a trusted IP."))
        elif action == "api_key_created":
            drafts.append(_benign_cloud(r, "API key created by a normal operator: routine automation setup."))
        elif action == "security_group_rule_added":
            port = r.get("port")
            cidr = r.get("cidr")
            if cidr == "0.0.0.0/0" and port in sensitive_ports:
                drafts.append(
                    _draft(
                        SourceType.cloud_audit,
                        _iso(r.get("timestamp", "")),
                        r.get("source_ip"),
                        "security_group_rule_added",
                        f"port {port} <- {cidr}",
                        "success",
                        {"event": r, "source": "cloud_audit"},
                        Label.incident,
                        "T1562.007",
                        f"Security group opened to the whole internet (0.0.0.0/0) on a "
                        f"sensitive port ({port}): weakening network defences.",
                    )
                )
            else:
                drafts.append(_benign_cloud(
                    r, f"Security group change on port {port} to {cidr}: not an internet-wide "
                       "opening of a sensitive port, so benign by our criterion."))
        elif action == "iam_policy_changed":
            if r.get("username") in failed_identities:
                drafts.append(
                    _draft(
                        SourceType.cloud_audit,
                        _iso(r.get("timestamp", "")),
                        r.get("source_ip"),
                        "iam_policy_changed",
                        r.get("target_user") or r.get("policy_name"),
                        "success",
                        {"event": r, "source": "cloud_audit"},
                        Label.incident,
                        "T1098",
                        "IAM policy changed by an identity that had recent console-login "
                        "failures: privilege escalation / account-takeover follow-through.",
                    )
                )
            else:
                drafts.append(_benign_cloud(
                    r, "IAM policy change by an identity with no preceding auth failures: "
                       "routine administration."))
    return drafts


def _benign_cloud(r: dict[str, Any], rationale: str) -> dict[str, Any]:
    return _draft(
        SourceType.cloud_audit,
        _iso(r.get("timestamp", "")),
        r.get("source_ip") or r.get("username"),
        r.get("action", "unknown"),
        r.get("username"),
        r.get("status", "unknown"),
        {"event": r, "source": "cloud_audit"},
        Label.benign,
        None,
        rationale,
    )


# --------------------------------------------------------------------------- #
# Synthetic benign background: deterministic everyday activity so the corpus
# has a realistic base rate (incidents a small minority). Everything here is
# benign by construction; a few benign look-alikes are included on purpose.
# --------------------------------------------------------------------------- #
_BASE = datetime(YEAR, 1, 5, 7, 0, 0, tzinfo=UTC)


def _synthetic_benign(rng: random.Random) -> list[dict[str, Any]]:
    drafts: list[dict[str, Any]] = []

    # --- linux_auth: normal logins, sudo, occasional fat-finger login ---
    users = ["deploy", "jsmith", "ops", "ci-runner", "mkuznetsov", "backup"]
    hosts = ["web-01", "web-02", "db-01", "app-01"]
    office_ips = ["203.0.113.5", "203.0.113.6", "198.51.100.10", "192.0.2.20", "10.0.4.11"]
    sudo_cmds = [
        "/usr/bin/systemctl status nginx",
        "/usr/bin/apt-get update",
        "/usr/bin/journalctl -u app",
        "/usr/bin/docker ps",
        "/bin/systemctl restart app",
    ]
    for i in range(58):
        ts = _BASE + timedelta(minutes=rng.randint(0, 60 * 24 * 8))
        roll = rng.random()
        user = rng.choice(users)
        host = rng.choice(hosts)
        ip = rng.choice(office_ips)
        if roll < 0.5:
            drafts.append(_draft(
                SourceType.linux_auth, ts, ip, "ssh_accepted_password", user, "success",
                {"line": f"{host} sshd: Accepted password for {user} from {ip}", "source": "linux_auth", "synthetic": True},
                Label.benign, None, "Routine successful SSH login by a staff account."))
        elif roll < 0.85:
            cmd = rng.choice(sudo_cmds)
            drafts.append(_draft(
                SourceType.linux_auth, ts, user, "sudo_command", cmd, "success",
                {"line": f"{host} sudo: {user} : COMMAND={cmd}", "command": cmd, "source": "linux_auth", "synthetic": True},
                Label.benign, None, "Authorised sudo command by an operator."))
        else:
            # Benign look-alike: a couple of failures then a success (fat finger).
            n = rng.randint(2, 4)
            drafts.append(_draft(
                SourceType.linux_auth, ts, ip, "ssh_failed_password", user, "failure",
                {"count": n, "line": f"{host} sshd: Failed password for {user} from {ip}",
                 "distinct_users": [user], "source": "linux_auth", "synthetic": True},
                Label.benign, None,
                f"{n} failed SSH logins for a single real account from an office IP then a success: "
                "a mistyped password, not an attack (benign look-alike)."))

    # --- nginx_access: normal browsing + health checks + a few benign 404s ---
    good_paths = ["/", "/about", "/pricing", "/docs", "/blog", "/api/health",
                  "/static/app.css", "/static/app.js", "/favicon.ico", "/dashboard"]
    client_ips = ["203.0.113.5", "203.0.113.60", "198.51.100.22", "192.0.2.30",
                  "203.0.113.80", "198.51.100.44", "10.0.4.9"]
    browsers = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Mozilla/5.0 (X11; Linux x86_64)",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)",
    ]
    for i in range(92):
        ts = _BASE + timedelta(seconds=rng.randint(0, 60 * 60 * 24 * 8))
        ip = rng.choice(client_ips)
        roll = rng.random()
        if roll < 0.08:
            # benign 404 (broken link), single request, normal browser
            path = f"/old/{rng.choice(['news', 'promo', 'v1', 'help'])}-{rng.randint(1, 9)}"
            drafts.append(_draft(
                SourceType.nginx_access, ts, ip, "http_get", path, "404",
                {"line": f'{ip} "GET {path}" 404', "path": path, "user_agent": rng.choice(browsers),
                 "source": "nginx_access", "synthetic": True},
                Label.benign, None, "Single 404 from a normal browser: a broken link, not a scan."))
        elif roll < 0.15:
            drafts.append(_draft(
                SourceType.nginx_access, ts, "10.0.4.9", "http_get", "/healthz", "200",
                {"line": '10.0.4.9 "GET /healthz" 200', "path": "/healthz", "user_agent": "kube-probe/1.29",
                 "source": "nginx_access", "synthetic": True},
                Label.benign, None, "Internal uptime/health-check probe."))
        else:
            path = rng.choice(good_paths)
            status = rng.choice(["200", "200", "200", "304", "301"])
            drafts.append(_draft(
                SourceType.nginx_access, ts, ip, "http_get", path, status,
                {"line": f'{ip} "GET {path}" {status}', "path": path, "user_agent": rng.choice(browsers),
                 "source": "nginx_access", "synthetic": True},
                Label.benign, None, "Ordinary page/asset request from a normal browser."))

    # --- windows_security: logons, logoffs, single failures, HR account create ---
    win_users = ["jsmith", "apetrov", "svc_backup", "helpdesk", "mkuznetsov"]
    win_hosts = ["WIN-SERVER01", "WIN-WKS07", "WIN-WKS12", "WIN-DC01"]
    for i in range(26):
        ts = _BASE + timedelta(minutes=rng.randint(0, 60 * 24 * 8))
        user = rng.choice(win_users)
        host = rng.choice(win_hosts)
        roll = rng.random()
        if roll < 0.55:
            drafts.append(_draft(
                SourceType.windows_security, ts, "10.0.4.15", "logon_success", host, "success",
                {"event": {"event_code": "4624", "host": host, "username": user}, "source": "windows_security", "synthetic": True},
                Label.benign, None, "Routine successful Windows logon."))
        elif roll < 0.8:
            drafts.append(_draft(
                SourceType.windows_security, ts, "10.0.4.15", "logoff", host, "success",
                {"event": {"event_code": "4634", "host": host, "username": user}, "source": "windows_security", "synthetic": True},
                Label.benign, None, "Windows logoff event."))
        elif roll < 0.92:
            drafts.append(_draft(
                SourceType.windows_security, ts, "10.0.4.16", "logon_failure", host, "failure",
                {"event": {"event_code": "4625", "host": host, "username": user}, "source": "windows_security", "synthetic": True},
                Label.benign, None, "Isolated Windows logon failure: a mistyped password."))
        else:
            # Benign look-alike: legit account creation by admin, no preceding failures.
            new_acct = f"new.hire{rng.randint(1, 40)}"
            drafts.append(_draft(
                SourceType.windows_security, ts, "helpdesk", "user_account_created", new_acct, "success",
                {"event": {"event_code": "4720", "host": host, "username": "helpdesk", "new_account": new_acct},
                 "source": "windows_security", "synthetic": True},
                Label.benign, None,
                "New account created by the help-desk with no preceding logon-failure burst: "
                "routine onboarding (benign look-alike for the backdoor scenario)."))

    # --- cloud_audit: logins, describe/list, private SG change, clean IAM change ---
    cloud_users = ["devops@example.com", "sre@example.com", "cfoulis@example.com"]
    cloud_ips = ["203.0.113.10", "203.0.113.11", "198.51.100.60"]
    regions = ["eu-central-1", "us-east-1", "eu-west-1"]
    read_actions = ["describe_instances", "list_buckets", "get_caller_identity", "describe_security_groups"]
    for i in range(18):
        ts = _BASE + timedelta(minutes=rng.randint(0, 60 * 24 * 8))
        user = rng.choice(cloud_users)
        ip = rng.choice(cloud_ips)
        region = rng.choice(regions)
        roll = rng.random()
        if roll < 0.4:
            drafts.append(_draft(
                SourceType.cloud_audit, ts, ip, "console_login", user, "success",
                {"event": {"action": "console_login", "status": "success", "username": user,
                           "source_ip": ip, "region": region}, "source": "cloud_audit", "synthetic": True},
                Label.benign, None, "Routine successful console login."))
        elif roll < 0.8:
            action = rng.choice(read_actions)
            drafts.append(_draft(
                SourceType.cloud_audit, ts, ip, action, user, "success",
                {"event": {"action": action, "status": "success", "username": user,
                           "source_ip": ip, "region": region}, "source": "cloud_audit", "synthetic": True},
                Label.benign, None, "Read-only cloud API call: routine operations."))
        elif roll < 0.9:
            # Benign look-alike: SG change but on 443 or to a private CIDR.
            port, cidr = rng.choice([(443, "0.0.0.0/0"), (22, "10.0.0.0/8"), (5432, "10.0.5.0/24")])
            drafts.append(_draft(
                SourceType.cloud_audit, ts, ip, "security_group_rule_added", f"port {port} <- {cidr}", "success",
                {"event": {"action": "security_group_rule_added", "status": "success", "username": user,
                           "source_ip": ip, "region": region, "port": port, "cidr": cidr},
                 "source": "cloud_audit", "synthetic": True},
                Label.benign, None,
                f"Security group change on port {port} to {cidr}: either a web port or a private range, "
                "not an internet-wide opening of a remote-access/DB port (benign look-alike)."))
        else:
            # Benign look-alike: IAM change with no preceding failures.
            drafts.append(_draft(
                SourceType.cloud_audit, ts, ip, "iam_policy_changed", "svc_deploy", "success",
                {"event": {"action": "iam_policy_changed", "status": "success", "username": user,
                           "source_ip": ip, "region": region, "policy_name": "DeployReadOnly",
                           "change_type": "policy_attached", "target_user": "svc_deploy"},
                 "source": "cloud_audit", "synthetic": True},
                Label.benign, None,
                "IAM policy attached by an operator with no preceding auth failures: routine (benign look-alike)."))

    return drafts


# --------------------------------------------------------------------------- #
# Assemble and write.
# --------------------------------------------------------------------------- #
_ID_PREFIX = {
    SourceType.linux_auth: "lx",
    SourceType.nginx_access: "ng",
    SourceType.windows_security: "win",
    SourceType.cloud_audit: "cld",
}


def build() -> list[LabeledUnit]:
    linux = _parse_linux(RAW_DIR / "linux_auth.log")
    nginx = _parse_nginx(RAW_DIR / "nginx_access.log")
    nginx_bf = _parse_nginx(RAW_DIR / "nginx_auth_bruteforce.log")
    windows = _parse_jsonl(RAW_DIR / "windows_security.jsonl")
    cloud = _parse_jsonl(RAW_DIR / "cloud_audit.jsonl")

    drafts: list[dict[str, Any]] = []
    drafts += _build_linux(linux)
    drafts += _build_nginx_access(nginx)
    drafts += _build_nginx_bruteforce(nginx_bf)
    drafts += _build_windows(windows)
    drafts += _build_cloud(cloud)

    rng = random.Random(20260909)
    drafts += _synthetic_benign(rng)

    # Stable ordering + ids: by (source, timestamp), numbered per source.
    drafts.sort(key=lambda d: (d["source"].value, d["ts"], d["action"], str(d["target"])))
    counters: dict[SourceType, int] = {}
    units: list[LabeledUnit] = []
    for d in drafts:
        src = d["source"]
        counters[src] = counters.get(src, 0) + 1
        unit_id = f"{_ID_PREFIX[src]}-{counters[src]:04d}"
        event = Event(
            id=unit_id,
            source=src,
            timestamp=d["ts"],
            actor=d["actor"],
            action=d["action"],
            target=d["target"],
            status=d["status"],
            raw=d["raw"],
        )
        units.append(
            LabeledUnit(
                unit_id=unit_id,
                event=event,
                label=d["label"],
                technique=d["technique"],
                rationale=d["rationale"],
            )
        )
    return units


def main() -> None:
    units = build()
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for u in units:
            f.write(json.dumps(u.model_dump(mode="json"), ensure_ascii=False) + "\n")

    incidents = sum(1 for u in units if u.is_incident)
    benign = len(units) - incidents
    by_source: dict[str, int] = {}
    for u in units:
        by_source[u.event.source.value] = by_source.get(u.event.source.value, 0) + 1
    print(f"Wrote {len(units)} labeled units to {OUT_PATH}")
    print(f"  incidents: {incidents} ({incidents / len(units):.1%})   benign: {benign}")
    print(f"  by source: {by_source}")


if __name__ == "__main__":
    main()
