"""LLM-based incident detector: prompt -> LangChain -> provider -> Verdict.

The pipeline for one :class:`Event` is:

1. Load a versioned prompt template from ``prompts/`` (``v1``/``v2``/``v3``).
2. Render the event into a compact, deterministic input block and substitute it
   into the template's ``{event}`` slot.
3. Send the prompt through a provider and get raw text back.
4. Parse the text leniently into a :class:`Verdict` (never raises on malformed
   model output — it degrades to a low-confidence benign verdict instead).

Two providers are supported:

- ``"ollama"`` — the real detector. Talks to a local `Ollama <https://ollama.com>`_
  model (default ``gemma3:latest``) via LangChain's ``ChatOllama`` at
  ``temperature=0``. Requires ``ollama pull gemma3:latest`` and the
  ``langchain-ollama`` package.
- ``"mock"`` — a deterministic, offline heuristic used by the tests and CI so the
  whole pipeline (prompt building, caching, parsing) can run without a model.
  **Mock output is NOT a real model result** and must never be reported as one:
  every mock reason is prefixed with ``[mock]`` for exactly this reason.

Because an LLM is not fully deterministic even at ``temperature=0``, responses are
cached on disk under ``corpus/llm_cache/`` keyed by a hash of
``(provider, model, prompt version, event id, full prompt text)``. The cache is
committed so the results table reproduces offline; ``refresh=True`` recomputes and
overwrites it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Optional

from detector.normalize import classify_web_payload
from detector.schema import Event, Prediction, Verdict

_REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = _REPO_ROOT / "prompts"
CACHE_DIR = _REPO_ROOT / "corpus" / "llm_cache"
DEFAULT_MODEL = "gemma3:latest"
DEFAULT_PROVIDER = "ollama"

_MOCK_NOTE = (
    "MOCK provider output — a deterministic offline heuristic, NOT a real model "
    "result. For tests/CI only; never report these numbers as LLM results."
)
_SENSITIVE_PORTS = {22, 3389, 3306, 5432, 27017}
_SENSITIVE_PATHS = ("/.env", "/.git", "/phpmyadmin", "/admin", "/wp-admin", "/config")
_SCANNER_UA = ("sqlmap", "nikto", "nmap", "masscan", "dirbuster", "gobuster")


# --------------------------------------------------------------------------- #
# Prompt loading and event rendering.
# --------------------------------------------------------------------------- #
_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def load_prompt(version: str, prompts_dir: Path | str = PROMPTS_DIR) -> str:
    """Load ``prompts/<version>.md`` and strip the leading changelog comment."""
    version = version if version.startswith("v") else f"v{version}"
    path = Path(prompts_dir) / f"{version}.md"
    text = path.read_text(encoding="utf-8")
    return _COMMENT_RE.sub("", text).strip()


# Raw keys surfaced to the model, in a fixed order for stable prompts.
_EVIDENCE_KEYS = [
    ("count", "count"),
    ("distinct_users", "targets"),
    ("distinct_targets", "targets"),
    ("status_sequence", "status_sequence"),
    ("final_status", "final_status"),
    ("user_agents", "user_agents"),
    ("sample_paths", "sample_paths"),
    ("path", "path"),
    ("user_agent", "user_agent"),
    ("command", "command"),
]
# Sub-fields lifted out of a nested raw["event"] object (Windows / cloud units).
_NESTED_KEYS = [
    ("event_code", "event_code"),
    ("command_line", "command_line"),
    ("process_name", "process_name"),
    ("new_account", "new_account"),
    ("port", "port"),
    ("cidr", "cidr"),
    ("protocol", "protocol"),
    ("policy_name", "policy_name"),
    ("change_type", "change_type"),
    ("target_user", "target_user"),
    ("failure_reason", "failure_reason"),
    ("description", "description"),
]


def _fmt_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        items = [str(v) for v in value]
        if len(items) > 6:
            items = items[:6] + [f"... (+{len(value) - 6} more)"]
        return ", ".join(items)
    return str(value)


def render_event(event: Event) -> str:
    """Render an :class:`Event` into a compact, deterministic prompt input block."""
    lines: list[str] = [
        f"source: {event.source.value}",
        f"action: {event.action}",
        f"actor: {event.actor if event.actor is not None else 'unknown'}",
        f"target: {event.target if event.target is not None else 'unknown'}",
        f"status: {event.status if event.status is not None else 'unknown'}",
    ]
    raw = event.raw or {}

    ws, we = raw.get("window_start"), raw.get("window_end")
    if ws and we:
        lines.append(f"window: {ws} .. {we}")

    seen: set[str] = set()
    for key, label in _EVIDENCE_KEYS:
        if key in raw and raw[key] not in (None, "", [], {}) and label not in seen:
            lines.append(f"{label}: {_fmt_value(raw[key])}")
            seen.add(label)

    nested = raw.get("event")
    if isinstance(nested, dict):
        for key, label in _NESTED_KEYS:
            if key in nested and nested[key] not in (None, "") and label not in seen:
                lines.append(f"{label}: {_fmt_value(nested[key])}")
                seen.add(label)

    samples = raw.get("sample_lines")
    if isinstance(samples, list) and samples:
        lines.append("sample_lines:")
        lines.extend(f"  - {s}" for s in samples)
    elif raw.get("line"):
        lines.append(f"sample_line: {raw['line']}")

    return "\n".join(lines)


def build_prompt(template: str, event: Event) -> str:
    """Substitute a rendered event into a prompt template's ``{event}`` slot."""
    rendered = render_event(event)
    if "{event}" in template:
        return template.replace("{event}", rendered)
    return f"{template}\n\n{rendered}"


# --------------------------------------------------------------------------- #
# Lenient parsing of model output into a Verdict.
# --------------------------------------------------------------------------- #
_TRUE_WORDS = {"true", "yes", "y", "1", "incident", "malicious", "attack", "suspicious"}
_FALSE_WORDS = {"false", "no", "n", "0", "benign", "normal", "clean", "none"}
_CONF_WORDS = {"high": 0.9, "medium": 0.6, "med": 0.6, "moderate": 0.6, "low": 0.3, "none": 0.0}


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        v = value.strip().lower()
        if v in _TRUE_WORDS:
            return True
        if v in _FALSE_WORDS:
            return False
    return None


def _coerce_confidence(value: Any) -> float:
    if isinstance(value, bool):
        return 0.9 if value else 0.1
    if isinstance(value, (int, float)):
        f = float(value)
    elif isinstance(value, str):
        v = value.strip().lower().rstrip("%")
        if v in _CONF_WORDS:
            return _CONF_WORDS[v]
        try:
            f = float(v)
        except ValueError:
            return 0.5
    else:
        return 0.5
    if f > 1.0:  # a percentage like 90 or an out-of-range guess
        f = f / 100.0 if f <= 100.0 else 1.0
    return max(0.0, min(1.0, f))


def _extract_json_object(text: str) -> Optional[str]:
    """Return the first balanced ``{...}`` block, ignoring braces inside strings."""
    depth = 0
    start = -1
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    return text[start : i + 1]
    return None


def parse_verdict(text: str) -> Verdict:
    """Parse raw model text into a :class:`Verdict`, degrading gracefully.

    Handles valid JSON, JSON wrapped in prose or code fences, and partial or
    malformed output. It never raises: unparseable text becomes a low-confidence
    benign verdict whose reason quotes what came back.
    """
    if text is None:
        text = ""
    raw = str(text).strip()
    snippet = _extract_json_object(raw)

    data: dict[str, Any] = {}
    if snippet is not None:
        try:
            parsed = json.loads(snippet)
            if isinstance(parsed, dict):
                data = parsed
        except json.JSONDecodeError:
            data = {}

    if data:
        is_incident = _coerce_bool(data.get("is_incident"))
        if is_incident is None:
            is_incident = _coerce_bool(data.get("incident"))
        technique = data.get("technique")
        confidence = _coerce_confidence(data.get("confidence"))
        reason = data.get("reason") or data.get("explanation") or ""
        if is_incident is None:
            # JSON present but no usable verdict flag: fall back to keyword scan.
            is_incident = _keyword_incident(raw)
            confidence = min(confidence, 0.5)
        if not is_incident:
            technique = None
        return Verdict(
            is_incident=is_incident,
            technique=technique,
            confidence=confidence,
            reason=str(reason)[:400] or "no reason provided",
        )

    # No JSON at all: last-resort keyword degradation, clearly low confidence.
    is_incident = _keyword_incident(raw)
    return Verdict(
        is_incident=is_incident,
        technique=None,
        confidence=0.3 if is_incident else 0.2,
        reason=("unparseable model output; " + (raw[:200] or "empty response")),
    )


def _keyword_incident(text: str) -> bool:
    low = text.lower()
    if any(w in low for w in ("not an incident", "benign", "no incident", "false")):
        return False
    return any(w in low for w in ("incident", "malicious", "attack", "brute", "exploit"))


# --------------------------------------------------------------------------- #
# Providers.
# --------------------------------------------------------------------------- #
class MockProvider:
    """Deterministic offline heuristic standing in for a model. NOT real results."""

    name = "mock"

    def generate(self, prompt: str, event: Optional[Event] = None) -> str:
        if event is None:
            return json.dumps({
                "is_incident": False, "technique": None, "confidence": 0.5,
                "reason": "[mock] no event provided",
            })
        return json.dumps(_mock_verdict_dict(event))


class OllamaProvider:
    """Real detector: a local Ollama model through LangChain's ``ChatOllama``."""

    name = "ollama"

    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.0, seed: int = 0):
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self._client = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "The 'ollama' provider needs the 'langchain-ollama' package. "
                "Install requirements and run `ollama pull gemma3:latest`, or use "
                "provider='mock' for offline runs."
            ) from exc
        # Ask Ollama for JSON output and pin the seed for reproducibility; fall
        # back gracefully if a given langchain-ollama version rejects a kwarg.
        for kwargs in (
            dict(model=self.model, temperature=self.temperature, seed=self.seed, format="json"),
            dict(model=self.model, temperature=self.temperature, format="json"),
            dict(model=self.model, temperature=self.temperature),
        ):
            try:
                self._client = ChatOllama(**kwargs)
                return
            except TypeError:
                continue
        self._client = ChatOllama(model=self.model)  # last resort

    def generate(self, prompt: str, event: Optional[Event] = None) -> str:
        self._ensure_client()
        response = self._client.invoke(prompt)
        content = getattr(response, "content", response)
        return content if isinstance(content, str) else str(content)


def get_provider(name: str, model: str = DEFAULT_MODEL, temperature: float = 0.0, seed: int = 0):
    """Construct a provider by name (``"ollama"`` or ``"mock"``)."""
    name = name.lower()
    if name == "mock":
        return MockProvider()
    if name == "ollama":
        return OllamaProvider(model=model, temperature=temperature, seed=seed)
    raise ValueError(f"Unknown provider '{name}'. Use 'ollama' or 'mock'.")


# --------------------------------------------------------------------------- #
# Mock heuristic. Reasons over the same signals a naive analyst would; kept
# intentionally simple and imperfect (e.g. it does not follow cross-event
# context, so it misses T1078 / T1098). Every reason is prefixed with [mock].
# --------------------------------------------------------------------------- #
def _v(is_incident: bool, technique: Optional[str], confidence: float, reason: str) -> dict[str, Any]:
    return {
        "is_incident": is_incident,
        "technique": technique if is_incident else None,
        "confidence": confidence,
        "reason": f"[mock] {reason}",
    }


def _mock_verdict_dict(event: Event) -> dict[str, Any]:
    action = (event.action or "").lower()
    raw = event.raw or {}
    nested = raw.get("event") if isinstance(raw.get("event"), dict) else {}
    count = raw.get("count")
    count = count if isinstance(count, int) else 0

    if action.startswith("http_"):
        return _mock_web(event, raw)

    if action == "ssh_failed_password":
        if count >= 8:
            return _v(True, "T1110.001", 0.85, "sustained SSH failed-password burst from one IP")
        return _v(False, None, 0.6, "few SSH failures: below a brute-force bar")

    if action == "logon_failure":
        if count >= 5:
            return _v(True, "T1110.001", 0.8, "burst of Windows failed logons from one source")
        return _v(False, None, 0.6, "isolated Windows logon failure")

    if action == "ssh_accepted_password":
        return _v(False, None, 0.4, "successful SSH login with no visible prior guessing")

    if action == "sudo_not_in_sudoers":
        return _v(False, None, 0.5, "single denied sudo: suspicious but not conclusive alone")

    if action == "process_creation":
        cmd = str(nested.get("command_line", "")).lower()
        if "-enc" in cmd or "-encodedcommand" in cmd:
            return _v(True, "T1059.001", 0.8, "encoded PowerShell command line")
        if "certutil" in cmd and ("http" in cmd or "urlcache" in cmd):
            return _v(True, "T1105", 0.75, "certutil downloading a remote executable")
        return _v(False, None, 0.5, "ordinary process creation")

    if action == "user_account_created":
        return _v(True, "T1136.001", 0.55, "new local account created: possible persistence backdoor")

    if action == "security_group_rule_added":
        port = nested.get("port")
        cidr = nested.get("cidr")
        try:
            port_i = int(port)
        except (TypeError, ValueError):
            port_i = -1
        if cidr == "0.0.0.0/0" and port_i in _SENSITIVE_PORTS:
            return _v(True, "T1562.007", 0.8, f"security group opened to 0.0.0.0/0 on sensitive port {port}")
        return _v(False, None, 0.5, "security group change that is not an internet-wide sensitive-port opening")

    if action == "iam_policy_changed":
        return _v(False, None, 0.5, "IAM policy change with no visible prior auth failures")

    return _v(False, None, 0.5, f"no incident signal for action '{event.action}'")


def _mock_web(event: Event, raw: dict[str, Any]) -> dict[str, Any]:
    path = str(raw.get("path") or event.target or "")
    uas = raw.get("user_agents") or ([raw["user_agent"]] if raw.get("user_agent") else [])
    ua_join = " ".join(str(u).lower() for u in uas)
    count = raw.get("count")
    count = count if isinstance(count, int) else 0

    payload = classify_web_payload(path, ua_join)
    sample_paths = raw.get("sample_paths") or []
    for p in sample_paths:
        payload = payload or classify_web_payload(str(p), None)
    if payload == "T1190":
        return _v(True, "T1190", 0.85, "attack payload in the web request")
    if payload == "T1595.002" or any(s in ua_join for s in _SCANNER_UA):
        return _v(True, "T1595.002", 0.8, "requests from a known scanner user-agent")

    # Login brute force: a run of POSTs to a login endpoint.
    if event.action == "http_post" and raw.get("status_sequence") and count >= 5:
        return _v(True, "T1110", 0.75, "many failed web-login POSTs from one IP")

    # Systematic probing of sensitive paths.
    joined_paths = " ".join(str(p).lower() for p in ([path] + list(sample_paths)))
    if any(s in joined_paths for s in _SENSITIVE_PATHS):
        return _v(True, "T1083", 0.7, "systematic probing of sensitive paths")

    # High-volume 404 burst from an ordinary browser: a broken-link crawl.
    if count >= 20 and event.status == "404" and "mozilla" in ua_join:
        return _v(False, None, 0.6, "high 404 volume but from a normal browser: broken-link crawl")

    return _v(False, None, 0.5, "ordinary web request")


# --------------------------------------------------------------------------- #
# Detector with an on-disk response cache.
# --------------------------------------------------------------------------- #
def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


class LLMDetector:
    """Classify :class:`Event` objects with a versioned prompt and a provider.

    Parameters
    ----------
    prompt_version:
        ``"v1"`` / ``"v2"`` / ``"v3"`` (or the bare number).
    provider:
        ``"ollama"`` (real, default) or ``"mock"`` (deterministic, offline).
    refresh:
        Recompute every response this run and overwrite the cache entries.
    use_cache:
        Read/write the on-disk cache. When False the provider is called every
        time and nothing is persisted.
    require_cache:
        Offline / reproduce-only mode (``make run --from-cache``). When True a
        cache miss raises instead of calling the provider, so a run over an
        incomplete cache fails loudly (and, via the runner, is skipped) rather
        than silently reaching out to the model. Mutually exclusive with
        ``refresh``.
    """

    def __init__(
        self,
        prompt_version: str = "v1",
        provider: str = DEFAULT_PROVIDER,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.0,
        seed: int = 0,
        prompts_dir: Path | str = PROMPTS_DIR,
        cache_dir: Path | str = CACHE_DIR,
        refresh: bool = False,
        use_cache: bool = True,
        autosave: bool = True,
        require_cache: bool = False,
    ):
        self.prompt_version = prompt_version if prompt_version.startswith("v") else f"v{prompt_version}"
        self.provider_name = provider.lower()
        self.model = "mock" if self.provider_name == "mock" else model
        self.provider = get_provider(self.provider_name, model=self.model, temperature=temperature, seed=seed)
        self.template = load_prompt(self.prompt_version, prompts_dir)
        self.cache_dir = Path(cache_dir)
        self.refresh = refresh
        self.use_cache = use_cache
        self.autosave = autosave
        self.require_cache = require_cache

        self._responses: dict[str, dict[str, Any]] = {}
        self._recomputed: set[str] = set()
        self._dirty = False
        self._cache_path = self.cache_dir / f"{self.provider_name}__{_safe(self.model)}__{self.prompt_version}.json"
        if self.use_cache:
            self._load_cache()

    # -- cache I/O ---------------------------------------------------------- #
    def _load_cache(self) -> None:
        if self._cache_path.exists():
            try:
                blob = json.loads(self._cache_path.read_text(encoding="utf-8"))
                self._responses = blob.get("responses", {})
            except (json.JSONDecodeError, OSError):
                self._responses = {}

    def save(self) -> None:
        if not self.use_cache:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        blob = {
            "provider": self.provider_name,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "responses": dict(sorted(self._responses.items())),
        }
        if self.provider_name == "mock":
            blob["note"] = _MOCK_NOTE
        self._cache_path.write_text(
            json.dumps(blob, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._dirty = False

    # -- classification ----------------------------------------------------- #
    def _cache_key(self, event: Event, prompt: str) -> str:
        payload = "\n".join([self.provider_name, self.model, self.prompt_version, event.id, prompt])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def classify(self, event: Event) -> Verdict:
        prompt = build_prompt(self.template, event)
        key = self._cache_key(event, prompt)

        use_cached = (
            self.use_cache
            and key in self._responses
            and not (self.refresh and key not in self._recomputed)
        )
        if use_cached:
            raw = self._responses[key]["response"]
        else:
            if self.require_cache:
                raise RuntimeError(
                    f"no cached response for unit {event.id} (provider={self.provider_name}, "
                    f"model={self.model}, prompt={self.prompt_version}) under --from-cache. "
                    "The response cache is incomplete for this run; populate it with "
                    "`make run-llm` (needs a local Ollama model) before reproducing offline."
                )
            raw = self.provider.generate(prompt, event)
            self._recomputed.add(key)
            if self.use_cache:
                self._responses[key] = {"response": raw, "unit_id": event.id}
                self._dirty = True

        return parse_verdict(raw)

    def classify_many(self, events: list[Event]) -> list[Verdict]:
        verdicts = [self.classify(e) for e in events]
        if self.use_cache and self._dirty and self.autosave:
            self.save()
        return verdicts

    def predict(self, event: Event) -> Prediction:
        return Prediction.from_verdict(event.id, self.classify(event))

    def predict_many(self, events: list[Event]) -> list[Prediction]:
        verdicts = self.classify_many(events)
        return [Prediction.from_verdict(e.id, v) for e, v in zip(events, verdicts)]


# --------------------------------------------------------------------------- #
# Corpus helpers and a small CLI to warm the cache.
# --------------------------------------------------------------------------- #
def load_corpus_events(labels_path: Path | str) -> list[Event]:
    """Load the authoritative analysis units (Events) from ``corpus/labels.jsonl``."""
    from detector.schema import LabeledUnit

    events: list[Event] = []
    with Path(labels_path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(LabeledUnit.model_validate(json.loads(line)).event)
    return events


def main() -> None:
    parser = argparse.ArgumentParser(description="Run / warm the LLM detector cache.")
    parser.add_argument("--provider", default="mock", choices=["mock", "ollama"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default="v1", help="Prompt version, e.g. v1 (or 'all').")
    parser.add_argument("--corpus", default=str(_REPO_ROOT / "corpus" / "labels.jsonl"))
    parser.add_argument("--refresh", action="store_true", help="Recompute and overwrite the cache.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N units.")
    args = parser.parse_args()

    events = load_corpus_events(args.corpus)
    if args.limit:
        events = events[: args.limit]

    versions = ["v1", "v2", "v3"] if args.prompt == "all" else [args.prompt]
    for version in versions:
        detector = LLMDetector(
            prompt_version=version,
            provider=args.provider,
            model=args.model,
            refresh=args.refresh,
        )
        verdicts = detector.classify_many(events)
        detector.save()
        incidents = sum(1 for v in verdicts if v.is_incident)
        print(
            f"[{args.provider} {detector.model} {version}] classified {len(verdicts)} units "
            f"({incidents} incident, {len(verdicts) - incidents} benign) -> {detector._cache_path}"
        )


if __name__ == "__main__":
    main()
