# Corpus and labeling criterion

This directory holds the labeled corpus that both the rules baseline and the
LLM detector are scored against. The metrics in the top-level README are only
as trustworthy as the labels here, so this file states the exact criterion and,
crucially, **how the labels were produced without using the detection rules.**

## Files

| File | What it is |
|---|---|
| `raw/` | Verbatim copies of the four mini-siem sample logs (the raw material). |
| `labels.jsonl` | One JSON object per analysis unit: the normalized event plus its ground-truth label, technique and a short rationale. |
| `build_corpus.py` | Deterministic script that parses `raw/` and writes `labels.jsonl`. |

Each row of `labels.jsonl` validates against `detector.schema.LabeledUnit`:

```json
{
  "unit_id": "lx-0040",
  "event": {"id": "lx-0040", "source": "linux_auth", "timestamp": "2026-01-10T09:10:15+00:00",
            "actor": "192.0.2.100", "action": "ssh_failed_password", "target": "admin, oracle, root",
            "status": "failure", "raw": {"count": 13, "window_start": "...", "sample_lines": ["..."]}},
  "label": "incident",
  "technique": "T1110.001",
  "rationale": "Sustained SSH password guessing ..."
}
```

## The honesty rule (why labels are independent of the rules)

The whole point of this project is to compare rule-based detection with an LLM.
If the ground truth were produced by *running the Sigma rules*, the rules would
trivially score a perfect result against their own output, and the comparison
would be meaningless.

So the labels here come from **knowledge of which activity was injected as an
attack**, encoded by hand in `build_corpus.py` (`_INCIDENT_KNOWLEDGE`). That map
lists the specific attacker IPs, hosts and payloads in the mini-siem scenario.
The builder **never imports `rules/engine.py`**. Counts are recorded for context
and never flip a label: a count may only decide how *benign* events are grouped,
not turn a benign unit into an incident or vice versa. The labels would be
identical if the Sigma rules did not exist.

A direct consequence: some units are labeled in a way a naive threshold rule
would get *wrong*, on purpose (see "Boundary cases" below). That is what makes
the later precision/recall numbers real.

### What this independence does and does not buy

The labels are independent of the *engine* — but for the signature classes the
labeling signal and the rule's match key are the same objective artifact. A web
exploit is labeled from the payload string in the request (`union select`,
`../`, `${jndi:`), and the rule matches that same string; scanning is labeled
from the scanner User-Agent the rule also keys on; the cloud security-group case
uses the same sensitive-port set. On those classes the rules cannot lose *by
construction*, so their perfect recall there is not independent evidence of rule
quality. The informative rules-vs-LLM comparison therefore lives in the
**count / correlation / host-process-semantics** classes — brute-force
thresholds, cross-event chaining (a login right after a burst, an IAM change
after failures) and process semantics (encoded PowerShell, LOLBin downloads) —
which is exactly where the rules baseline still misses incidents.

## Unit of analysis

A **unit** is one thing that receives a single verdict. It is not always one log
line:

- **Aggregated units.** A brute-force burst or a scan from one source is a
  single coherent activity — a count-based rule and an LLM both need the whole
  burst to judge it — so all of its lines collapse into one unit. The unit's
  `raw` keeps `count`, `window_start`/`window_end` and a few `sample_lines`.
- **Point units.** A single login, one exploit request, one Windows event or one
  cloud action is its own unit.

Aggregation groups events by acting entity (source IP, or host+IP for Windows);
it is a grouping convenience, not a detection threshold.

## Labeling criterion

A unit is labeled **`incident`** if and only if it is one of the following known
adversary behaviours, judged from the raw content and the injected scenario:

| Behaviour | Technique | Signal used to label (not a rule) |
|---|---|---|
| Credential brute force (SSH, web login, Windows logon) | T1110 / T1110.001 | Sustained failed-auth burst from a known attacker source in the scenario. |
| Compromise via guessed credentials | T1078 | A successful login from a source that had just been brute-forcing the same host. |
| Web exploitation | T1190 | A request whose URL-decoded path or User-Agent carries a known payload: SQLi, path traversal, OS command injection, Log4Shell/JNDI. |
| Web reconnaissance / scanning | T1595.002 / T1083 | A burst from a known scanner User-Agent (gobuster, nikto, sqlmap) or systematic hits on sensitive paths (`/.env`, `/.git`, `/phpmyadmin`, `/admin`). |
| Endpoint post-exploitation | T1059.001 / T1105 | Encoded PowerShell (`-enc`) or a LOLBin download (`certutil` fetching a remote exe). |
| Persistence backdoor | T1136.001 | A new local account created on a host that had just seen a logon-failure burst. |
| Defence weakening in the cloud | T1562.007 | A security group opened to `0.0.0.0/0` on a remote-access or database port (22, 3389, 3306, 5432, 27017). |
| Cloud identity abuse | T1098 | An IAM policy change by an identity that had recent console-login failures. |

Everything else is **`benign`**: normal logins, ordinary browsing, authorised
sudo and admin work, read-only API calls, and low-volume single failures.

## Boundary cases (deliberate, and where naive rules diverge from truth)

These are labeled from the criterion above, *against* what a pure threshold rule
would say, and they are the honest source of false positives / false negatives:

- **`ng-0071`** — 35 consecutive `404`s from `192.0.2.200`, but from an ordinary
  browser User-Agent on sequential `/pageN` paths. Labeled **benign** (a
  broken-link crawl). A "count 404s per IP" rule over-fires here.
- **`lx-0041`** — 2 SSH invalid-user attempts from `198.51.100.200`. Labeled
  **benign**: two tries is noise, not an attack.
- **`lx-0045`** — a single denied `sudo` by an unknown user. Labeled **benign**
  on its own (suspicious, but not enough evidence as a standalone unit).
- **`cld-0015`** — 2 failed console logins. Labeled **benign** (below any bar).
- **Synthetic look-alikes** — a fat-finger login (a few failures then success),
  a help-desk account creation with no preceding failures, a security group
  opened only on port 443 or to a private CIDR, and an IAM change with no
  preceding failures. All **benign**, all designed to tempt a false positive.

## Corpus composition

The mini-siem sample logs are attack demonstrations, so on their own they are
attack-heavy and unrealistic. To give a realistic base rate (incidents are a
small minority of a real SOC feed), the builder adds a deterministic,
obviously-benign background of everyday activity across all four sources.

| | Units |
|---|---|
| Total | 231 |
| Incidents | 22 (9.5%) |
| Benign | 209 |
| — of which real (from `raw/`) | 15 |
| — of which synthetic background | 194 |
| By source | linux_auth 66, nginx_access 108, windows_security 32, cloud_audit 25 |

Rebuild with:

```bash
python corpus/build_corpus.py
```

The output is fully deterministic (fixed seed, fixed catalogs), so `labels.jsonl`
is stable across runs and safe to commit.
