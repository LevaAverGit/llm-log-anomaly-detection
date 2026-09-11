# Rules-vs-LLM log anomaly detection — common tasks.
# Everything runs locally; the LLM detector uses Ollama, no API keys.

# Prefer the interpreter the project is tested on (3.11), then any modern
# python3; override with `make PYTHON=... install` if you need a specific one.
PYTHON ?= $(shell command -v python3.11 python3.12 python3.13 python3 2>/dev/null | head -n1)
VENV   := .venv
PY     := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip

.PHONY: install test run run-llm run-models corpus clean

## install: create a virtualenv and install dependencies
install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

## corpus: rebuild corpus/labels.jsonl from corpus/raw/ (deterministic)
corpus:
	$(PY) corpus/build_corpus.py

## test: run the test suite
test:
	$(PY) -m pytest -q

## run: reproduce every row whose response cache is present, offline (no network,
##      no model call — reads corpus/llm_cache/). The rules row always reproduces;
##      the LLM/hybrid rows fill in once `make run-llm` has committed their cache.
run:
	$(PY) -m runner.run --all --from-cache

## run-llm: run the LLM detector live against the local Ollama model
##          (requires `ollama pull gemma3:latest`). Warms the response cache for
##          the two prompt versions the README table uses, then assembles every
##          row (rules, LLM v1/v3, hybrid) and the 3+3 error analysis from that
##          cache and rewrites the README — after this, `make run` reproduces it
##          all offline.
run-llm:
	$(PY) -m runner.run --approach llm --prompt v1 --refresh
	$(PY) -m runner.run --approach llm --prompt v3 --refresh
	$(PY) -m runner.run --all --from-cache

## run-models: HEAVY. Cross-model comparison — the same corpus and prompts
##             (v1 + v3) across every model in models.json, then render the
##             "Cross-model comparison" README section and reports/models_summary.*.
##             Needs each Ollama model pulled (ollama pull qwen2.5:7b, etc.);
##             gemma3 reproduces from the committed cache. GigaChat is included
##             only if `pip install -r requirements-gigachat.txt` is done and
##             GIGACHAT_CREDENTIALS is exported — otherwise it is skipped and
##             reported as unavailable. Edit models.json to match what you have.
run-models:
	$(PY) -m runner.run --cross-model

## clean: remove caches and generated reports
clean:
	rm -rf .pytest_cache **/__pycache__ reports/generated
