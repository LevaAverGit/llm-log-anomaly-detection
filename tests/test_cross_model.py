"""Cross-model comparison: runner iteration, GigaChat graceful skip, table render.

All offline. The only provider ever exercised end-to-end is the deterministic
``mock`` (no Ollama, no network, no GigaChat call), and the per-model cache-key
check inspects the detector's pure key/path helpers without ever invoking a
model. Nothing here fabricates a real result.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _support as S  # noqa: E402
from detector import llm as llm_mod  # noqa: E402
from detector.schema import SourceType  # noqa: E402
from eval import cross_model as cm  # noqa: E402
from eval.metrics import score  # noqa: E402


# --------------------------------------------------------------------------- #
# 1) The multi-model runner iterates every model and keys the cache per model.
# --------------------------------------------------------------------------- #
def test_multimodel_runner_iterates_and_keys_cache_per_model(tmp_path):
    """Running two models writes a per-model prediction file for each, and the
    detector's response cache is keyed on the model name so they never collide."""
    from runner import run as runner

    pred_dir = tmp_path / "pred"
    reports_dir = tmp_path / "rep"
    rc = runner.main([
        "--models", "mock:model-aaa,mock:model-bbb",
        "--versions", "v1",
        "--pred-dir", str(pred_dir),
        "--reports-dir", str(reports_dir),
    ])
    assert rc == 0

    models_dir = pred_dir / cm.MODELS_SUBDIR
    # The runner iterated BOTH models (distinct per-model prediction files) plus
    # the rules baseline reference row.
    assert (models_dir / cm.run_key("mock", "model-aaa", "v1")).with_suffix(".jsonl").exists()
    assert (models_dir / cm.run_key("mock", "model-bbb", "v1")).with_suffix(".jsonl").exists()
    assert (models_dir / f"{cm.RULES_RUN}.jsonl").exists()

    # The response cache is keyed on the model: two Ollama models get distinct
    # cache files AND distinct per-event cache keys; the same model is stable.
    # (Constructing an Ollama detector does not touch the network — the client
    # is created lazily on the first classify call, which never happens here.)
    d_qwen = llm_mod.LLMDetector(provider="ollama", model="qwen2.5:7b", use_cache=False)
    d_deep = llm_mod.LLMDetector(provider="ollama", model="deepseek-r1:7b", use_cache=False)
    d_qwen2 = llm_mod.LLMDetector(provider="ollama", model="qwen2.5:7b", use_cache=False)

    assert d_qwen._cache_path != d_deep._cache_path
    assert d_qwen._cache_path == d_qwen2._cache_path

    event = S.make_event("u1", SourceType.linux_auth)
    prompt = "the same prompt text for every model"
    assert d_qwen._cache_key(event, prompt) != d_deep._cache_key(event, prompt)
    assert d_qwen._cache_key(event, prompt) == d_qwen2._cache_key(event, prompt)


# --------------------------------------------------------------------------- #
# 2) GigaChat skips gracefully when GIGACHAT_CREDENTIALS is unset.
# --------------------------------------------------------------------------- #
def test_gigachat_provider_skips_without_credentials(monkeypatch):
    """Unset credential => not available, and using the provider raises a clear
    error instead of returning a fabricated verdict."""
    monkeypatch.delenv(llm_mod.GIGACHAT_ENV, raising=False)

    assert llm_mod.gigachat_available() is False

    provider = llm_mod.get_provider("gigachat", model="GigaChat")
    with pytest.raises(RuntimeError) as excinfo:
        provider.generate("some prompt", None)
    # The error must name the missing credential (or the missing package), so the
    # runner can report the skip — never silently return output.
    message = str(excinfo.value)
    assert llm_mod.GIGACHAT_ENV in message or "langchain-gigachat" in message


def test_gigachat_model_is_skipped_and_reported_by_runner(tmp_path, monkeypatch):
    """A cross-model run with only GigaChat and no credentials produces no
    GigaChat predictions (it is skipped) rather than crashing or faking them."""
    from runner import run as runner

    monkeypatch.delenv(llm_mod.GIGACHAT_ENV, raising=False)
    pred_dir = tmp_path / "pred"
    reports_dir = tmp_path / "rep"
    rc = runner.main([
        "--models", "gigachat:GigaChat",
        "--versions", "v1",
        "--pred-dir", str(pred_dir),
        "--reports-dir", str(reports_dir),
    ])
    # No model produced results -> non-zero, and no GigaChat prediction file.
    assert rc == 1
    models_dir = pred_dir / cm.MODELS_SUBDIR
    assert not (models_dir / cm.run_key("gigachat", "GigaChat", "v1")).with_suffix(".jsonl").exists()


# --------------------------------------------------------------------------- #
# 3) The model x prompt-version F1 table renders on a toy result set.
# --------------------------------------------------------------------------- #
def _toy_metric(display_name: str):
    """Score a tiny hand-made set (P=0.75, R=0.60) into a RunMetrics."""
    toy = [
        ("u01", True, True), ("u02", True, True), ("u03", True, True),
        ("u04", True, False), ("u05", True, False), ("u06", False, True),
        ("u07", False, False), ("u08", False, False),
    ]
    labels = [S.make_labeled_unit(uid, truth, technique="T1110" if truth else None)
              for uid, truth, _ in toy]
    preds = [S.make_prediction(uid, pred) for uid, _, pred in toy]
    return score(labels, preds, run=display_name, display_name=display_name)


def test_model_version_table_renders():
    specs = [
        cm.ModelSpec(provider="ollama", model="gemma3:latest", label="gemma3"),
        cm.ModelSpec(provider="ollama", model="qwen2.5:7b", label="Qwen2.5 7B"),
    ]
    versions = ["v1", "v3"]

    metric = _toy_metric("gemma3 · v1")
    # Only gemma3 v1 has a result; every other (model, version) is pending.
    metrics_by = {(cm.model_key("ollama", "gemma3:latest"), "v1"): metric}
    rules_metric = _toy_metric("Rules (Sigma baseline)")

    md = cm.model_version_table_markdown(specs, versions, metrics_by, rules_metric)

    # Column headers = prompt versions; rows = models plus the rules reference.
    assert "F1 (v1)" in md and "F1 (v3)" in md
    assert "gemma3" in md and "Qwen2.5 7B" in md
    assert "Rules (Sigma baseline)" in md
    # The one computed cell shows the real F1; missing cells show the pending note.
    assert f"{metric.f1:.2f}" in md
    assert cm._MISSING_NOTE in md


def test_render_fills_readme_markers(tmp_path):
    """End to end: render() fills its own markers from toy predictions and leaves
    an unrelated section untouched."""
    import json

    # A toy corpus and one model's predictions on disk.
    toy = [("u01", True, "T1110"), ("u02", False, None), ("u03", True, "T1059.001")]
    labels_path = tmp_path / "labels.jsonl"
    with labels_path.open("w", encoding="utf-8") as fh:
        for uid, inc, tech in toy:
            fh.write(S.make_labeled_unit(uid, inc, technique=tech).model_dump_json() + "\n")

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    spec = cm.ModelSpec(provider="ollama", model="gemma3:latest", label="gemma3")
    key = cm.run_key(spec.provider, spec.model, "v1")
    # gemma3 v1: catch u01, miss u03, no false positives.
    rows = [("u01", True, "T1110"), ("u02", False, None), ("u03", False, None)]
    with (models_dir / f"{key}.jsonl").open("w", encoding="utf-8") as fh:
        for uid, inc, tech in rows:
            fh.write(json.dumps({"unit_id": uid, "is_incident": inc, "technique": tech}) + "\n")

    readme = tmp_path / "README.md"
    readme.write_text(
        "## Keep me\nuntouched prose\n\n"
        "<!-- BEGIN:cross_model_table -->\nOLD\n<!-- END:cross_model_table -->\n\n"
        "<!-- BEGIN:cross_model_errors -->\nOLD\n<!-- END:cross_model_errors -->\n",
        encoding="utf-8",
    )

    written = cm.render(
        [spec], ["v1"], models_dir,
        readme_path=readme, labels_path=labels_path, reports_dir=tmp_path / "rep",
    )
    assert cm.MARKER_TABLE in written and cm.MARKER_ERRORS in written

    text = readme.read_text(encoding="utf-8")
    assert "untouched prose" in text          # unrelated content preserved
    assert "OLD" not in text                   # both marker blocks replaced
    assert "gemma3" in text                    # model row rendered
    assert "F1 (v1)" in text                   # version column rendered
    assert (tmp_path / "rep" / "models_summary.md").exists()
