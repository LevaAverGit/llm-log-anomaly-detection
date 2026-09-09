"""Write the results table and the error examples into README.md.

The top-level README is what a reviewer reads first, so its numbers must come
straight from the scored predictions rather than being edited by hand. This
module regenerates three regions of README.md, each delimited by an explicit
HTML-comment marker pair so the surrounding prose is never touched:

    <!-- BEGIN:results_table -->            ... <!-- END:results_table -->
    <!-- BEGIN:rules_miss_llm_catch -->     ... <!-- END:rules_miss_llm_catch -->
    <!-- BEGIN:llm_misfire_rules_right -->  ... <!-- END:llm_misfire_rules_right -->

It is idempotent: running it again with the same predictions produces the same
file. If no predictions exist yet it refuses to touch README (so a stale table
is never wiped to nothing); if only some runs exist it fills what it can and
leaves the rest as placeholders.
"""

from __future__ import annotations

import re
from pathlib import Path

from eval import compare, errors
from eval.metrics import (
    DEFAULT_LABELS_PATH,
    DEFAULT_PREDICTIONS_DIR,
    DEFAULT_REPORTS_DIR,
    predictions_path,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_README_PATH = _REPO_ROOT / "README.md"

MARKER_TABLE = "results_table"
MARKER_RULES_MISS = "rules_miss_llm_catch"
MARKER_LLM_MISFIRE = "llm_misfire_rules_right"


def _replace_block(text: str, marker: str, inner: str) -> tuple[str, int]:
    """Replace the content between the BEGIN/END markers for ``marker``.

    Returns the new text and the number of blocks replaced (0 if the marker pair
    is absent).
    """
    pattern = re.compile(
        r"(<!-- BEGIN:%s -->)(.*?)(<!-- END:%s -->)" % (re.escape(marker), re.escape(marker)),
        re.DOTALL,
    )

    def _sub(match: re.Match) -> str:
        return f"{match.group(1)}\n{inner}\n{match.group(3)}"

    return pattern.subn(_sub, text)


def render(
    readme_path: Path | str = DEFAULT_README_PATH,
    labels_path: Path | str = DEFAULT_LABELS_PATH,
    pred_dir: Path | str = DEFAULT_PREDICTIONS_DIR,
    rules_run: str = errors.DEFAULT_RULES_RUN,
    llm_run: str = errors.DEFAULT_LLM_RUN,
    reports_dir: Path | str = DEFAULT_REPORTS_DIR,
) -> list[str]:
    """Update README.md in place and refresh the ``reports/`` files it links to.

    Returns the list of markers that were written. Report side effects are
    confined to ``reports_dir`` so a caller pointing at a scratch directory never
    touches the committed ``reports/``.
    """
    readme_path = Path(readme_path)
    metrics = compare.summarize(labels_path, pred_dir)
    if not metrics:
        raise SystemExit(
            f"No prediction files found in {pred_dir}; refusing to overwrite the "
            "README table. Run the detector first (e.g. `make run`)."
        )

    text = readme_path.read_text(encoding="utf-8")
    written: list[str] = []
    missing_markers: list[str] = []

    # 1) Results table (and the fuller summary under reports/).
    compare.write_reports(metrics, reports_dir)
    table_md = compare.readme_table_markdown(metrics)
    text, n = _replace_block(text, MARKER_TABLE, table_md)
    (written if n else missing_markers).append(MARKER_TABLE)

    # 2) Error examples (only if both the rules and LLM runs are available).
    have_errors = (
        predictions_path(rules_run, pred_dir).exists()
        and predictions_path(llm_run, pred_dir).exists()
    )
    if have_errors:
        sets = errors.analyze(labels_path, pred_dir, rules_run, llm_run)
        errors.write_reports(sets, reports_dir)  # keep reports/ in sync
        for marker, md in (
            (MARKER_RULES_MISS, errors.rules_miss_markdown(sets)),
            (MARKER_LLM_MISFIRE, errors.llm_misfire_markdown(sets)),
        ):
            text, n = _replace_block(text, marker, md)
            (written if n else missing_markers).append(marker)

    readme_path.write_text(text, encoding="utf-8")

    if missing_markers:
        joined = ", ".join(f"<!-- BEGIN:{m} -->/<!-- END:{m} -->" for m in missing_markers)
        print(f"warning: marker pair(s) not found in {readme_path.name}: {joined}")
    return written


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Render results into README.md between markers.")
    parser.add_argument("--readme", default=str(DEFAULT_README_PATH))
    parser.add_argument("--labels", default=str(DEFAULT_LABELS_PATH))
    parser.add_argument("--pred-dir", default=str(DEFAULT_PREDICTIONS_DIR))
    parser.add_argument("--rules-run", default=errors.DEFAULT_RULES_RUN)
    parser.add_argument("--llm-run", default=errors.DEFAULT_LLM_RUN)
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    args = parser.parse_args()

    written = render(
        args.readme, args.labels, args.pred_dir, args.rules_run, args.llm_run, args.reports_dir
    )
    print(f"Updated {args.readme}: wrote {', '.join(written) if written else 'nothing'}.")


if __name__ == "__main__":
    main()
