"""Pytest configuration: make the repository root importable.

Placed here so ``import detector``, ``import detector.llm`` and
``import eval.metrics`` resolve to the project sources no matter which
directory pytest is invoked from.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = Path(__file__).resolve().parent

for path in (str(REPO_ROOT), str(TESTS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)
