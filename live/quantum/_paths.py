"""Locate the canonical Quantum Queen backtest package and put it on sys.path.

The live Quantum stack deliberately *imports* the exact reverse-engineered modules
(`backtest.config`, `backtest.basket`, `backtest.sizing`, `backtest.features`, and
`scripts/price_align.py` + `scripts/analyze_entry.py`) rather than copying their
math. Copying would create the single worst failure mode for this port — silent
feature drift between the validated backtest and the live bot. Importing the
originals makes drift impossible by construction; the feature-parity gate then only
has to prove the *incremental* recomputation matches, not the formulas.

Resolution order for the quantumstrategy repo root (the dir that contains
`backtest/` and `scripts/`):
  1. env ``QUANTUM_REPO`` (set this on the Windows box after copying the repo).
  2. the repo sitting next to ``goldreplay/`` — i.e. ``modeltomarket/quantumstrategy``
     (the macOS dev layout), three levels up from this file.
  3. ``../../quantumstrategy`` (flatter copy layouts beside ``live/``).

"Vendoring for deploy" = copy ``quantumstrategy/backtest`` + ``quantumstrategy/scripts``
+ ``quantumstrategy/data/price`` (for the parity oracle only; not needed live) next to
this repo and point ``QUANTUM_REPO`` at it. No code is duplicated.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()


def _candidates():
    env = os.environ.get("QUANTUM_REPO")
    if env:
        yield Path(env).expanduser()
    # goldreplay/live/quantum/_paths.py -> repo roots near it
    mtm_root = _HERE.parents[3]               # .../modeltomarket (holds goldreplay/ + quantumstrategy/)
    yield mtm_root / "quantumstrategy"
    yield _HERE.parents[2] / "quantumstrategy"  # flatter copy beside live/


def quantum_repo() -> Path:
    for c in _candidates():
        if c and (c / "backtest").is_dir() and (c / "scripts").is_dir():
            return c
    tried = "\n  ".join(str(c) for c in _candidates())
    raise RuntimeError(
        "Could not locate the quantumstrategy repo (needs backtest/ + scripts/).\n"
        "Set QUANTUM_REPO to the copied repo root. Tried:\n  " + tried
    )


def ensure_on_path() -> Path:
    """Add the repo root (for ``import backtest``) and scripts/ to sys.path."""
    root = quantum_repo()
    for p in (str(root), str(root / "scripts")):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


ROOT = ensure_on_path()
