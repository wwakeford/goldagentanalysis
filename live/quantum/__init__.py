"""Live Quantum Queen stack — the deploy side of goldreplay (``goldreplay/live/``).

goldreplay researches the Quantum Queen grid bias-free (the no-lookahead replay
harness); this package is where the *same* validated strategy runs against a live
MT5 broker. Importing it puts the canonical quantumstrategy backtest modules on
sys.path (see ``_paths``), then exposes the live decision core and feature engine:

  * ``QuantumEngine``        -- per-bar decision state machine (mirrors backtest/engine.py)
  * ``QuantumFeatureState``  -- incremental live recomputation of the M1 features
  * ``build_live_config``    -- the shipped, validated winner BacktestConfig

The runner (``live_quantum.QuantumRunner``) and entrypoint (``run_live_quantum``) live
one level up in ``goldreplay/live/`` and drive these through the shared Broker.
"""
from __future__ import annotations

from . import _paths  # noqa: F401  (side effect: sys.path)
from .config_live import build_live_config
from .quantum_engine import QuantumEngine
from .quantum_features import QuantumFeatureState

__all__ = ["build_live_config", "QuantumEngine", "QuantumFeatureState"]
