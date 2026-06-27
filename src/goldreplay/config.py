# Vendored (trimmed) from quantumstrategy/backtest/config.py — research-only flip/
# cat-stop/regime-gate flags dropped; defaults encode the documented BALANCED EA rule.
"""All tunable parameters for the gold grid strategy in one place.

Every magic number from the reverse-engineering (entry_rule.md, basket close
findings) is a named field here. Defaults are the BALANCED risk posture: flat
lots, no martingale, hard-capped depth & adverse excursion, and the EA's session
windows / trend floors / breakout fractions in BROKER time (UTC+3).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Set


def _default_windows() -> Dict[str, Set[int]]:
    return {
        "evening": {18, 19, 21, 22, 23},   # UTC 15-20 (London close / NY afternoon)
        "midnight": {3},                    # UTC 0    (Asian open)
        "morning": {8, 9, 10, 11},          # UTC 5-8  (pre-London / London open)
    }


def _default_min_trend() -> Dict[str, float]:
    return {"evening": 20.0, "midnight": 18.0, "morning": 10.0}


def _default_breakout() -> Dict[str, float]:
    return {"evening": 0.80, "midnight": 0.75, "morning": 0.82}


@dataclass
class BacktestConfig:
    # --- capital / contract / leverage ---
    initial_capital: float = 1_000_000.0
    contract: float = 100.0          # XAUUSD: 1 lot = 100 oz ($1/oz = $100/lot)
    leverage_cap: float = 28.0       # binding Quanthack constraint
    account_leverage: float = 500.0  # 1:500 available
    stop_out_frac: float = 0.50      # MT5 stop-out: liquidate when equity < 50% of used margin

    # --- entry (from data/reference/strategy_card.md) ---
    windows: Dict[str, Set[int]] = field(default_factory=_default_windows)
    min_trend_usd: Dict[str, float] = field(default_factory=_default_min_trend)
    breakout_frac: Dict[str, float] = field(default_factory=_default_breakout)
    long_only: bool = False
    entry_strictness: float = 1.0
    trend_lookback: int = 240        # M1 bars = 4h
    range_lookback: int = 60         # M1 bars = 60 min
    rsi_len: int = 14
    warmup_bars: int = 240

    # --- grid ---
    grid_step_usd: float = 1.90      # adverse move per added leg (median observed)
    max_depth: int = 6               # hard cap (live max 10; deep tail is the risk)
    max_adverse_usd: float = 20.0    # $/oz basket MAE: sizing risk-budget + grid-freeze distance
    hard_stop: bool = True
    hard_stop_usd: float = 0.0
    max_hold_hours: float = 48.0     # time-stop (0 = off)

    # --- sizing (BALANCED, risk-based) ---
    risk_per_basket_frac: float = 0.01
    target_fraction: float = 0.40
    lot_step: float = 0.01
    fixed_lot: float = 0.0           # if >0, override sizing with a flat lot

    # --- close ---
    tp_mode: str = "flat"            # "flat" | "vol_scaled"
    g_per_lot: float = 100.0         # $/lot target (== $1.0/oz)
    vol_scale_k: float = 0.90
    vol_clamp: tuple = (1.0, 12.0)
    sell_asym: bool = False
    g_per_lot_sell: float = 145.0

    # --- re-entry / execution fidelity ---
    cooldown_min: int = 1
    mark_side: str = "closeside"     # bid for long, ask for short

    def to_dict(self) -> dict:
        d = asdict(self)
        d["windows"] = {k: sorted(v) for k, v in self.windows.items()}
        d["vol_clamp"] = list(self.vol_clamp)
        return d

    def window_for_hour(self, broker_hour: int) -> str:
        for name, hrs in self.windows.items():
            if broker_hour in hrs:
                return name
        return "other"
