# Vendored from quantumstrategy/backtest/sizing.py — BALANCED, risk-based sizing.
"""Position sizing off a fixed MAX-LOSS budget, paired with a hard per-basket stop.

    risk$            = risk_per_basket_frac * initial_capital
    max_basket_lots  = risk$ / (max_adverse_usd * contract)
    leg_lot          = max_basket_lots / max_depth

Price-independent (no low-gold blow-up); bounds a stopped basket's loss to ~risk$.
"""
from __future__ import annotations

from .config import BacktestConfig


def round_to_step(x: float, step: float) -> float:
    return round(round(x / step) * step, 10)


def leg_lot(cfg: BacktestConfig, price: float) -> float:
    """Flat per-leg lot size for a basket entered at `price`."""
    if cfg.fixed_lot > 0:
        return cfg.fixed_lot

    risk_usd = cfg.risk_per_basket_frac * cfg.initial_capital
    max_basket_lots = risk_usd / (cfg.max_adverse_usd * cfg.contract)

    budget = cfg.target_fraction * cfg.leverage_cap * cfg.initial_capital
    max_lots_by_lev = budget / (cfg.contract * price)
    max_basket_lots = min(max_basket_lots, max_lots_by_lev)

    lot = round_to_step(max_basket_lots / cfg.max_depth, cfg.lot_step)
    return max(lot, cfg.lot_step)
