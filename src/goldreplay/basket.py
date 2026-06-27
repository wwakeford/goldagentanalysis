# Vendored from quantumstrategy/backtest/basket.py — pure logic, no I/O.
"""Basket state machine: a one-directional grid managed and closed as a unit.

    FLAT --entry--> OPEN(1 leg)
    OPEN --adverse >= grid_step & depth<max & notional ok--> OPEN(+1 leg)
    OPEN --floating$ >= G*total_lots--> CLOSED   (aggregate take-profit)

floating$ uses the close-side price (bid for a long, ask for a short) per the
close-rule reverse-engineering. Powers server-side P&L in the replay engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import pandas as pd

from .config import BacktestConfig


@dataclass
class Basket:
    direction: int                       # +1 long / -1 short
    open_time: pd.Timestamp
    cfg: BacktestConfig
    g_per_lot: float                     # resolved $/lot target for this basket
    legs: List[Tuple[float, float]] = field(default_factory=list)  # (price, lot)

    @property
    def n_legs(self) -> int:
        return len(self.legs)

    @property
    def total_lots(self) -> float:
        return sum(lot for _, lot in self.legs)

    @property
    def vwap(self) -> float:
        tot = self.total_lots
        return sum(p * lot for p, lot in self.legs) / tot if tot else 0.0

    @property
    def last_leg_price(self) -> float:
        return self.legs[-1][0]

    def floating_usd(self, closeside_price: float) -> float:
        """Aggregate floating P&L ($) at the given close-side price."""
        return self.direction * (closeside_price - self.vwap) * self.cfg.contract * self.total_lots

    def mae_usd_per_oz(self, closeside_price: float) -> float:
        """Adverse excursion in $/oz (positive = how far underwater)."""
        d = self.direction * (closeside_price - self.vwap)
        return max(0.0, -d)

    def gross_notional(self, price: float) -> float:
        return self.cfg.contract * self.total_lots * price

    def add_leg(self, price: float, lot: float) -> None:
        self.legs.append((price, lot))

    def should_close(self, closeside_price: float) -> bool:
        return self.floating_usd(closeside_price) >= self.g_per_lot * self.total_lots

    def should_stop(self, closeside_price: float) -> bool:
        stop_dist = self.cfg.hard_stop_usd if self.cfg.hard_stop_usd > 0 else self.cfg.max_adverse_usd
        return self.cfg.hard_stop and self.mae_usd_per_oz(closeside_price) >= stop_dist

    def should_add(self, closeside_price: float, price: float, lot: float) -> bool:
        if self.n_legs >= self.cfg.max_depth:
            return False
        adverse_from_last = self.direction * (self.last_leg_price - price)
        if adverse_from_last < self.cfg.grid_step_usd:
            return False
        if self.mae_usd_per_oz(closeside_price) >= self.cfg.max_adverse_usd:
            return False  # freeze grid: ride to TP rather than dig deeper
        new_notional = self.cfg.contract * (self.total_lots + lot) * price
        budget = self.cfg.target_fraction * self.cfg.leverage_cap * self.cfg.initial_capital
        if new_notional > budget:
            return False
        return True


def resolve_target(cfg: BacktestConfig, direction: int, atr_1h: float) -> float:
    """Resolve the $/lot take-profit target for a new basket."""
    if cfg.tp_mode == "vol_scaled" and atr_1h == atr_1h:  # not NaN
        d_oz = min(max(cfg.vol_scale_k * atr_1h, cfg.vol_clamp[0]), cfg.vol_clamp[1])
        return d_oz * cfg.contract
    g = cfg.g_per_lot
    if cfg.sell_asym and direction < 0:
        g = cfg.g_per_lot_sell
    return g
