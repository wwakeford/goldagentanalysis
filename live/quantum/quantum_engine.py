"""Per-bar decision state machine — the live mirror of backtest/engine.py.

This is the scalar, streaming version of the per-bar block of
``quantumstrategy/backtest/engine.py`` (lines 75–186): given one M1 ``FeatureRow``
it returns at most ONE intent — ``("open", trend, px)`` / ``("add", dir, px)`` /
``("close", record)`` — exactly the tuples the scalper runner's ``_execute`` already
consumes. It reuses ``Basket`` / ``resolve_target`` / ``leg_lot`` / ``_close_record``
unmodified, so the decision logic cannot drift from the validated backtest.

What it deliberately does NOT do: mark-to-market equity and the force-liquidation /
DQ check. Those are the live broker + runner margin guard's job (``_tripped``); the
engine is pure price-driven decision. Sizing of the *internal* basket uses the same
``leg_lot`` as the backtest — every close/add/entry decision is lot-independent
(thresholds scale with total_lots) except the slack notional-budget gate, so the
real order lot (margin-capped in the runner) never changes decision timing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import _paths  # noqa: F401
from backtest.config import BacktestConfig
from backtest.basket import Basket, resolve_target
from backtest.sizing import leg_lot
from backtest.engine import _close_record

from .quantum_features import FeatureRow

_MIN_NS = 60_000_000_000


class QuantumEngine:
    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg
        self.basket: Basket | None = None
        self.flip_run = 0
        self.cooldown_ns = int(cfg.cooldown_min * 60 * 1_000_000_000)
        self.max_hold_ns = int(cfg.max_hold_hours * 3600 * 1_000_000_000)
        self.budget = cfg.target_fraction * cfg.leverage_cap * cfg.initial_capital
        self.cooldown_until = np.int64(0)

    # convenience for the runner / reconcile
    @property
    def side(self) -> int:
        return self.basket.direction if self.basket is not None else 0

    def on_bar(self, row: FeatureRow):
        """Advance one closed M1 bar. Returns an intent tuple or None."""
        cfg = self.cfg
        ts = np.int64(row.ts_ns)
        b_bid, b_ask = row.bid_close, row.ask_close
        atr = row.atr_1h
        intent = None

        # ----------------------- basket management ----------------------- #
        if self.basket is not None:
            d = self.basket.direction
            cs = b_bid if d > 0 else b_ask
            entry_side = b_ask if d > 0 else b_bid

            mae_now = self.basket.mae_usd_per_oz(cs)
            cat_stop = (cfg.cat_stop_atr_mult > 0 and atr == atr
                        and mae_now >= cfg.cat_stop_atr_mult * atr)

            flip_exit = False
            if cfg.trend_flip_exit:
                age_min = (int(ts) - self.basket.open_time.value) / 6e10
                mae_gate = cfg.flip_max_mae_usd
                if cfg.flip_mae_atr_mult > 0 and atr == atr:
                    mae_gate = max(mae_gate, cfg.flip_mae_atr_mult * atr)
                if age_min >= cfg.flip_min_age_min and mae_now <= mae_gate:
                    if cfg.flip_mode == "mom":
                        m = row.mom_4h
                        flipped = (m == m) and (d * m <= -cfg.flip_mom_margin_usd)
                    elif cfg.flip_mode == "htf":
                        slope = row.ma200 - row.ma200_lag
                        flipped = (slope == slope) and (d * slope <= -cfg.flip_mom_margin_usd)
                    else:  # "ma200": price on the wrong side of MA200 for this basket
                        md, ma = row.mid_close, row.ma200
                        flipped = (md == md) and (ma == ma) and (d * (md - ma) < 0)
                    self.flip_run = self.flip_run + 1 if flipped else 0
                    flip_exit = self.flip_run >= max(1, cfg.flip_persist_bars)
                else:
                    self.flip_run = 0

            timed_out = self.max_hold_ns > 0 and (int(ts) - self.basket.open_time.value) >= self.max_hold_ns

            if (self.basket.should_close(cs) or self.basket.should_stop(cs)
                    or timed_out or flip_exit or cat_stop):
                pnl = self.basket.floating_usd(cs)
                if self.basket.should_close(cs):
                    reason = "tp"
                elif self.basket.should_stop(cs):
                    reason = "stop"
                elif timed_out:
                    reason = "timeout"
                elif flip_exit:
                    reason = "flip"
                else:
                    reason = "catstop"
                record = _close_record(self.basket, pd.Timestamp(int(ts)), cs, pnl, reason)
                self.basket = None
                self.flip_run = 0
                self.cooldown_until = ts + self.cooldown_ns
                return ("close", record)

            lot = leg_lot(cfg, entry_side)
            if self.basket.should_add(cs, entry_side, lot):
                self.basket.add_leg(entry_side, lot)
                return ("add", d, entry_side, lot)

        # ------------------------------ entry ---------------------------- #
        if self.basket is None and ts >= self.cooldown_until:
            w = row.window
            if w in cfg.min_trend_usd:
                mt = cfg.min_trend_usd[w] * cfg.entry_strictness
                m = row.mom_4h
                if m == m and abs(m) >= mt:
                    trend = 1 if m > 0 else -1
                    blocked = cfg.long_only and trend < 0
                    # regime gate (mirror of backtest/engine.py): don't open a basket
                    # against a confirmed HTF trend (MA200 slope over flip_htf_lookback
                    # == regime_gate_lookback bars). row.ma200_lag is NaN until the ring
                    # is full, so rslope!=rslope -> gate inactive during warm-up (matches
                    # the backtest `i >= regime_gate_lookback` guard).
                    if cfg.regime_gate and not blocked:
                        rslope = row.ma200 - row.ma200_lag
                        if rslope == rslope and trend * rslope <= -cfg.regime_gate_margin:
                            blocked = True
                    if not blocked:
                        rng = row.rng_pos
                        aligned = rng if trend > 0 else 1.0 - rng
                        if aligned == aligned and aligned >= cfg.breakout_frac[w]:
                            entry_px = b_ask if trend > 0 else b_bid
                            lot = leg_lot(cfg, entry_px)
                            if cfg.contract * lot * entry_px <= self.budget:
                                g = resolve_target(cfg, trend, atr)
                                self.basket = Basket(direction=trend,
                                                     open_time=pd.Timestamp(int(ts)),
                                                     cfg=cfg, g_per_lot=g)
                                self.basket.add_leg(entry_px, lot)
                                self.flip_run = 0
                                return ("open", trend, entry_px, lot)
        return intent
