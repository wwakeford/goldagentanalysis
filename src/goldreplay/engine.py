# Vendored (trimmed) from quantumstrategy/backtest/engine.py — the research-only
# trend-flip / catastrophe-stop / regime-gate / force-flat branches are removed so
# this is the plain documented EA rule (the honest deterministic baseline).
"""M1-bar event loop with continuous mark-to-market equity.

Per bar, in strict order:
  1. mark-to-market the open basket at the close-side price -> equity, peak
  2. DQ / force-liquidation check (MT5 stop-out model)
  3. basket management: aggregate-TP close, else hard-stop, else timeout, else grid-add
  4. entry: the session-gated 4h-trend 60-min-breakout rule

Lookahead invariant: every decision at bar t uses only features/prices known at
bar t, and fills at bar t's own bid/ask. Long marks/closes at bid and fills at
ask; short marks/closes at ask and fills at bid (pay the spread, conservative).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .basket import Basket, resolve_target
from .sizing import leg_lot


@dataclass
class EngineResult:
    equity: pd.DataFrame
    baskets: pd.DataFrame
    dq: bool
    dq_reason: str
    dq_time: object


def run(cfg: BacktestConfig, feats: pd.DataFrame) -> EngineResult:
    f = feats.reset_index(drop=True)
    n = len(f)

    ts = f["timestamp"].values.astype("datetime64[ns]")
    ts_i64 = ts.astype(np.int64)
    bid = f["bid_close"].values
    ask = f["ask_close"].values
    mom = f["mom_4h"].values
    rng = f["rng_pos"].values
    atr = f["atr_1h"].values
    win = f["window"].values

    min_trend = np.full(n, np.nan)
    breakout = np.full(n, np.nan)
    for name in cfg.windows:
        mask = win == name
        min_trend[mask] = cfg.min_trend_usd[name] * cfg.entry_strictness
        breakout[mask] = cfg.breakout_frac[name]

    cooldown_ns = int(cfg.cooldown_min * 60 * 1_000_000_000)
    max_hold_ns = int(cfg.max_hold_hours * 3600 * 1_000_000_000)
    budget = cfg.target_fraction * cfg.leverage_cap * cfg.initial_capital

    cash = cfg.initial_capital
    basket: Basket | None = None
    cooldown_until = np.int64(0)

    eq_arr = np.empty(n)
    closed: list[dict] = []
    dq = False
    dq_reason = ""
    dq_time = None

    warm = min(cfg.warmup_bars, n)
    for i in range(warm, n):
        b_bid = bid[i]
        b_ask = ask[i]

        if basket is not None:
            cs = b_bid if basket.direction > 0 else b_ask
            floating = basket.floating_usd(cs)
            gross = basket.gross_notional(cs)
        else:
            floating = 0.0
            gross = 0.0
        equity = cash + floating
        eq_arr[i] = equity

        if basket is not None:
            used_margin = gross / cfg.account_leverage
            if equity < cfg.stop_out_frac * used_margin:
                dq, dq_reason, dq_time = True, "forced_liquidation", pd.Timestamp(ts[i])
                cash = equity
                eq_arr[i:] = equity
                break

        if basket is not None:
            cs = b_bid if basket.direction > 0 else b_ask
            entry_side = b_ask if basket.direction > 0 else b_bid
            timed_out = max_hold_ns > 0 and (ts_i64[i] - basket.open_time.value) >= max_hold_ns
            if basket.should_close(cs) or basket.should_stop(cs) or timed_out:
                pnl = basket.floating_usd(cs)
                cash += pnl
                reason = ("tp" if basket.should_close(cs)
                          else "stop" if basket.should_stop(cs) else "timeout")
                closed.append(_close_record(basket, ts[i], cs, pnl, reason))
                basket = None
                cooldown_until = ts_i64[i] + cooldown_ns
            elif basket.should_add(cs, entry_side, leg_lot(cfg, entry_side)):
                basket.add_leg(entry_side, leg_lot(cfg, entry_side))

        if basket is None and ts_i64[i] >= cooldown_until:
            mt = min_trend[i]
            if mt == mt:
                m = mom[i]
                if m == m and abs(m) >= mt:
                    trend = 1 if m > 0 else -1
                    if cfg.long_only and trend < 0:
                        continue
                    aligned = rng[i] if trend > 0 else 1.0 - rng[i]
                    if aligned == aligned and aligned >= breakout[i]:
                        entry_px = b_ask if trend > 0 else b_bid
                        lot = leg_lot(cfg, entry_px)
                        if cfg.contract * lot * entry_px <= budget:
                            g = resolve_target(cfg, trend, atr[i])
                            basket = Basket(direction=trend, open_time=pd.Timestamp(ts[i]),
                                            cfg=cfg, g_per_lot=g)
                            basket.add_leg(entry_px, lot)

    eq_arr[:warm] = cfg.initial_capital
    equity_df = pd.DataFrame({"timestamp": ts, "equity": eq_arr})
    baskets_df = pd.DataFrame(closed)
    return EngineResult(equity_df, baskets_df, dq, dq_reason, dq_time)


def _close_record(b: Basket, close_ts, close_price: float, pnl: float, reason: str) -> dict:
    total_lots = b.total_lots
    return {
        "direction": "buy" if b.direction > 0 else "sell",
        "close_reason": reason,
        "n_legs": b.n_legs,
        "first_open": b.open_time,
        "close_time": pd.Timestamp(close_ts),
        "duration_min": (pd.Timestamp(close_ts) - b.open_time).total_seconds() / 60.0,
        "total_volume": total_lots,
        "vwap": b.vwap,
        "close_price": close_price,
        "total_profit": pnl,
        "d_close_oz": pnl / (b.cfg.contract * total_lots) if total_lots else np.nan,
    }
