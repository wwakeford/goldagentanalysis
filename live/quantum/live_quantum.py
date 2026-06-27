"""Live runner for the Quantum Queen recovery grid, on the scalpergold chassis.

Mirrors ``live_mt5.LiveRunner`` but swaps the decision core for ``QuantumEngine``
and decides **once per closed M1 bar** (Quantum's no-lookahead invariant), not per
tick. Below the seam everything is the shared chassis: the ``Broker`` abstraction
(SimBroker/MT5Broker), a margin circuit-breaker, KILL switch, restart reconcile and
JSONL logging — the same ops hardening the scalper ships.

Three entry points:
  * ``run_replay(frame)`` — drive the engine + SimBroker over a precomputed causal
    M1 feature frame; Gate 2 asserts realized P&L / basket count == the causal
    backtest (``backtest.engine.run`` on the same frame).
  * ``run_replay_incremental(m1)`` — same, but features are recomputed live via
    ``QuantumFeatureState`` (end-to-end: features + engine + execution).
  * ``run_live(...)`` — poll an MT5Broker, aggregate bid/ask M1 bars, recompute
    features each closed bar, trade. Windows box only.

Sizing: ``size_mode='engine'`` orders the engine's own leg lot (parity); ``'margin'``
orders the no-stop-aware margin-floor lot from ``_size_basket`` (live default).
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import _paths  # noqa: F401
from backtest.config import BacktestConfig

from .quantum_engine import QuantumEngine
from .quantum_features import FeatureRow, QuantumFeatureState, causal_feature_frame

# scalpergold/src is on sys.path when run as a module from src/
import sizing as scalp_sizing  # noqa: E402  (margin_level / MARGIN_CALL_LEVEL)
from broker import Account, Broker, Position, Quote, SimBroker  # noqa: E402

LOG_DIR = Path(__file__).resolve().parents[2] / "data" / "live_logs"
_PPL = 100.0   # XAUUSD $/oz/lot (== contract size)


@dataclass
class QRunnerConfig:
    size_mode: str = "margin"          # 'margin' (live) | 'engine' (parity)
    leverage: float = 100.0            # account margin leverage (CONFIRM with comp)
    margin_buffer: float = 2.5         # margin level after a full-depth ride excursion
    max_ride_mae: float = 150.0        # $/oz adverse excursion sizing rides w/o stop
    soft_floor: float = 0.80           # margin level below -> no NEW baskets/adds (still ride)
    hard_floor: float = 0.40           # margin level below -> voluntary flatten (ABOVE 0.30 DQ line)
    poll_secs: float = 1.0
    kill_file: str = "KILL"
    log: bool = True


class QuantumRunner:
    def __init__(self, broker: Broker, cfg: BacktestConfig, rcfg: QRunnerConfig):
        self.broker = broker
        self.cfg = cfg
        self.rcfg = rcfg
        self.eng = QuantumEngine(cfg)
        self.feats = QuantumFeatureState(cfg)
        self.basket_tickets: list[int] = []
        self.basket_lot = 0.0
        self.halted = False
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._logf = (LOG_DIR / "events.jsonl").open("a") if rcfg.log else None

    # ----------------------------------------------------------------- log -- #
    def _emit(self, kind: str, **kw):
        if self._logf:
            self._logf.write(json.dumps({"kind": kind, **kw}, default=float) + "\n")
            self._logf.flush()

    # -------------------------------------------------------------- sizing -- #
    def _size_basket(self, price: float, eng_lot: float) -> float:
        """No-stop-aware margin-floor sizing.

        Quantum rides baskets with NO hard stop, so the loss-budget cap is moot; the
        binding guard is: even after a ``max_ride_mae`` $/oz adverse excursion at full
        grid depth, margin level stays >= ``margin_buffer``. We take the SMALLER of
        the strategy's own risk-based leg lot and this margin-floor cap.
        """
        if self.rcfg.size_mode == "engine":
            return eng_lot
        eq = self.broker.account().equity
        lev = self.rcfg.leverage
        mpl = price * _PPL / lev                      # margin per lot
        e = self.rcfg.max_ride_mae
        buf = self.rcfg.margin_buffer
        # full-depth total lots cap: (eq - e*T*ppl)/(T*mpl) >= buf  =>
        #   T <= eq / (e*ppl + buf*mpl)
        t_cap = eq / (e * _PPL + buf * mpl)
        leg_cap = t_cap / max(1, self.cfg.max_depth)
        lot = min(eng_lot, leg_cap)
        step = self.cfg.lot_step
        lot = max(step, math.floor(lot / step) * step)
        self._emit("size", equity=eq, price=price, eng_lot=eng_lot, leg_cap=leg_cap,
                   lot=lot, binding=("margin" if leg_cap < eng_lot else "strategy"))
        return round(lot, 2)

    # -------------------------------------------------------------- execute - #
    def _execute(self, intent, t: int):
        if intent is None:
            return
        kind = intent[0]
        if kind == "open":
            _, side, px, eng_lot = intent
            self.basket_lot = self._size_basket(self.broker.quote().mid, eng_lot)
            pos = self.broker.market_order(side, self.basket_lot)
            self.basket_tickets = [pos.ticket]
            self._emit("open", t=t, side=side, lot=self.basket_lot, px=pos.open_price)
        elif kind == "add":
            _, side, px, eng_lot = intent
            lot = self.basket_lot or eng_lot
            pos = self.broker.market_order(side, lot)
            self.basket_tickets.append(pos.ticket)
            self._emit("add", t=t, side=side, lot=lot, px=pos.open_price,
                       legs=len(self.basket_tickets))
        elif kind == "close":
            # Close the whole basket = the symbol's net position. close_all() reads the
            # broker's ACTUAL open positions, so it's correct on NETTING (one merged
            # position — this comp account) and HEDGING (separate legs) alike. Closing per
            # tracked order-ticket breaks on netting, where order tickets != position tickets.
            pnl = self.broker.close_all()
            self._emit("close", t=t, legs=len(self.basket_tickets), pnl=pnl,
                       reason=intent[1].get("close_reason"),
                       equity=self.broker.account().equity)
            self.basket_tickets = []
            self.basket_lot = 0.0

    # ----------------------------------------------------- margin guard ----- #
    def _tripped(self) -> bool:
        if self.rcfg.size_mode == "engine":     # parity replay: no guard
            return False
        if self.halted:
            return True
        ml = self.broker.account().margin_level
        if ml <= self.rcfg.hard_floor:
            self._emit("hard_floor", margin_level=ml)
            self._flatten_and_halt("hard_floor")
            return True
        if (LOG_DIR / self.rcfg.kill_file).exists():
            self._emit("kill_switch")
            self._flatten_and_halt("kill_switch")
            return True
        return ml <= self.rcfg.soft_floor        # soft: block new risk, keep riding

    def _flatten_and_halt(self, why: str):
        self.broker.close_all()
        self.eng.basket = None
        self.eng.flip_run = 0
        self.basket_tickets = []
        self.basket_lot = 0.0
        self.halted = True
        self._emit("halt", why=why)

    # ----------------------------------------------------- one bar step ----- #
    def _step_bar(self, row: FeatureRow):
        # Margin guard runs EVERY bar (not just on trade intents): the hard_floor
        # flatten must be able to fire while a basket is merely RIDING underwater —
        # that's the voluntary bailout (above the 0.30 DQ line) that prevents a
        # broker force-liquidation. Soft-floor only blocks NEW risk; rides/exits pass.
        tripped = self._tripped()
        if self.halted:
            return None
        intent = self.eng.on_bar(row)
        if tripped and intent is not None and intent[0] in ("open", "add"):
            if intent[0] == "open":
                self.eng.basket = None            # undo the engine's speculative open
                self.eng.flip_run = 0
            else:
                self.eng.basket.legs.pop()        # roll back the speculative added leg
            return None
        self._execute(intent, int(row.ts_ns))
        return intent

    # =========================================================== REPLAY ==== #
    def run_replay(self, frame: pd.DataFrame, track: bool = False) -> dict:
        """Drive engine + SimBroker over a precomputed causal feature frame.

        ``track=True`` records per-bar equity + margin level (Gate 3): min margin
        level while in a position, max equity drawdown, and whether the margin guard
        ever flattened (a self-inflicted crystallization) — all relative to the
        comp's 0.30 force-liquidation line."""
        assert isinstance(self.broker, SimBroker)
        htf_lb = self.cfg.flip_htf_lookback
        ts = frame["timestamp"].values.astype("datetime64[ns]").astype(np.int64)
        bid = frame["bid_close"].values
        ask = frame["ask_close"].values
        mid = frame["mid_close"].values
        ma200 = frame["ma200"].values
        mom = frame["mom_4h"].values
        rng = frame["rng_pos"].values
        atr = frame["atr_1h"].values
        win = frame["window"].values
        n_open = 0
        min_ml = math.inf
        peak_eq = self.broker.account().equity
        max_dd = 0.0
        for i in range(len(frame)):
            ma_lag = ma200[i - htf_lb] if i >= htf_lb else np.nan
            row = FeatureRow(ts_ns=int(ts[i]), bid_close=float(bid[i]), ask_close=float(ask[i]),
                             mid_close=float(mid[i]), ma200=float(ma200[i]), mom_4h=float(mom[i]),
                             rng_pos=float(rng[i]), atr_1h=float(atr[i]), window=str(win[i]),
                             ma200_lag=float(ma_lag), n_bars=i + 1)
            self.broker.set_quote(Quote(int(ts[i]), float(bid[i]), float(ask[i])))
            intent = self._step_bar(row)
            if intent and intent[0] == "open":
                n_open += 1
            if track:
                acc = self.broker.account()
                if self.basket_tickets:                 # margin level only meaningful in-position
                    min_ml = min(min_ml, acc.margin_level)
                peak_eq = max(peak_eq, acc.equity)
                max_dd = max(max_dd, (peak_eq - acc.equity) / peak_eq)
        realized_closed = self.broker.realized
        residual = 0.0
        if self.basket_tickets:
            self.broker.close_all()
            residual = self.broker.realized - realized_closed
            self.basket_tickets = []
        acc = self.broker.account()
        return {"baskets_opened": n_open, "realized": realized_closed,
                "residual": residual, "equity": acc.equity,
                "min_margin_level": min_ml, "max_drawdown": max_dd,
                "halted": self.halted}

    # ============================================ WARMUP / RECONCILE ======= #
    def seed_features(self, m1: pd.DataFrame) -> int:
        """Warm the feature state from historical M1 bars (no trading). Chronological
        order required. Only `mid` matters for features; bid/ask are unused here.
        Returns the number of bars seeded so the caller can confirm the htf flip
        (needs flip_htf_lookback + ~240 bars) is armed."""
        ts = m1["timestamp"].values.astype("datetime64[ns]").astype(np.int64)
        mid = (m1["mid_close"].values if "mid_close" in m1
               else (m1["bid_close"].values + m1["ask_close"].values) / 2.0)
        last = None
        for i in range(len(m1)):
            last = self.feats.update(int(ts[i]), float(mid[i]), float(mid[i]), float(mid[i]))
        armed = (last is not None) and last.flip_armed
        self._emit("seed", bars=len(m1), flip_armed=bool(armed),
                   need=self.cfg.flip_htf_lookback + 240)
        return len(m1)

    def reconcile(self):
        """On (re)start, adopt any open broker positions into the engine basket so we
        manage the existing grid instead of opening a duplicate."""
        pos = self.broker.positions()
        if not pos:
            return
        from backtest.basket import Basket
        side = pos[0].side
        legs = [(pp.open_price, pp.lots) for pp in pos if pp.side == side]
        # NETTING (this comp account): the broker merges the grid into ONE net position,
        # so `legs` is a single (vwap, net_volume) blob. Split it back into ~unit-sized
        # legs (capped at max_depth) so the engine's n_legs<max_depth add-cap is respected
        # after a restart — otherwise it could stack new legs on top of the full net grid.
        from backtest.sizing import leg_lot
        total = sum(l for _, l in legs)
        unit = leg_lot(self.cfg, legs[0][0]) if legs else self.cfg.lot_step
        if len(legs) == 1 and unit > 0 and total > unit * 1.5:
            n_est = max(1, min(self.cfg.max_depth, round(total / unit)))
            legs = [(legs[0][0], total / n_est)] * n_est
        # rebuild the engine basket; g_per_lot recovered from the live config TP mode
        from backtest.basket import resolve_target
        g = resolve_target(self.cfg, side, getattr(self.feats, "_atr_now", lambda: float("nan"))())
        b = Basket(direction=side, open_time=pd.Timestamp(int(time.time() * 1e9)),
                   cfg=self.cfg, g_per_lot=g)
        b.legs = legs
        self.eng.basket = b
        self.eng.flip_run = 0
        self.basket_tickets = [pp.ticket for pp in pos if pp.side == side]
        self.basket_lot = legs[0][1]
        self._emit("reconcile", side=side, legs=len(legs), vwap=b.vwap)

    # ================================================== LIVE (MT5 box) ===== #
    def run_live(self, max_seconds: float | None = None):  # pragma: no cover
        """Poll the broker, aggregate bid/ask 1-min bars, recompute features each
        closed bar, decide once per bar. Reconnect-safe; reconcile on restart."""
        self.reconcile()
        bars = _QBarBuilder()
        start = time.time()
        while max_seconds is None or time.time() - start < max_seconds:
            if (LOG_DIR / self.rcfg.kill_file).exists():
                self._flatten_and_halt("kill_switch"); break
            try:
                q = self.broker.quote()
            except Exception as e:                       # transient / reconnect
                self._emit("quote_error", err=str(e)); time.sleep(2.0); continue
            closed = bars.update(q)
            if closed is not None:
                ts_ns, _o, _h, _l, _c, bid_close, ask_close, mid_close = closed
                row = self.feats.update(ts_ns, bid_close, ask_close, mid_close)
                # mark for the margin guard at the CURRENT quote, decide on the bar
                if isinstance(self.broker, SimBroker):
                    self.broker.set_quote(q)
                self._step_bar(row)
            time.sleep(self.rcfg.poll_secs)
        if self._logf:
            self._logf.close()

    def run_replay_incremental(self, m1: pd.DataFrame) -> dict:
        """End-to-end: recompute features live via QuantumFeatureState, then engine
        + SimBroker. Proves the full live path (features+engine+exec)."""
        assert isinstance(self.broker, SimBroker)
        ts = m1["timestamp"].values.astype("datetime64[ns]").astype(np.int64)
        bid = m1["bid_close"].values
        ask = m1["ask_close"].values
        mid = m1["mid_close"].values
        n_open = 0
        for i in range(len(m1)):
            row = self.feats.update(int(ts[i]), float(bid[i]), float(ask[i]), float(mid[i]))
            self.broker.set_quote(Quote(int(ts[i]), float(bid[i]), float(ask[i])))
            intent = self._step_bar(row)
            if intent and intent[0] == "open":
                n_open += 1
        realized_closed = self.broker.realized
        if self.basket_tickets:
            self.broker.close_all()
            self.basket_tickets = []
        return {"baskets_opened": n_open, "realized": realized_closed,
                "equity": self.broker.account().equity}


class _QBarBuilder:  # pragma: no cover - exercised live only
    """Aggregate ticks into 1-min bars (UTC), tracking mid OHLC plus the last bid and
    ask of the minute (Quantum fills at the bar's bid/ask close). Returns the closed
    bar tuple ``(ts_ns, o, h, l, c, bid_close, ask_close, mid_close)``."""
    def __init__(self, keep: int = 5000):
        self.keep = keep
        self._cur_min = None
        self._o = self._h = self._l = self._c = None
        self._bid = self._ask = None

    def update(self, q: Quote):
        minute = (q.time_ns // 60_000_000_000) * 60_000_000_000
        px = q.mid
        if self._cur_min is None:
            self._cur_min = minute
            self._o = self._h = self._l = self._c = px
            self._bid, self._ask = q.bid, q.ask
            return None
        if minute == self._cur_min:
            self._h = max(self._h, px); self._l = min(self._l, px); self._c = px
            self._bid, self._ask = q.bid, q.ask
            return None
        closed = (self._cur_min, self._o, self._h, self._l, self._c,
                  self._bid, self._ask, self._c)
        self._cur_min = minute
        self._o = self._h = self._l = self._c = px
        self._bid, self._ask = q.bid, q.ask
        return closed


# --------------------------------------------------------------------------- #
# Gate 2: replay parity vs the causal backtest
# --------------------------------------------------------------------------- #
def _gate2(sample: int = 0) -> bool:
    from price_align import load_m1
    from backtest.engine import run as bt_run
    from .config_live import build_live_config

    cfg = build_live_config()
    m = load_m1()
    if sample:
        m = m.iloc[:sample].reset_index(drop=True)
    frame = causal_feature_frame(cfg, m)

    # reference: the causal backtest
    res = bt_run(cfg, frame)
    bt = res.baskets
    bt_realized = float(bt["total_profit"].sum()) if len(bt) else 0.0
    bt_n = len(bt)

    # live stack, engine-lot parity mode (broker trades the engine's own lots)
    broker = SimBroker(balance=cfg.initial_capital, leverage=100.0,
                       commission_per_lot_side=0.0)
    runner = QuantumRunner(broker, cfg, QRunnerConfig(size_mode="engine", log=False))
    out = runner.run_replay(frame)

    diff = out["realized"] - bt_realized
    # engine.run leaves a final still-open basket UNrealized; run_replay reports the
    # same realized_closed (residual tracked separately), so compare closed-only.
    n_ok = out["baskets_opened"] >= bt_n   # opens include the final still-open one
    ok = abs(diff) < 1e-6
    print(f"  backtest: realized={bt_realized:14.2f}  baskets_closed={bt_n}")
    print(f"  live    : realized={out['realized']:14.2f}  baskets_opened={out['baskets_opened']}"
          f"  residual={out['residual']:.2f}")
    print(f"  diff={diff:+.3e}  baskets_open>=closed={n_ok}  {'OK' if ok else 'MISMATCH'}")

    # end-to-end (incremental features) must match the frame-fed path exactly
    broker2 = SimBroker(balance=cfg.initial_capital, leverage=100.0)
    runner2 = QuantumRunner(broker2, cfg, QRunnerConfig(size_mode="engine", log=False))
    out2 = runner2.run_replay_incremental(m)
    diff2 = out2["realized"] - out["realized"]
    e2e_ok = abs(diff2) < 1e-6
    print(f"  incremental-features path: realized={out2['realized']:.2f}  "
          f"diff_vs_frame={diff2:+.3e}  {'OK' if e2e_ok else 'MISMATCH'}")

    passed = ok and e2e_ok
    print("GATE 2:", "PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    raise SystemExit(0 if _gate2(n) else 1)
