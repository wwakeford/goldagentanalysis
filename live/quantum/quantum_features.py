"""Incremental live recomputation of the Quantum Queen M1 features.

The backtest precomputes features over the full M1 history; live we must compute
the SAME values one bar at a time from aggregated 1-min bars. Two things matter:

1. **Causality.** The backtest's ``atr_1h`` is *lookahead*: it resamples to hourly
   OHLC and applies each hour's ATR to the minutes *inside* that hour — i.e. a
   minute at 14:30 sees the full 14:00–15:00 range, which isn't known until the
   hour closes. A live bot cannot do that. So the live feature engine uses a
   **causal** ATR: a minute in hour H sees the 14-hour rolling ATR computed
   through hour **H-1**. ``causal_feature_frame`` builds the matching vectorized
   oracle (and is the feed for the causal replay-parity reference), so live and
   backtest are validated to be the *same causal* system. The strategy's edge under
   this causal ATR is re-checked separately (see ``backtest`` re-run in the gates).

2. **Exactness.** ``ma200`` (240-bar mean), ``mom_4h`` (240-bar diff) and
   ``rng_pos`` (60-bar hi/lo) are causal in the backtest already and are reproduced
   bit-for-bit. ``window`` is the broker-hour (UTC+3) session label.

``QuantumFeatureState.update`` returns the latest ``FeatureRow`` each closed bar;
``__main__`` is Gate 1: incremental == causal vectorized oracle over the full M1.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from . import _paths  # noqa: F401  (sys.path)
from backtest.config import BacktestConfig
from price_align import utc_to_broker
from analyze_entry import add_m1_features, TREND_LOOKBACK, RANGE_LOOKBACK

_HOUR_NS = 3_600_000_000_000
_MIN_NS = 60_000_000_000


# --------------------------------------------------------------------------- #
# Causal vectorized oracle  (== what the live engine must reproduce)
# --------------------------------------------------------------------------- #
def causal_atr_1h(m: pd.DataFrame, n: int = 14) -> np.ndarray:
    """Hourly ATR ($/oz) forward-filled onto M1, but CAUSAL: each minute sees the
    n-hour ATR through the *previous* completed hour (shift by one present hour)."""
    s = m.set_index("timestamp")["mid_close"]
    h = s.resample("1h").ohlc().dropna(how="all")
    prev_close = h["close"].shift()
    tr = pd.concat(
        [h["high"] - h["low"], (h["high"] - prev_close).abs(), (h["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(n, min_periods=1).mean().shift(1)   # <-- causal: previous hour
    return atr.reindex(s.index, method="ffill").values


def causal_feature_frame(cfg: BacktestConfig, m: pd.DataFrame) -> pd.DataFrame:
    """The exact frame the live engine consumes, built vectorized with a CAUSAL atr.

    Reuses ``add_m1_features`` for the (already-causal) ma200/mom_4h/rng_pos, swaps in
    ``causal_atr_1h``, and labels the broker-time session window. This is BOTH the
    Gate-1 oracle and the Gate-2 causal-backtest reference feed.
    """
    f = add_m1_features(m)                         # ma200, mom_4h, rng_pos, rsi, vol60, bhour
    f["atr_1h"] = causal_atr_1h(m, n=cfg.rsi_len)
    broker_ts = utc_to_broker(f["timestamp"])
    f["broker_hour"] = broker_ts.dt.hour
    f["window"] = f["broker_hour"].map(cfg.window_for_hour)
    return f.sort_values("timestamp").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Incremental live feature state
# --------------------------------------------------------------------------- #
@dataclass
class FeatureRow:
    ts_ns: int
    bid_close: float
    ask_close: float
    mid_close: float
    ma200: float
    mom_4h: float
    rng_pos: float
    atr_1h: float
    window: str
    ma200_lag: float        # ma200 cfg.flip_htf_lookback bars ago (for the htf slope)
    n_bars: int             # bars seen so far (warmup gauge)

    @property
    def warm_entry(self) -> bool:
        return self.n_bars >= TREND_LOOKBACK and self.mom_4h == self.mom_4h

    @property
    def flip_armed(self) -> bool:
        return self.ma200_lag == self.ma200_lag      # not NaN


class QuantumFeatureState:
    """Maintain rolling buffers and emit one FeatureRow per closed M1 bar, matching
    ``causal_feature_frame`` row-for-row (validated by Gate 1)."""

    def __init__(self, cfg: BacktestConfig, keep: int = 4000):
        self.cfg = cfg
        self.htf_lb = cfg.flip_htf_lookback
        self.atr_n = cfg.rsi_len
        keep = max(keep, self.htf_lb + TREND_LOOKBACK + 8)
        self.mids: deque[float] = deque(maxlen=keep)
        # ma200_ring holds the last (htf_lb + 1) ma200 values; when full, ring[0] is
        # exactly ma200 htf_lb bars ago (== oracle ma200.shift(htf_lb)).
        self.ma200_ring: deque[float] = deque(maxlen=self.htf_lb + 1)
        self.n_bars = 0
        # causal hourly ATR state
        self._hour_key: Optional[int] = None
        self._h_high = self._h_low = self._h_close = np.nan
        self._completed_tr: deque[float] = deque(maxlen=self.atr_n)
        self._prev_hour_close = np.nan

    # -- causal hourly ATR ------------------------------------------------- #
    def _atr_now(self) -> float:
        """14-hour rolling mean of TR through the PREVIOUS completed hour."""
        if not self._completed_tr:
            return np.nan
        return float(np.mean(self._completed_tr))

    def _roll_hour(self, ts_ns: int, mid: float) -> None:
        """Update the in-progress hour; on an hour change, finalize it into the
        completed-TR ring (so ``_atr_now`` only ever reflects completed hours)."""
        hk = (ts_ns // _HOUR_NS) * _HOUR_NS
        if self._hour_key is None:
            self._hour_key = hk
            self._h_high = self._h_low = self._h_close = mid
            return
        if hk == self._hour_key:
            self._h_high = max(self._h_high, mid)
            self._h_low = min(self._h_low, mid)
            self._h_close = mid
            return
        # hour changed -> finalize the just-completed hour into the TR ring
        if self._prev_hour_close != self._prev_hour_close:   # first completed hour
            tr = self._h_high - self._h_low
        else:
            pc = self._prev_hour_close
            tr = max(self._h_high - self._h_low, abs(self._h_high - pc), abs(self._h_low - pc))
        self._completed_tr.append(tr)
        self._prev_hour_close = self._h_close
        # start the new (present) hour
        self._hour_key = hk
        self._h_high = self._h_low = self._h_close = mid

    # -- main update ------------------------------------------------------- #
    def update(self, ts_ns: int, bid_close: float, ask_close: float,
               mid_close: Optional[float] = None) -> FeatureRow:
        mid = mid_close if mid_close is not None else (bid_close + ask_close) / 2.0
        # ATR uses ONLY completed hours, so roll the hour BEFORE appending this bar's
        # mid (matches: minute in hour H sees ATR through H-1).
        self._roll_hour(ts_ns, mid)
        atr = self._atr_now()

        self.mids.append(mid)
        self.n_bars += 1
        nb = self.n_bars

        # ma200 = mean of last 240 mids (NaN until 240 bars) — causal, exact.
        if nb >= TREND_LOOKBACK:
            tail = list(self.mids)[-TREND_LOOKBACK:]
            ma200 = float(np.mean(tail))
        else:
            ma200 = np.nan
        self.ma200_ring.append(ma200)

        # mom_4h = mid_now - mid[-240]  (NaN until index >= 240)
        if nb > TREND_LOOKBACK:
            mom = mid - self.mids[-(TREND_LOOKBACK + 1)]
        else:
            mom = np.nan

        # rng_pos = (mid - lo60) / (hi60 - lo60) over the 60-bar window ending here
        if nb >= RANGE_LOOKBACK:
            win = list(self.mids)[-RANGE_LOOKBACK:]
            hi, lo = max(win), min(win)
            rng = (mid - lo) / (hi - lo) if hi != lo else np.nan
        else:
            rng = np.nan

        # ma200 cfg.flip_htf_lookback bars ago (for the htf slope in the engine):
        # when the ring is full, ring[0] is exactly ma200[i - htf_lb].
        ma200_lag = (self.ma200_ring[0]
                     if len(self.ma200_ring) == self.htf_lb + 1 else np.nan)

        broker_hour = (pd.Timestamp(ts_ns) + pd.Timedelta(hours=3)).hour
        window = self.cfg.window_for_hour(broker_hour)

        return FeatureRow(ts_ns=ts_ns, bid_close=bid_close, ask_close=ask_close,
                          mid_close=mid, ma200=ma200, mom_4h=mom, rng_pos=rng,
                          atr_1h=atr, window=window, ma200_lag=ma200_lag, n_bars=nb)


# --------------------------------------------------------------------------- #
# Gate 1: incremental features == causal vectorized oracle, over full M1
# --------------------------------------------------------------------------- #
def _gate1(sample: int = 0) -> bool:
    from price_align import load_m1
    cfg = BacktestConfig(flip_htf_lookback=2160)
    m = load_m1()
    if sample:
        m = m.iloc[:sample].reset_index(drop=True)
    oracle = causal_feature_frame(cfg, m)

    st = QuantumFeatureState(cfg)
    ma200 = np.empty(len(m)); mom = np.empty(len(m)); rng = np.empty(len(m))
    atr = np.empty(len(m)); lag = np.empty(len(m)); win = []
    bid = m["bid_close"].values; ask = m["ask_close"].values
    mid = m["mid_close"].values; ts = m["timestamp"].values.astype("datetime64[ns]").astype(np.int64)
    for i in range(len(m)):
        r = st.update(int(ts[i]), float(bid[i]), float(ask[i]), float(mid[i]))
        ma200[i] = r.ma200; mom[i] = r.mom_4h; rng[i] = r.rng_pos
        atr[i] = r.atr_1h; lag[i] = r.ma200_lag; win.append(r.window)

    ok = True
    for name, got, exp in [("ma200", ma200, oracle["ma200"].values),
                           ("mom_4h", mom, oracle["mom_4h"].values),
                           ("rng_pos", rng, oracle["rng_pos"].values),
                           ("atr_1h", atr, oracle["atr_1h"].values)]:
        close = np.isclose(got, exp, rtol=1e-9, atol=1e-6, equal_nan=True)
        nbad = int((~close).sum())
        mx = np.nanmax(np.abs(got - exp)) if np.isfinite(got - exp).any() else 0.0
        print(f"  {name:8s} mismatches={nbad:6d}  max|diff|={mx:.3e}  {'OK' if nbad == 0 else 'FAIL'}")
        ok &= nbad == 0
    # ma200_lag vs oracle ma200 shifted by htf_lb
    exp_lag = pd.Series(oracle["ma200"].values).shift(cfg.flip_htf_lookback).values
    close = np.isclose(lag, exp_lag, rtol=1e-9, atol=1e-6, equal_nan=True)
    print(f"  ma200_lag mismatches={int((~close).sum()):6d}  {'OK' if close.all() else 'FAIL'}")
    ok &= bool(close.all())
    win_ok = (np.array(win) == oracle["window"].values).all()
    print(f"  window    {'OK' if win_ok else 'FAIL'}")
    ok &= bool(win_ok)
    print("GATE 1:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    raise SystemExit(0 if _gate1(n) else 1)
