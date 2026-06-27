# Vendored from quantumstrategy (scripts/analyze_entry.add_m1_features +
# backtest/features.add_atr_1h) — the exact reverse-engineered entry features so
# the agent and the deterministic baseline can never drift from the EA.
"""M1 feature precompute.

`add_m1_features` reproduces the candidate entry features the Quantum Queen EA's
first-leg rule was reverse-engineered from: 4h trend momentum, MA200, RSI(14),
60-min range position ("breakout"), short-term volatility. `add_atr_1h` adds the
hourly ATR ($/oz) used by the volatility-scaled take-profit. `build_features`
assembles the full frame (cached to disk) and labels each bar's session window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .align import load_m1, utc_to_broker
from .config import BacktestConfig
from .paths import FEATURES_CACHE

TREND_LOOKBACK = 240  # M1 bars = 4h
RANGE_LOOKBACK = 60   # M1 bars = 60 min
RSI_LEN = 14


def add_m1_features(m: pd.DataFrame) -> pd.DataFrame:
    """Compute the candidate entry features on the M1 grid (1-min bars)."""
    m = m.copy()
    p = m["mid_close"]
    d = p.diff()
    up = d.clip(lower=0).rolling(RSI_LEN).mean()
    dn = (-d.clip(upper=0)).rolling(RSI_LEN).mean()
    m["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    m["ma200"] = p.rolling(TREND_LOOKBACK).mean()
    m["mom_4h"] = p - p.shift(TREND_LOOKBACK)
    hi = p.rolling(RANGE_LOOKBACK).max()
    lo = p.rolling(RANGE_LOOKBACK).min()
    m["hi60"] = hi
    m["lo60"] = lo
    m["rng_pos"] = (p - lo) / (hi - lo)
    m["vol60"] = d.abs().rolling(RANGE_LOOKBACK).mean()
    m["bhour"] = m["timestamp"].dt.hour
    return m


def add_atr_1h(m: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    """Hourly ATR in $/oz, forward-filled onto the M1 grid."""
    m = m.copy()
    s = m.set_index("timestamp")["mid_close"]
    h = s.resample("1h").ohlc().dropna(how="all")
    prev_close = h["close"].shift()
    tr = pd.concat(
        [h["high"] - h["low"], (h["high"] - prev_close).abs(), (h["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(n, min_periods=1).mean()
    m["atr_1h"] = atr.reindex(s.index, method="ffill").values
    return m


def build_features(cfg: BacktestConfig, m: pd.DataFrame | None = None,
                   use_cache: bool = True) -> pd.DataFrame:
    """Full M1 feature frame ready for the engine/enricher.

    The price-only features depend solely on fixed lookback params, so the frame
    is cached to disk and reused. The per-bar window labels (which depend on
    cfg.windows) are always recomputed. `timestamp` is naive UTC.
    """
    if use_cache and m is None and FEATURES_CACHE.exists():
        f = pd.read_parquet(FEATURES_CACHE)
    else:
        if m is None:
            m = load_m1()
        f = add_m1_features(m)
        f = add_atr_1h(f, n=cfg.rsi_len)
        if use_cache:
            f.to_parquet(FEATURES_CACHE, index=False)
    broker_ts = utc_to_broker(f["timestamp"])
    f["broker_hour"] = broker_ts.dt.hour
    f["window"] = f["broker_hour"].map(cfg.window_for_hour)
    return f.sort_values("timestamp").reset_index(drop=True)
