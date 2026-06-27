"""M1 history backfill for live feature warm-up.

The htf trend-flip exit (the tail-protector that killed the DOWN_2026 steamroller)
needs ~``flip_htf_lookback`` + 240 M1 bars of history before it is armed. Rather
than trade blind for the first ~1.5 days, we backfill at startup. Two sources:

  1. the live broker itself (``MT5Broker.history_m1``) — same feed we trade on;
  2. Dukascopy (the feed the backtest was built on) as a fallback / for SimBroker
     dry-runs on macOS.
"""
from __future__ import annotations

import pandas as pd

from . import _paths  # noqa: F401


def from_broker(broker, n_bars: int) -> pd.DataFrame | None:
    fn = getattr(broker, "history_m1", None)
    if fn is None:
        return None
    try:
        df = fn(n_bars)
        return df if df is not None and len(df) else None
    except Exception:
        return None


def from_dukascopy(n_bars: int, symbol: str = "XAUUSD") -> pd.DataFrame | None:
    """Fetch the last ~n_bars minutes from Dukascopy (needs dukascopy-python).
    Returns naive-UTC timestamp + bid/ask/mid close."""
    try:
        import datetime as dt
        import dukascopy_python
        from dukascopy_python.instruments import INSTRUMENT_FX_METALS_XAU_USD
    except Exception:
        return None
    try:
        # pad generously; markets are closed on weekends so calendar days > bars/1440
        days = max(3, int(n_bars / 1440 * 1.6) + 2)
        end = dt.datetime.utcnow()
        start = end - dt.timedelta(days=days)
        df = dukascopy_python.fetch(
            INSTRUMENT_FX_METALS_XAU_USD,
            dukascopy_python.INTERVAL_MIN_1,
            dukascopy_python.OFFER_SIDE_BID, start, end,
        )
        if df is None or len(df) == 0:
            return None
        bid = df["close"].astype(float)
        # fetch ask side too for a real spread; fall back to bid if unavailable
        try:
            dfa = dukascopy_python.fetch(
                INSTRUMENT_FX_METALS_XAU_USD, dukascopy_python.INTERVAL_MIN_1,
                dukascopy_python.OFFER_SIDE_ASK, start, end)
            ask = dfa["close"].astype(float).reindex(bid.index).ffill()
        except Exception:
            ask = bid
        ts = pd.to_datetime(bid.index)
        if getattr(ts, "tz", None) is not None:
            ts = ts.tz_convert(None)
        out = pd.DataFrame({"timestamp": ts, "bid_close": bid.values,
                            "ask_close": ask.values,
                            "mid_close": (bid.values + ask.values) / 2.0})
        return out.tail(n_bars).reset_index(drop=True)
    except Exception:
        return None


def backfill(broker, cfg, extra: int = 300) -> pd.DataFrame:
    """Get >= flip_htf_lookback + 240 + extra M1 bars, broker first then Dukascopy."""
    need = cfg.flip_htf_lookback + 240 + extra
    df = from_broker(broker, need)
    src = "broker"
    if df is None or len(df) < cfg.flip_htf_lookback + 240:
        dk = from_dukascopy(need)
        if dk is not None and (df is None or len(dk) > len(df)):
            df, src = dk, "dukascopy"
    if df is None:
        raise RuntimeError("backfill failed: no broker history and no Dukascopy feed")
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.attrs["source"] = src
    return df
