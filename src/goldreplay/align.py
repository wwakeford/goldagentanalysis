# Vendored from quantumstrategy/scripts/price_align.py — do not edit the UTC+3 math.
"""Aligning broker-time data to the UTC Dukascopy price feed.

The broker/server timezone was solved empirically (not assumed): shifting trade
times by -3h minimizes |entry_price - Dukascopy mid| (median error $0.18, inside
the spread), so **server time = UTC+3** (standard MT5 EET).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .paths import M1_PARQUET

# Broker server time = UTC + this many hours (solved empirically upstream).
SERVER_UTC_OFFSET_HOURS = 3


def broker_to_utc(ts: pd.Series | pd.Timestamp):
    """Convert naive broker-server timestamps to naive UTC."""
    return ts - pd.Timedelta(hours=SERVER_UTC_OFFSET_HOURS)


def utc_to_broker(ts: pd.Series | pd.Timestamp):
    """Convert naive UTC timestamps to naive broker-server time."""
    return ts + pd.Timedelta(hours=SERVER_UTC_OFFSET_HOURS)


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if getattr(df["timestamp"].dt, "tz", None) is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert(None)  # -> naive UTC
    return df.sort_values("timestamp").reset_index(drop=True)


def load_m1() -> pd.DataFrame:
    """M1 OHLC (bid/ask/mid), naive-UTC `timestamp`."""
    return _load(M1_PARQUET)
