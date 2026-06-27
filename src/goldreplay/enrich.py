"""Point-in-time enrichment for the no-lookahead walk-forward (gold analog of
intradayRS's `_enriched_bars`).

The episode's M1 bars are resampled to 15-minute DECISION BARS — what the agent
sees, one at a time. Every field on decision bar i is taken AS-OF the last M1
bar in its 15-minute interval, so it is computed only from M1 timestamps <= that
bar's close. The 4h/60-min/200-bar lookbacks (mom_4h, rng_pos, ma200) legitimately
reach into PRE-episode bars — that is past data, not future. Exposing the prefix
0..i therefore leaks nothing about the future.

The full M1 slice (`m1_records`) is kept SERVER-SIDE only: the engine walks it
between decision bars to manage the grid and compute realized P&L. The agent
never receives it.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from .align import utc_to_broker
from .config import BacktestConfig
from .episodes import episode_bounds_utc, parse_episode_id

DECISION_FREQ = "15min"
WARMUP_HOURS = 4


def _episode_feature_slice(feat: pd.DataFrame, eid: str) -> tuple[pd.DataFrame, pd.Timestamp]:
    """(M1 feature rows inside the episode window, episode start_utc)."""
    d, window = parse_episode_id(eid)
    start_utc, end_utc = episode_bounds_utc(d, window)
    ts = feat["timestamp"]
    m = feat.loc[(ts >= start_utc) & (ts < end_utc)].sort_values("timestamp").reset_index(drop=True)
    return m, start_utc


def m1_records(feat: pd.DataFrame, eid: str) -> list[dict]:
    """Server-only full-resolution M1 path for fills / grid / MFE-MAE.

    Each record: index `i`, ts, bid/ask/mid close, atr_1h. NEVER printed to the agent.
    """
    m, _ = _episode_feature_slice(feat, eid)
    out = []
    for i, (_, r) in enumerate(m.iterrows()):
        out.append({
            "i": i,
            "ts": r["timestamp"].isoformat(),
            "bid": round(float(r["bid_close"]), 3),
            "ask": round(float(r["ask_close"]), 3),
            "mid": round(float(r["mid_close"]), 3),
            "atr_1h": None if pd.isna(r["atr_1h"]) else round(float(r["atr_1h"]), 3),
        })
    return out


def decision_bars(feat: pd.DataFrame, eid: str, cfg: BacktestConfig) -> list[dict]:
    """15-minute decision bars carrying only as-of-i fields, each linked to the
    index of its as-of M1 bar in `m1_records` (so the engine can advance the grid)."""
    m, start_utc = _episode_feature_slice(feat, eid)
    if m.empty:
        return []
    open_mid = float(m.iloc[0]["mid_close"])

    # bin every M1 row to its 15-minute interval; the LAST row of each bin is the
    # as-of decision point (the bar the agent acts on once it has closed).
    binned = m.copy()
    binned["bin"] = binned["timestamp"].dt.floor(DECISION_FREQ)

    bars: list[dict] = []
    for _, grp in binned.groupby("bin", sort=True):
        last = grp.iloc[-1]                       # as-of row for this decision bar
        m1_idx = int(grp.index[-1])               # index into the episode M1 slice
        mid = float(last["mid_close"])
        mids = grp["mid_close"].astype(float)
        bh = int(utc_to_broker(last["timestamp"]).hour)
        win = cfg.window_for_hour(bh)
        hi60 = None if pd.isna(last["hi60"]) else round(float(last["hi60"]), 3)
        lo60 = None if pd.isna(last["lo60"]) else round(float(last["lo60"]), 3)
        bars.append({
            "i": len(bars),
            "m1_idx": m1_idx,
            "time": utc_to_broker(last["timestamp"]).strftime("%H:%M"),
            "ts_utc": last["timestamp"].isoformat(),
            "mid": round(mid, 3),
            "bid": round(float(last["bid_close"]), 3),
            "ask": round(float(last["ask_close"]), 3),
            "spread": round(float(last["spread"]), 3),
            "o15": round(float(mids.iloc[0]), 3),
            "h15": round(float(mids.max()), 3),
            "l15": round(float(mids.min()), 3),
            "c15": round(mid, 3),
            "mom_4h": None if pd.isna(last["mom_4h"]) else round(float(last["mom_4h"]), 2),
            "rng_pos": None if pd.isna(last["rng_pos"]) else round(float(last["rng_pos"]), 3),
            "rsi": None if pd.isna(last["rsi"]) else round(float(last["rsi"]), 1),
            "atr_1h": None if pd.isna(last["atr_1h"]) else round(float(last["atr_1h"]), 2),
            "ma200": None if pd.isna(last["ma200"]) else round(float(last["ma200"]), 2),
            "window": win,
            "broker_hour": bh,
            "cum_move": round(mid - open_mid, 2),
            "dist_hi60": None if hi60 is None else round(hi60 - mid, 2),
            "dist_lo60": None if lo60 is None else round(mid - lo60, 2),
            "min_trend_floor": cfg.min_trend_usd.get(win),
            "breakout_frac": cfg.breakout_frac.get(win),
        })
    return bars


def warmup_block(feat: pd.DataFrame, eid: str, cfg: BacktestConfig) -> str:
    """A compact pre-episode 4h context read (legitimate past data) for the prompt."""
    d, window = parse_episode_id(eid)
    start_utc, _ = episode_bounds_utc(d, window)
    ts = feat["timestamp"]
    pre = feat.loc[(ts >= start_utc - pd.Timedelta(hours=WARMUP_HOURS)) & (ts < start_utc)]
    if pre.empty:
        return "(no warmup bars available)"
    first = pre.iloc[0]
    last = pre.iloc[-1]
    mom = last["mom_4h"]
    trend = "up" if (mom == mom and mom > 0) else ("down" if (mom == mom and mom < 0) else "flat")
    return (
        f"Pre-episode 4h context (broker {utc_to_broker(first['timestamp']):%H:%M}"
        f"-{utc_to_broker(last['timestamp']):%H:%M}, legitimate past data):\n"
        f"  4h momentum mom_4h={mom:+.1f} ({trend} trend)  |  price {last['mid_close']:.2f}"
        f"  ma200={last['ma200']:.2f}  rsi={last['rsi']:.0f}\n"
        f"  60-min range pos rng_pos={last['rng_pos']:.2f}  atr_1h={last['atr_1h']:.2f}"
    )
