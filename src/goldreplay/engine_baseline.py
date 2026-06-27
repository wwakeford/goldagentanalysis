"""Deterministic EA-rule baseline on a single episode — the ENGINE column of the
three-way comparison.

Runs the vendored `engine.run` (the plain documented Quantum Queen rule) on the
episode window plus a 4h warmup lead-in, then keeps only the baskets the engine
OPENED inside the window. This is the "what the mechanical rule would have done"
counterfactual against the Claude agent and the real live EA.
"""
from __future__ import annotations

import pandas as pd

from . import episodes as ep
from .config import BacktestConfig
from .engine import run
from .features import build_features


def run_episode(eid: str, cfg: BacktestConfig | None = None,
                feat: pd.DataFrame | None = None) -> dict:
    cfg = cfg or BacktestConfig()
    if feat is None:
        feat = build_features(cfg)
    d, window = ep.parse_episode_id(eid)
    start_utc, end_utc = ep.episode_bounds_utc(d, window)
    lead = start_utc - pd.Timedelta(hours=4)

    ts = feat["timestamp"]
    sl = feat.loc[(ts >= lead) & (ts < end_utc)].reset_index(drop=True)
    res = run(cfg, sl)

    b = res.baskets
    if b.empty:
        return {"eid": eid, "n_baskets": 0, "total_profit_usd": 0.0,
                "total_per_oz": 0.0, "win_rate": None, "avg_hold_min": None,
                "directions": [], "baskets": b}
    # keep only baskets OPENED within the episode window
    b = b[(b["first_open"] >= start_utc) & (b["first_open"] < end_utc)].reset_index(drop=True)
    if b.empty:
        return {"eid": eid, "n_baskets": 0, "total_profit_usd": 0.0,
                "total_per_oz": 0.0, "win_rate": None, "avg_hold_min": None,
                "directions": [], "baskets": b}
    return {
        "eid": eid,
        "n_baskets": int(len(b)),
        "total_profit_usd": round(float(b["total_profit"].sum()), 2),
        "total_per_oz": round(float(b["d_close_oz"].sum()), 3),
        "win_rate": round(float((b["total_profit"] > 0).mean()), 3),
        "avg_hold_min": round(float(b["duration_min"].mean()), 1),
        "directions": b["direction"].tolist(),
        "baskets": b,
    }
