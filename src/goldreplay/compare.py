"""The three-way comparison — the technology showcase.

For each episode, line up three independent traders on the SAME gold session:

  AGENT    — the Claude subagent that walked it bar-by-bar with no lookahead
             (reports/<eid>/<eid>.replay.json)
  ENGINE   — the deterministic documented EA rule (engine_baseline.run_episode)
  LIVE_EA  — the real Quantum Queen baskets that actually traded (ground truth)

Absolute $ is NOT comparable across the three (the engine runs a $1M account, the
live EA ran ~$1k, the agent is risk-sized to $1M), so the headline metric is
**$/oz** (price-move per ounce, account-size-independent) plus win rate, basket
count, and average hold.
"""
from __future__ import annotations

import argparse

import pandas as pd

from . import episodes as ep
from .config import BacktestConfig
from .engine_baseline import run_episode
from .features import build_features
from .paths import REPORTS, THREE_WAY_MD


def _agent_row(eid: str) -> dict | None:
    p = REPORTS / eid / f"{eid}.replay.json"
    if not p.exists():
        return None
    import json
    doc = json.loads(p.read_text())
    bs = doc.get("baskets", [])
    if not bs:
        return {"source": "AGENT", "n_baskets": 0, "total_per_oz": 0.0, "win_rate": None,
                "avg_hold_min": None, "dirs": ""}
    wins = sum(1 for b in bs if b["pnl_per_oz"] > 0)
    return {
        "source": "AGENT",
        "n_baskets": len(bs),
        "total_per_oz": round(sum(b["pnl_per_oz"] for b in bs), 3),
        "win_rate": round(wins / len(bs), 3),
        "avg_hold_min": round(sum(b["hold_min"] for b in bs) / len(bs), 1),
        "dirs": ",".join(b["direction"][0] for b in bs),  # l/s
    }


def _engine_row(eid: str, cfg: BacktestConfig, feat: pd.DataFrame) -> dict:
    r = run_episode(eid, cfg, feat)
    return {
        "source": "ENGINE", "n_baskets": r["n_baskets"], "total_per_oz": r["total_per_oz"],
        "win_rate": r["win_rate"], "avg_hold_min": r["avg_hold_min"],
        "dirs": ",".join(d[0] for d in r["directions"]),  # b/s
    }


def _live_row(eid: str, baskets: pd.DataFrame, cfg: BacktestConfig) -> dict:
    b = ep.align_ea_baskets(eid, baskets)
    if b.empty:
        return {"source": "LIVE_EA", "n_baskets": 0, "total_per_oz": 0.0, "win_rate": None,
                "avg_hold_min": None, "dirs": ""}
    per_oz = b["total_profit"] / (cfg.contract * b["total_volume"])
    return {
        "source": "LIVE_EA",
        "n_baskets": int(len(b)),
        "total_per_oz": round(float(per_oz.sum()), 3),
        "win_rate": round(float((b["total_profit"] > 0).mean()), 3),
        "avg_hold_min": round(float(b["duration_min"].mean()), 1),
        "dirs": ",".join(d[0] for d in b["direction"].tolist()),  # b/s
    }


def three_way(eids: list[str]) -> pd.DataFrame:
    cfg = BacktestConfig()
    feat = build_features(cfg)
    baskets = pd.read_parquet(ep.BASKETS_PARQUET)
    rows = []
    for eid in eids:
        a = _agent_row(eid)
        if a is None:
            print(f"  (skip {eid}: no agent replay result yet)")
            continue
        for r in (a, _engine_row(eid, cfg, feat), _live_row(eid, baskets, cfg)):
            rows.append({"episode_id": eid, **r})
    return pd.DataFrame(rows)


def render_md(df: pd.DataFrame) -> str:
    lines = ["# Three-way comparison — Claude agent vs deterministic rule vs live EA", ""]
    lines.append("Metric is **$/oz** (account-size-independent). `dirs`: agent l/s, engine & live b/s.")
    lines.append("")
    cols = ["source", "n_baskets", "total_per_oz", "win_rate", "avg_hold_min", "dirs"]
    for eid, g in df.groupby("episode_id"):
        lines.append(f"## {eid}")
        lines.append("")
        head = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        body = ["| " + " | ".join("" if pd.isna(r[c]) else str(r[c]) for c in cols) + " |"
                for _, r in g.iterrows()]
        lines += [head, sep, *body, ""]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Three-way episode comparison.")
    p.add_argument("--episodes", default=None, help="comma-separated episode ids")
    p.add_argument("--all-replayed", action="store_true", help="every episode with an agent result")
    args = p.parse_args()

    if args.all_replayed:
        eids = sorted({p.parent.name for p in REPORTS.glob("*/*.replay.json")})
    elif args.episodes:
        eids = [e.strip() for e in args.episodes.split(",") if e.strip()]
    else:
        p.error("pass --episodes <eid,...> or --all-replayed")

    df = three_way(eids)
    if df.empty:
        raise SystemExit("nothing to compare — run replays first")
    THREE_WAY_MD.parent.mkdir(parents=True, exist_ok=True)
    THREE_WAY_MD.write_text(render_md(df))
    print(df.to_string(index=False))
    print(f"\n-> {THREE_WAY_MD}")


if __name__ == "__main__":
    main()
