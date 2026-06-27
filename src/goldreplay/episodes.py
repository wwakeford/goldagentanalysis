"""Episode catalog — the single-instrument analog of intradayRS's watchlist.

An *episode* is a (broker date, session window) slice of gold, e.g.
`2025-06-12.evening`. Windows are contiguous broker-hour (UTC+3) ranges around
the EA's three trading sessions. The catalog lists every episode that the live
EA traded (so the three-way comparison always has a ground-truth column),
records the UTC bounds + bar counts + a clean-warmup flag, and supports
selecting one by id, at random, at random among active episodes, or the latest.
"""
from __future__ import annotations

import argparse
import random
from datetime import date, datetime, timedelta

import pandas as pd

from .align import broker_to_utc, load_m1
from .paths import BASKETS_PARQUET, CATALOG_CSV, EPISODES_DIR

# Contiguous broker-hour (UTC+3) bounds per window: [start_hour, end_hour).
# end_hour may be 24 (rolls to next broker day). Padded around the EA's exact
# active hours so the agent sees the lead-in and the session resolve.
WINDOW_BROKER_BOUNDS = {
    "evening": (18, 24),    # broker 18:00-24:00  (UTC 15-21)
    "midnight": (2, 5),     # broker 02:00-05:00  (UTC 23-02) around the Asian open
    "morning": (8, 12),     # broker 08:00-12:00  (UTC 05-09)
}
WARMUP_BARS = 240           # 4h of M1 needed before the window for valid features


def episode_id(d: date, window: str) -> str:
    return f"{d.isoformat()}.{window}"


def parse_episode_id(eid: str) -> tuple[date, str]:
    ds, window = eid.rsplit(".", 1)
    if window not in WINDOW_BROKER_BOUNDS:
        raise SystemExit(f"unknown window {window!r}; valid: {list(WINDOW_BROKER_BOUNDS)}")
    return date.fromisoformat(ds), window


def episode_bounds_utc(d: date, window: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """[start, end) of the episode in naive UTC (broker bounds shifted -3h)."""
    start_h, end_h = WINDOW_BROKER_BOUNDS[window]
    broker_start = datetime(d.year, d.month, d.day) + timedelta(hours=start_h)
    broker_end = datetime(d.year, d.month, d.day) + timedelta(hours=end_h)
    return broker_to_utc(pd.Timestamp(broker_start)), broker_to_utc(pd.Timestamp(broker_end))


def _window_for_broker_hour(bh: int) -> str | None:
    for name, (s, e) in WINDOW_BROKER_BOUNDS.items():
        if s <= bh < e:
            return name
    return None


def build_catalog(m: pd.DataFrame | None = None,
                  baskets: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (broker date, window) the live EA traded, with UTC bounds,
    M1 bar counts, a clean-warmup flag, and the aligned EA ground truth."""
    if m is None:
        m = load_m1()
    if baskets is None:
        baskets = pd.read_parquet(BASKETS_PARQUET)

    b = baskets.copy()
    b["broker_date"] = b["first_open"].dt.date
    b["broker_hour"] = b["first_open"].dt.hour
    b["window"] = b["broker_hour"].map(_window_for_broker_hour)
    b = b[b["window"].notna()]

    ts = m["timestamp"]
    m_min, m_max = ts.min(), ts.max()

    rows = []
    for (d, window), g in b.groupby(["broker_date", "window"]):
        start_utc, end_utc = episode_bounds_utc(d, window)
        if start_utc < m_min or end_utc > m_max:
            continue
        in_win = (ts >= start_utc) & (ts < end_utc)
        n_bars = int(in_win.sum())
        if n_bars == 0:
            continue
        warmup_ok = (start_utc - m_min) >= timedelta(hours=4) and \
            int(((ts >= start_utc - timedelta(hours=4)) & (ts < start_utc)).sum()) >= WARMUP_BARS
        rows.append({
            "episode_id": episode_id(d, window),
            "date": d.isoformat(),
            "window": window,
            "start_utc": start_utc.isoformat(),
            "end_utc": end_utc.isoformat(),
            "n_m1_bars": n_bars,
            "clean_warmup": bool(warmup_ok),
            "ea_basket_ids": ",".join(str(x) for x in g["basket_id"].tolist()),
            "ea_n_baskets": int(len(g)),
            "ea_total_profit": round(float(g["total_profit"].sum()), 2),
            "has_ea_activity": True,
        })
    cat = pd.DataFrame(rows).sort_values("episode_id").reset_index(drop=True)
    return cat


def load_catalog() -> pd.DataFrame:
    if not CATALOG_CSV.exists():
        raise SystemExit("no episodes/catalog.csv — run `python -m goldreplay.episodes build`")
    return pd.read_csv(CATALOG_CSV)


def align_ea_baskets(eid: str, baskets: pd.DataFrame | None = None) -> pd.DataFrame:
    """The real EA baskets whose first_open falls inside this episode's window."""
    if baskets is None:
        baskets = pd.read_parquet(BASKETS_PARQUET)
    d, window = parse_episode_id(eid)
    start_utc, end_utc = episode_bounds_utc(d, window)
    fo_utc = broker_to_utc(baskets["first_open"])
    m = (fo_utc >= start_utc) & (fo_utc < end_utc)
    return baskets.loc[m].reset_index(drop=True)


def select_episodes(spec: str) -> list[str]:
    """Resolve a selection spec to a list of episode ids.

    spec: explicit `<eid>` | `random` | `random-active` | `latest`.
    (Every catalog row is EA-active, so `random` and `random-active` coincide;
    both are offered for parity with intradayRS.)
    """
    if "." in spec and spec not in {"random", "random-active", "latest"}:
        parse_episode_id(spec)  # validate
        return [spec]
    cat = load_catalog()
    cat = cat[cat["clean_warmup"]]
    if cat.empty:
        raise SystemExit("no episodes with a clean 4h warmup in the catalog")
    if spec in {"random", "random-active"}:
        return [random.choice(cat["episode_id"].tolist())]
    if spec == "latest":
        return [cat.sort_values("episode_id")["episode_id"].iloc[-1]]
    raise SystemExit(f"unknown episode spec {spec!r}")


def main() -> None:
    p = argparse.ArgumentParser(description="Episode catalog for gold replay.")
    p.add_argument("cmd", choices=["build", "list"])
    p.add_argument("--active", action="store_true", help="(list) only EA-active episodes")
    p.add_argument("--window", default=None, help="(list) filter to one window")
    args = p.parse_args()

    if args.cmd == "build":
        cat = build_catalog()
        EPISODES_DIR.mkdir(parents=True, exist_ok=True)
        cat.to_csv(CATALOG_CSV, index=False)
        print(f"wrote {len(cat)} episodes -> {CATALOG_CSV}")
        print(cat.groupby("window")["episode_id"].count().to_string())
        print(f"clean-warmup episodes: {int(cat['clean_warmup'].sum())}")
        return

    cat = load_catalog()
    if args.window:
        cat = cat[cat["window"] == args.window]
    cols = ["episode_id", "n_m1_bars", "clean_warmup", "ea_n_baskets", "ea_total_profit"]
    print(cat[cols].to_string(index=False))


if __name__ == "__main__":
    main()
