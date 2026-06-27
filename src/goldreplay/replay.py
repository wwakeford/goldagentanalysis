"""Decision-gated walk-forward GOLD trader (in-chat, no API).

A Claude subagent walks ONE gold session (episode) forward 15 minutes at a time
with NO knowledge of the future. It decides when to OPEN a basket (direction),
optionally how to manage it, and when to CLOSE — committing every decision for
the current bar before the next is revealed. The full session lives server-side
in a state file; the agent only ever receives one decision bar at a time and
*cannot advance* until it commits. That gates lookahead by construction.

In the default `grid` manage mode the agent ARMS the basket and a deterministic
grid module (the vendored Basket state machine) handles the $1.90-step adds and
the ~$1/oz aggregate take-profit — exactly how the live EA works — and reports
back what it did on the next bar. In `manual` mode the agent controls adds/close
itself. Realized P&L, R-multiple and hold-window MFE/MAE are computed SERVER-SIDE
at close and never fed back into a pending decision.

CLI:
  prep      --episode <spec>             resolve + build prompt + manifest + confirm
  manifest  --episode <spec>             build prompt(s) + _replay_manifest.json
  start     --episode <eid>              init state, print warmup + first bar
  step      --episode <eid> --decision '<json>'
  status    --episode <eid>              reprint current bar without advancing
  summary   --date <YYYY-MM-DD>          roll up a date's episodes
  aggregate [--all | --episode <eid>]    flat trade table
  report                                 expectancy markdown
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import date

import pandas as pd

from . import episodes as ep
from .basket import Basket, resolve_target
from .config import BacktestConfig
from .enrich import decision_bars, m1_records, warmup_block
from .features import build_features
from .findings_store import check_slug, taxonomy_text, valid_slugs
from .paths import REPLAY_EXPECTANCY, REPLAY_TRADES, REPORTS

VALID_DIR = {"long": 1, "short": -1}

# ---------------- paths ----------------

def _state_path(eid: str):
    return REPORTS / eid / f"{eid}.replay.state.json"


def _out_path(eid: str):
    return REPORTS / eid / f"{eid}.replay.json"


def _prompt_path(eid: str):
    return REPORTS / eid / f"{eid}.replay.prompt.txt"


# ---------------- basket (de)serialization ----------------

def _basket_obj(bd: dict, cfg: BacktestConfig) -> Basket:
    b = Basket(direction=bd["direction"], open_time=pd.Timestamp(bd["open_time"]),
               cfg=cfg, g_per_lot=bd["g_per_lot"])
    b.legs = [tuple(leg) for leg in bd["legs"]]
    return b


# ---------------- P&L (server-side, post-hoc; never fed to a pending decision) ----------------

def _compute_outcome(bd: dict, m1: list[dict], exit_idx: int, exit_price: float,
                     reason: str, outcome: str | None, forced: bool, exit_time: str) -> dict:
    """Finalize a closed basket: realized $ / per-oz / R + hold-window MFE/MAE."""
    cfg = BacktestConfig()
    b = _basket_obj(bd, cfg)
    direction, vwap, lots = b.direction, b.vwap, b.total_lots
    pnl_usd = direction * (exit_price - vwap) * cfg.contract * lots
    per_oz = direction * (exit_price - vwap)

    ei = bd["open_m1_idx"]
    favs, advs = [], []
    for r in m1[ei:exit_idx + 1]:
        cs = r["bid"] if direction > 0 else r["ask"]   # close-side
        d = direction * (cs - vwap)
        favs.append(d)
        advs.append(d)
    mfe = max(favs) if favs else 0.0
    mae = min(advs) if advs else 0.0

    inv = bd.get("invalidation")
    risk_oz = abs(bd["first_fill"] - inv) if inv else cfg.max_adverse_usd
    r_mult = per_oz / risk_oz if risk_oz > 0 else None

    hold_min = (pd.Timestamp(m1[exit_idx]["ts"]) - pd.Timestamp(bd["open_time"])).total_seconds() / 60.0

    return {
        "direction": "long" if direction > 0 else "short",
        "manage": bd["manage"], "setup": bd["setup"], "thesis": bd["thesis"],
        "open_time": bd["open_bar_time"], "close_time": exit_time,
        "first_lot": bd["legs"][0][1], "legs": bd["legs"], "n_legs": len(bd["legs"]),
        "vwap": round(vwap, 3), "tp_per_oz": bd["tp_per_oz"], "invalidation": inv,
        "close_price": round(exit_price, 3), "close_reason": reason,
        "outcome": outcome, "forced": forced,
        "pnl_usd": round(pnl_usd, 2), "pnl_per_oz": round(per_oz, 3),
        "r_multiple": None if r_mult is None else round(r_mult, 2),
        "mfe_per_oz": round(mfe, 3), "mae_per_oz": round(mae, 3),
        "hold_min": round(hold_min, 1),
        "vs_1oz_tp": "hit" if per_oz >= 1.0 else "missed",
    }


# ---------------- grid advance (server-side forward walk between decision bars) ----------------

def _advance_grid(st: dict, cfg: BacktestConfig, to_m1_idx: int) -> list[dict]:
    """Walk the underlying M1 path from just after the current basket frontier up
    to `to_m1_idx` (inclusive), managing the grid in `grid` mode. Returns the list
    of grid events (adds / auto-close) to surface to the agent on the next bar.

    In `manual` mode the engine does nothing here — the agent manages adds/close.
    """
    bd = st.get("basket")
    if bd is None:
        return []
    events: list[dict] = []
    if bd["manage"] != "grid":
        bd["frontier_idx"] = to_m1_idx
        return events

    from .sizing import leg_lot
    from .align import utc_to_broker
    m1 = st["m1"]
    from_idx = bd.get("frontier_idx", bd["open_m1_idx"]) + 1
    b = _basket_obj(bd, cfg)

    for j in range(from_idx, to_m1_idx + 1):
        r = m1[j]
        bt = utc_to_broker(pd.Timestamp(r["ts"])).strftime("%H:%M")
        cs = r["bid"] if b.direction > 0 else r["ask"]        # close-side
        entry_side = r["ask"] if b.direction > 0 else r["bid"]  # add-side
        if b.should_close(cs):
            out = _compute_outcome({**bd, "legs": list(b.legs)}, m1, j, cs, "tp", "win", False, bt)
            st["baskets_closed"].append(out)
            st["basket"] = None
            events.append({"m1_ts": r["ts"], "type": "tp_close", "price": round(cs, 3),
                           "pnl_usd": out["pnl_usd"], "pnl_per_oz": out["pnl_per_oz"]})
            return events
        if b.should_stop(cs):
            out = _compute_outcome({**bd, "legs": list(b.legs)}, m1, j, cs, "stop", "loss", False, bt)
            st["baskets_closed"].append(out)
            st["basket"] = None
            events.append({"m1_ts": r["ts"], "type": "stop_close", "price": round(cs, 3),
                           "pnl_usd": out["pnl_usd"], "pnl_per_oz": out["pnl_per_oz"]})
            return events
        lot = leg_lot(cfg, entry_side)
        if b.should_add(cs, entry_side, lot):
            b.add_leg(entry_side, lot)
            events.append({"m1_ts": r["ts"], "type": "grid_add", "price": round(entry_side, 3),
                           "n_legs": b.n_legs, "vwap": round(b.vwap, 3)})

    # basket still open: persist updated legs + frontier
    bd["legs"] = [list(leg) for leg in b.legs]
    bd["frontier_idx"] = to_m1_idx
    return events


# ---------------- apply one decision ----------------

def _apply_decision(st: dict, dec: dict, cfg: BacktestConfig) -> None:
    from .sizing import leg_lot
    bar = st["bars"][st["ptr"]]
    slugs = valid_slugs()

    obs = dec.get("observation")
    if obs:
        tag = dec.get("tag")
        if tag is not None:
            check_slug(tag, slugs, "tag")
        st["observations"].append({"time": bar["time"], "note": obs, "tag": tag})

    op = dec.get("open")
    if op:
        if st.get("basket") is not None:
            raise SystemExit("open given but a basket is already open — close it first")
        side = op.get("direction")
        if side not in VALID_DIR:
            raise SystemExit(f"open.direction must be one of {sorted(VALID_DIR)}")
        direction = VALID_DIR[side]
        setup = op.get("setup")
        if not setup:
            raise SystemExit("open requires a 'setup' slug")
        check_slug(setup, slugs, "open setup")
        fill = bar["ask"] if direction > 0 else bar["bid"]   # pay the spread
        lot_spec = op.get("lot", "auto")
        lot = leg_lot(cfg, fill) if lot_spec in (None, "auto") else float(lot_spec)
        tp_per_oz = op.get("tp_per_oz")
        g = tp_per_oz * cfg.contract if tp_per_oz else resolve_target(cfg, direction, bar.get("atr_1h") or float("nan"))
        st["basket"] = {
            "direction": direction, "open_time": bar["ts_utc"], "open_bar_time": bar["time"],
            "open_m1_idx": bar["m1_idx"], "frontier_idx": bar["m1_idx"],
            "g_per_lot": g, "tp_per_oz": tp_per_oz if tp_per_oz else round(g / cfg.contract, 3),
            "legs": [[round(fill, 3), lot]], "first_fill": round(fill, 3),
            "manage": op.get("manage", "grid"), "setup": setup,
            "thesis": op.get("thesis", ""), "invalidation": op.get("invalidation"),
        }

    ga = dec.get("grid_add")
    if ga:
        bd = st.get("basket")
        if bd is None:
            raise SystemExit("grid_add given but no basket is open")
        if bd["manage"] != "manual":
            raise SystemExit("grid_add only honored when the basket's manage=='manual'")
        direction = bd["direction"]
        price = bar["ask"] if direction > 0 else bar["bid"]
        lot = leg_lot(cfg, price) if ga.get("lot", "auto") in (None, "auto") else float(ga["lot"])
        bd["legs"].append([round(price, 3), lot])

    cl = dec.get("close")
    if cl:
        bd = st.get("basket")
        if bd is None:
            raise SystemExit("close given but there is no open basket")
        direction = bd["direction"]
        exit_price = bar["bid"] if direction > 0 else bar["ask"]   # close-side
        out = _compute_outcome(bd, st["m1"], bar["m1_idx"], exit_price,
                               cl.get("reason", "discretionary"), cl.get("outcome"),
                               False, bar["time"])
        out["close_note"] = cl.get("note", "")
        st["baskets_closed"].append(out)
        st["basket"] = None

    pm = dec.get("postmortem")
    if pm:
        text = pm.get("text") or pm.get("note")
        if not text:
            raise SystemExit("postmortem requires 'text'")
        if not st["baskets_closed"]:
            raise SystemExit("postmortem given but no basket has closed")
        st["baskets_closed"][-1]["postmortem"] = text


# ---------------- emit / finalize ----------------

def _basket_status_line(st: dict, bar: dict) -> str:
    bd = st.get("basket")
    if bd is None:
        return "position: FLAT"
    cfg = BacktestConfig()
    b = _basket_obj(bd, cfg)
    cs = bar["bid"] if b.direction > 0 else bar["ask"]
    per_oz = b.direction * (cs - b.vwap)
    return (f"position: {'LONG' if b.direction > 0 else 'SHORT'} {b.n_legs} leg(s) "
            f"vwap={b.vwap:.2f} floating={per_oz:+.2f}$/oz (${b.floating_usd(cs):+.0f}) "
            f"tp={bd['tp_per_oz']}$/oz manage={bd['manage']} setup={bd['setup']}")


def _bar_line(bar: dict) -> str:
    return (f"BAR {bar['time']} (i={bar['i']}) | mid={bar['mid']} bid={bar['bid']} ask={bar['ask']} "
            f"| 15m o={bar['o15']} h={bar['h15']} l={bar['l15']} c={bar['c15']} "
            f"| mom_4h={bar['mom_4h']} rng_pos={bar['rng_pos']} rsi={bar['rsi']} atr_1h={bar['atr_1h']} "
            f"| window={bar['window']} floor={bar['min_trend_floor']} brk={bar['breakout_frac']} "
            f"| cum_move={bar['cum_move']} dist_hi60={bar['dist_hi60']} dist_lo60={bar['dist_lo60']}")


def _emit(st: dict, grid_events: list[dict] | None = None) -> None:
    bars, ptr = st["bars"], st["ptr"]
    if ptr >= len(bars):
        return
    bar = bars[ptr]
    if grid_events:
        for e in grid_events:
            print(f"GRID {e['type']} @ {e['m1_ts']} price={e.get('price')} "
                  + (f"n_legs={e['n_legs']} vwap={e['vwap']}" if e["type"] == "grid_add"
                     else f"pnl=${e.get('pnl_usd')} ({e.get('pnl_per_oz')}$/oz)"))
    print(_bar_line(bar))
    print(_basket_status_line(st, bar))
    print(f"Awaiting decision for this bar. Run: replay step --episode {st['episode_id']} --decision '<json>'")


def _finalize(st: dict) -> None:
    # auto-flatten any open basket at the episode's last M1 close (forced)
    bd = st.get("basket")
    if bd is not None:
        m1 = st["m1"]
        last = m1[-1]
        direction = bd["direction"]
        cs = last["bid"] if direction > 0 else last["ask"]
        out = _compute_outcome(bd, m1, len(m1) - 1, cs, "eod_flat", None, True, st["bars"][-1]["time"])
        out["close_note"] = "auto-flattened at episode close (no close committed)"
        st["baskets_closed"].append(out)
        st["basket"] = None

    baskets = st["baskets_closed"]
    out = {
        "episode_id": st["episode_id"], "date": st["date"], "window": st["window"],
        "start_utc": st["start_utc"], "end_utc": st["end_utc"],
        "ea_ground_truth": st["ea_ground_truth"],
        "observations": st["observations"],
        "baskets": baskets,
        "n_observations": len(st["observations"]),
        "n_baskets": len(baskets),
        "setups_triggered": sorted({b["setup"] for b in baskets}),
        "total_pnl_usd": round(sum(b["pnl_usd"] for b in baskets), 2),
    }
    _out_path(st["episode_id"]).write_text(json.dumps(out, indent=2))


# ---------------- commands ----------------

def _build_feats():
    return build_features(BacktestConfig())


def start(eid: str) -> None:
    cfg = BacktestConfig()
    feat = _build_feats()
    bars = decision_bars(feat, eid, cfg)
    if not bars:
        raise SystemExit(f"no decision bars for {eid} — check the catalog")
    m1 = m1_records(feat, eid)
    d, window = ep.parse_episode_id(eid)
    start_utc, end_utc = ep.episode_bounds_utc(d, window)
    ea = ep.align_ea_baskets(eid)
    st = {
        "episode_id": eid, "date": d.isoformat(), "window": window,
        "start_utc": start_utc.isoformat(), "end_utc": end_utc.isoformat(),
        "bars": bars, "m1": m1, "ptr": 0, "basket": None,
        "observations": [], "baskets_closed": [],
        "ea_ground_truth": {
            "n_baskets": int(len(ea)),
            "directions": ea["direction"].tolist() if not ea.empty else [],
            "total_profit_usd": round(float(ea["total_profit"].sum()), 2) if not ea.empty else 0.0,
            "basket_ids": ea["basket_id"].tolist() if not ea.empty else [],
        },
    }
    sp = _state_path(eid)
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(st))
    print(f"REPLAY {eid} | {len(bars)} decision bars ({bars[0]['time']}-{bars[-1]['time']} broker) "
          f"| live EA traded {st['ea_ground_truth']['n_baskets']} basket(s) here")
    print()
    print(warmup_block(feat, eid, cfg))
    print()
    _emit(st)


def step(eid: str, decision: str) -> None:
    cfg = BacktestConfig()
    sp = _state_path(eid)
    if not sp.exists():
        raise SystemExit(f"no replay state — run `replay start --episode {eid}` first")
    st = json.loads(sp.read_text())
    if st["ptr"] >= len(st["bars"]):
        print("DONE (already finalized)")
        return
    try:
        dec = json.loads(decision)
    except json.JSONDecodeError as e:
        raise SystemExit(f"--decision is not valid JSON: {e}")

    _apply_decision(st, dec, cfg)

    # advance the grid forward to the NEXT decision bar's as-of M1 index
    ptr = st["ptr"]
    next_ptr = ptr + 1
    grid_events: list[dict] = []
    if next_ptr < len(st["bars"]):
        to_idx = st["bars"][next_ptr]["m1_idx"]
        grid_events = _advance_grid(st, cfg, to_idx)

    st["ptr"] = next_ptr
    if st["ptr"] >= len(st["bars"]):
        _finalize(st)
        sp.write_text(json.dumps(st))
        n = len(st["baskets_closed"])
        print(f"DONE {eid}: {len(st['observations'])} observations, {n} basket(s), "
              f"total ${st['baskets_closed'] and round(sum(b['pnl_usd'] for b in st['baskets_closed']), 2) or 0.0} "
              f"-> {_out_path(eid)}")
        return

    sp.write_text(json.dumps(st))
    _emit(st, grid_events)


def status(eid: str) -> None:
    sp = _state_path(eid)
    if not sp.exists():
        raise SystemExit("no replay state")
    st = json.loads(sp.read_text())
    if st["ptr"] >= len(st["bars"]):
        print("DONE (finalized)")
        return
    _emit(st)


# ---------------- prompt / manifest ----------------

PROMPT_TEMPLATE = """You are a gold (XAUUSD) trader walking ONE session forward 15 minutes at a time with NO knowledge of the future. You see one decision bar at a time and must commit for it before the next is revealed. Each bar you may: (a) log a notable observation, (b) OPEN a basket when a setup triggers, (c) CLOSE the open basket. Tag setups to the taxonomy below.

Working directory: {root}
Episode: {eid}   (date {date}, {window} session, broker UTC+3)
Session entry reference (the live EA's rule — a REFERENCE, not a mandate; you may deviate):
{strategy_card}

{warmup}

## Loop (run from the working directory)
1. `python -m goldreplay.replay start --episode {eid}` — prints the pre-episode warmup and the FIRST decision bar.
2. `python -m goldreplay.replay step --episode {eid} --decision '<json>'` — records your decision for the CURRENT bar, advances the grid, prints the NEXT bar (and any grid events that fired).
3. Repeat until a line starting with `DONE` prints, then stop.

## Decision JSON (single-quoted; any field may be null/omitted per bar)
{{"observation":"<notable price-action moment, or null>","tag":"<setup slug | propose:<snake_slug> | null>",
 "open":{{"direction":"long|short","lot":"auto","tp_per_oz":1.0,"manage":"grid","setup":"<slug>","thesis":"<one line: the session+4h-trend+breakout read>","invalidation":<price that kills the thesis>}},
 "close":{{"reason":"tp|trend_flip|invalidation|time|discretionary|eod_flat","outcome":"win|loss|scratch","note":"<one line>"}},
 "postmortem":{{"text":"<how it ultimately resolved once later bars revealed it>"}}}}

A quiet bar: pass {{"observation": null}} to advance. In `grid` manage mode (recommended) you only OPEN and CLOSE — the deterministic grid adds legs on ~$1.90 adverse moves and closes the whole basket at the ~$1/oz aggregate take-profit on its own, and tells you on the next bar. Use manage:"manual" only for full hand control (then you may also send grid_add).

## Read the tape (a lens, not a checklist; fields per bar: mom_4h, rng_pos, rsi, atr_1h, window, cum_move, dist_hi60/lo60)
1. Session + trend FIRST. Is `window` a real session (evening/midnight/morning, not "other")? Is the 4h trend strong — |mom_4h| at/above the bar's `min_trend_floor`? A weak/flat 4h trend is not a trade.
2. Breakout, not chase. The EA enters at the trend-side extreme of the 60-min range: long wants `rng_pos` >= `breakout_frac` (~0.80); short wants `rng_pos` <= 1-breakout_frac (~0.20). Distance to hi60/lo60 shows the room.
3. Direction = sign of the 4h trend. Long only when mom_4h>0, short only when mom_4h<0 — never fight the 4h trend.
4. The trap (avoid). A basket armed AGAINST the higher-timeframe trend that price never reverts to is the tail risk (`counter_trend_trap`). When in doubt, don't arm it.
5. Room vs vol. The aggregate TP is ~$1/oz; in high `atr_1h` regimes a single basket can need more room — that's why TP can scale with vol.

## Discipline
- Arm at most 1-2 baskets this session; quality over quantity. Most sessions are zero or one good arm.
- Let the grid work to TP — do not panic-close a basket that is adding legs and grinding back; that recovery IS the strategy. Close early only on a confirmed trend flip against you or a broken invalidation.
- Be flat by the session end; an un-closed basket is auto-flattened at the last bar and marked `forced`.
- P&L / R / MFE / MAE are computed server-side from your open/close — you never need to compute them.

## Setup taxonomy (a vocabulary for tagging, NOT a menu of permitted trades)
{taxonomy}

## Critical
- You may ONLY learn future prices via `step`. NEVER read data/price/*.parquet, the features cache, or any cached file — that is lookahead and invalidates the run.

Run the full loop now. When you see DONE, reply with a 4-6 line summary: how gold traded this session, which setup(s) you armed, and per basket — direction/entry, how it resolved (win/loss/scratch via TP or your close), whether you let the grid work, and whether a better decision existed. Your result JSON is written to {result}."""


def _strategy_card_text() -> str:
    from .paths import STRATEGY_CARD
    if STRATEGY_CARD.exists():
        # use the one-line summary block (first paragraph after the title)
        txt = STRATEGY_CARD.read_text()
        return txt
    return ("Session windows (broker UTC+3): evening 18-23, midnight ~3, morning 8-11. "
            "Enter in the 4h-trend direction when |mom_4h| exceeds the window floor and price "
            "breaks the trend-side 60-min extreme; grid-add ~$1.90 adverse; close the basket at "
            "~$1/oz aggregate take-profit.")


def manifest(spec: str) -> list[str]:
    cfg = BacktestConfig()
    eids = ep.select_episodes(spec)
    from .paths import ROOT
    rows = []
    for eid in eids:
        d, window = ep.parse_episode_id(eid)
        feat = _build_feats()
        wb = warmup_block(feat, eid, cfg)
        prompt = PROMPT_TEMPLATE.format(
            root=ROOT, eid=eid, date=d.isoformat(), window=window,
            strategy_card=_strategy_card_text(), warmup=wb,
            taxonomy=taxonomy_text(), result=_out_path(eid),
        )
        pp = _prompt_path(eid)
        pp.parent.mkdir(parents=True, exist_ok=True)
        pp.write_text(prompt)
        rows.append({"episode_id": eid, "prompt_path": str(pp), "result_path": str(_out_path(eid))})
    # one manifest per selection; keyed under the first episode's folder for single-episode runs
    man_dir = REPORTS / eids[0]
    man_dir.mkdir(parents=True, exist_ok=True)
    man = man_dir / "_replay_manifest.json"
    man.write_text(json.dumps(rows, indent=2))
    print(f"wrote {len(rows)} replay prompt(s); manifest: {man}")
    return eids


def prep(spec: str) -> None:
    eids = manifest(spec)
    cat = ep.load_catalog().set_index("episode_id")
    print("\nresolved episode(s):")
    for eid in eids:
        if eid in cat.index:
            r = cat.loc[eid]
            print(f"  {eid}: {r['n_m1_bars']} M1 bars | live EA {r['ea_n_baskets']} basket(s) "
                  f"(${r['ea_total_profit']})")
        else:
            print(f"  {eid}")
    print(f"\nNEXT: confirm with the user, then in Claude Code chat read the manifest and spawn one "
          f"general-purpose Agent per row IN PARALLEL, passing prompt_path verbatim.")
    print(f"Then: python -m goldreplay.replay summary --date {eids[0].split('.')[0]}")


# ---------------- summary / aggregate / report ----------------

def summary(d: str) -> None:
    """Roll up every replayed episode whose date == d."""
    rows = []
    per_setup: dict[str, dict] = defaultdict(lambda: {"n": 0, "long": 0, "short": 0, "pnl": 0.0})
    files = sorted(REPORTS.glob(f"{d}.*/{d}.*.replay.json"))
    if not files:
        raise SystemExit(f"no *.replay.json for {d} — run agents first")
    for f in files:
        doc = json.loads(f.read_text())
        baskets = doc.get("baskets", [])
        rows.append({
            "episode_id": doc["episode_id"], "window": doc["window"],
            "n_observations": doc.get("n_observations", 0), "n_baskets": len(baskets),
            "setups": doc.get("setups_triggered", []),
            "pnl_usd": doc.get("total_pnl_usd", 0.0),
            "ea_pnl_usd": doc.get("ea_ground_truth", {}).get("total_profit_usd"),
        })
        for b in baskets:
            agg = per_setup[b["setup"]]
            agg["n"] += 1
            agg["long" if b["direction"] == "long" else "short"] += 1
            agg["pnl"] = round(agg["pnl"] + b["pnl_usd"], 2)
    out = {"date": d, "episodes": rows, "per_setup": dict(per_setup),
           "total_pnl_usd": round(sum(r["pnl_usd"] for r in rows), 2)}
    (REPORTS / f"_replay_summary_{d}.json").write_text(json.dumps(out, indent=2))
    print(f"{len(rows)} episode(s) for {d}: total agent P&L ${out['total_pnl_usd']}")
    for r in rows:
        print(f"  {r['episode_id']}: {r['n_baskets']} basket(s) ${r['pnl_usd']} "
              f"(EA ${r['ea_pnl_usd']}) setups={r['setups']}")
    for slug, a in sorted(per_setup.items(), key=lambda kv: kv[1]['n'], reverse=True):
        print(f"  setup {slug}: {a['n']} ({a['long']}L/{a['short']}S) ${a['pnl']}")


TRADE_COLUMNS = [
    "episode_id", "date", "window", "setup", "direction", "manage", "first_lot", "n_legs",
    "open_time", "close_time", "close_reason", "outcome", "result", "forced",
    "pnl_usd", "pnl_per_oz", "r_multiple", "mfe_per_oz", "mae_per_oz", "hold_min", "vs_1oz_tp",
    "ea_n_baskets", "ea_total_profit",
]


def _classify(per_oz: float | None) -> str | None:
    if per_oz is None:
        return None
    if per_oz > 0.02:
        return "win"
    if per_oz < -0.02:
        return "loss"
    return "scratch"


def aggregate(eid: str | None, all_eps: bool) -> None:
    if all_eps:
        files = sorted(REPORTS.glob("*/*.replay.json"))
    else:
        files = [_out_path(eid)] if _out_path(eid).exists() else []
    if not files:
        raise SystemExit("no replay results found — run replays first")
    rows = []
    for f in files:
        doc = json.loads(f.read_text())
        ea = doc.get("ea_ground_truth", {})
        for b in doc.get("baskets", []):
            rows.append({
                "episode_id": doc["episode_id"], "date": doc["date"], "window": doc["window"],
                "setup": b["setup"], "direction": b["direction"], "manage": b["manage"],
                "first_lot": b["first_lot"], "n_legs": b["n_legs"],
                "open_time": b["open_time"], "close_time": b["close_time"],
                "close_reason": b["close_reason"], "outcome": b.get("outcome"),
                "result": _classify(b.get("pnl_per_oz")), "forced": b["forced"],
                "pnl_usd": b["pnl_usd"], "pnl_per_oz": b["pnl_per_oz"],
                "r_multiple": b.get("r_multiple"), "mfe_per_oz": b["mfe_per_oz"],
                "mae_per_oz": b["mae_per_oz"], "hold_min": b["hold_min"], "vs_1oz_tp": b["vs_1oz_tp"],
                "ea_n_baskets": ea.get("n_baskets"), "ea_total_profit": ea.get("total_profit_usd"),
            })
    df = pd.DataFrame(rows, columns=TRADE_COLUMNS)
    REPLAY_TRADES.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(REPLAY_TRADES.with_suffix(".parquet"), index=False)
    df.to_csv(REPLAY_TRADES.with_suffix(".csv"), index=False)
    print(f"aggregated {len(df)} agent baskets across {df['episode_id'].nunique()} episodes "
          f"-> {REPLAY_TRADES.with_suffix('.csv')}")


def report() -> None:
    pq = REPLAY_TRADES.with_suffix(".parquet")
    if not pq.exists():
        raise SystemExit(f"{pq} missing — run `aggregate --all` first")
    df = pd.read_parquet(pq)

    def block(group_cols):
        def agg(g):
            scored = g["result"].dropna()
            r = g["r_multiple"].dropna()
            return pd.Series({
                "n": len(g),
                "win%": round(100 * (scored == "win").sum() / len(scored), 1) if len(scored) else None,
                "avg_R": round(r.mean(), 2) if len(r) else None,
                "total_pnl$": round(g["pnl_usd"].sum(), 2),
                "avg_pnl/oz": round(g["pnl_per_oz"].mean(), 3),
                "avg_mae/oz": round(g["mae_per_oz"].mean(), 3),
                "forced%": round(100 * g["forced"].sum() / len(g), 1),
                "tp_hit%": round(100 * (g["vs_1oz_tp"] == "hit").sum() / len(g), 1),
            })
        return df.groupby(group_cols, dropna=False).apply(agg, include_groups=False).reset_index()

    def to_md(b):
        cols = list(b.columns)
        head = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        body = ["| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |"
                for r in b.itertuples(index=False)]
        return "\n".join([head, sep, *body])

    lines = ["# Replay expectancy (realized, no-lookahead)", ""]
    lines.append(f"{len(df)} agent baskets across {df['episode_id'].nunique()} episodes "
                 f"({df['date'].min()} → {df['date'].max()}).")
    lines.append("")
    for title, cols in (("By setup", ["setup"]), ("By window", ["window"]),
                        ("By direction", ["direction"]), ("By close reason", ["close_reason"])):
        lines += [f"## {title}", "", to_md(block(cols)), ""]
    REPLAY_EXPECTANCY.write_text("\n".join(lines))
    print(f"{len(df)} baskets -> {REPLAY_EXPECTANCY}")
    print(block(["setup"]).to_string(index=False))


# ---------------- cli ----------------

def main() -> None:
    p = argparse.ArgumentParser(description="Walk-forward no-lookahead gold replay.")
    p.add_argument("cmd", choices=["prep", "manifest", "start", "step", "status",
                                   "summary", "aggregate", "report"])
    p.add_argument("--episode", default=None, help="episode id or spec (random/random-active/latest)")
    p.add_argument("--date", default=None, help="YYYY-MM-DD (summary)")
    p.add_argument("--decision", default=None, help="decision JSON (step)")
    p.add_argument("--all", action="store_true", help="aggregate every replayed episode")
    args = p.parse_args()

    if args.cmd == "report":
        report(); return
    if args.cmd == "summary":
        if not args.date:
            p.error("summary needs --date")
        summary(args.date); return
    if args.cmd == "aggregate":
        if not args.all and not args.episode:
            p.error("aggregate needs --episode or --all")
        aggregate(args.episode, args.all); return
    if args.cmd in {"prep", "manifest"}:
        if not args.episode:
            p.error(f"{args.cmd} needs --episode")
        (prep if args.cmd == "prep" else manifest)(args.episode); return

    if not args.episode:
        p.error(f"{args.cmd} needs --episode")
    if args.cmd == "start":
        start(args.episode)
    elif args.cmd == "status":
        status(args.episode)
    else:
        if args.decision is None:
            p.error("step needs --decision")
        step(args.episode, args.decision)


if __name__ == "__main__":
    main()
