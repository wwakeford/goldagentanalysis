"""Hindsight discovery fan-out (the second subagent mode).

Where `replay` is PROSPECTIVE and no-lookahead, `research` is explicitly HINDSIGHT:
the agent sees the WHOLE session at once and labels 1-3 notable gold price-action
moments, tagging each to the shared taxonomy and giving a hypothetical entry/exit
plus an LLM estimate of the outcome. These findings are DISCOVERY, not evidence —
`distill` folds them into the lessons playbook but weights the realized,
no-lookahead replay expectancy ABOVE the hindsight estimates.

CLI:
  generate  --episode <spec>     build hindsight prompt(s) + _research_manifest.json
  aggregate --episode <spec>     fold result JSONs into the findings index
  distill                        print the lessons-update prompt (agent-driven)
  rollup                         rebuild findings/by_pattern/<tag>/ from the index
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

from . import episodes as ep
from .config import BacktestConfig
from .enrich import decision_bars
from .features import build_features
from .findings_store import taxonomy_text
from .paths import BY_PATTERN, BY_TICKER, FINDINGS, REPLAY_EXPECTANCY, REPORTS

RESEARCH_DIR = REPORTS / "research"

SYSTEM = """You are analyzing ONE gold (XAUUSD) trading session in HINDSIGHT. You see the full {window} session (broker UTC+3) below; this is by design — label what was notable. Spot 1-3 genuinely notable price-action moments and report them as JSON.

Stage 1 = hindsight pattern discovery. Reuse a taxonomy `slug` as the tag if one fits; coin a new `snake_case` tag (no `propose:` prefix needed here) only if novel.

Episode: {eid}   Live EA ground truth: {ea}

## Session 15-min bars (broker time; full session — hindsight is allowed in THIS mode)
{table}

## Setup taxonomy
{taxonomy}

Reply with JSON only — no prose, no fences. Schema:
{{"episode_id":"{eid}","window":"{window}","regime":"<one line: trend/vol/session character>",
 "findings":[
   {{"window_et":"HH:MM-HH:MM","tag":"<slug_or_new>","what":"<1-2 sentences>",
     "entry":{{"side":"long|short","trigger":"<rule prose>","price":0.0,"invalidation":0.0}},
     "exit":{{"rule":"<how exited>","price":0.0}},
     "llm_est_outcome_pct":0.0,"llm_est_r":0.0,"confidence":0.0}}
 ]}}"""


def _table(feat, eid: str, cfg: BacktestConfig) -> str:
    bars = decision_bars(feat, eid, cfg)
    lines = ["time,mid,o15,h15,l15,c15,mom_4h,rng_pos,rsi,atr_1h,window,cum_move"]
    for b in bars:
        lines.append(f"{b['time']},{b['mid']},{b['o15']},{b['h15']},{b['l15']},{b['c15']},"
                     f"{b['mom_4h']},{b['rng_pos']},{b['rsi']},{b['atr_1h']},{b['window']},{b['cum_move']}")
    return "\n".join(lines)


def _res_path(eid: str):
    return RESEARCH_DIR / eid / f"{eid}.json"


def generate(spec: str) -> list[str]:
    cfg = BacktestConfig()
    feat = build_features(cfg)
    eids = ep.select_episodes(spec)
    baskets_by_eid = {}
    rows = []
    for eid in eids:
        d, window = ep.parse_episode_id(eid)
        ea = ep.align_ea_baskets(eid)
        ea_str = f"{len(ea)} basket(s), {ea['direction'].tolist() if not ea.empty else []}"
        prompt = SYSTEM.format(eid=eid, window=window, ea=ea_str,
                               table=_table(feat, eid, cfg), taxonomy=taxonomy_text())
        pdir = RESEARCH_DIR / eid
        pdir.mkdir(parents=True, exist_ok=True)
        pp = pdir / f"{eid}.prompt.txt"
        pp.write_text(prompt)
        rows.append({"episode_id": eid, "prompt_path": str(pp), "result_path": str(_res_path(eid))})
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    man = RESEARCH_DIR / "_research_manifest.json"
    man.write_text(json.dumps(rows, indent=2))
    print(f"wrote {len(rows)} research prompt(s); manifest: {man}")
    return eids


def aggregate(spec: str) -> None:
    eids = ep.select_episodes(spec) if spec else None
    files = []
    if eids:
        files = [_res_path(e) for e in eids if _res_path(e).exists()]
    else:
        files = sorted(RESEARCH_DIR.glob("*/*.json"))
    if not files:
        raise SystemExit("no research result JSONs found — run agents first")
    idx_path = BY_TICKER / "XAUUSD" / "index.json"
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    index = json.loads(idx_path.read_text()) if idx_path.exists() else []
    seen = {e["episode_id"] for e in index}
    added = 0
    for f in files:
        doc = json.loads(f.read_text())
        eid = doc.get("episode_id", f.parent.name)
        if eid in seen:
            continue
        findings = doc.get("findings", [])
        index.append({
            "episode_id": eid, "regime": doc.get("regime"),
            "tags": sorted({fd.get("tag") for fd in findings if fd.get("tag")}),
            "n_findings": len(findings),
            "headline": findings[0].get("what") if findings else None,
            "raw_path": str(f),
        })
        added += 1
    idx_path.write_text(json.dumps(index, indent=2))
    print(f"folded {added} new episode(s) into {idx_path} ({len(index)} total)")


def distill() -> None:
    """Print the lessons-update prompt: recent findings + REALIZED replay expectancy."""
    idx = BY_TICKER / "XAUUSD" / "index.json"
    findings = json.loads(idx.read_text()) if idx.exists() else []
    expectancy = REPLAY_EXPECTANCY.read_text() if REPLAY_EXPECTANCY.exists() else "(none yet)"
    print("DISTILL — feed this to a single Agent to update findings/lessons/index.json:\n")
    print("You maintain the gold setup taxonomy in findings/lessons/index.json. Promote recurring")
    print("patterns into lessons and revise confidence. CRITICAL: weight the REALIZED, no-lookahead")
    print("replay expectancy ABOVE the hindsight llm_est_* — cut the confidence of any setup whose")
    print("realized avg_R is negative even if hindsight looked good; raise it where the realized")
    print("avg_R is positive over a meaningful n.\n")
    print("=== REALIZED REPLAY EXPECTANCY (prospective, no-lookahead — strongest evidence) ===")
    print(expectancy)
    print(f"\n=== HINDSIGHT FINDINGS INDEX ({len(findings)} episodes) ===")
    print(json.dumps(findings[-20:], indent=2))


def rollup() -> None:
    idx = BY_TICKER / "XAUUSD" / "index.json"
    if not idx.exists():
        raise SystemExit("no findings index — run `research aggregate` first")
    index = json.loads(idx.read_text())
    by_tag: dict[str, list] = defaultdict(list)
    for e in index:
        for tag in e.get("tags", []):
            by_tag[tag].append({"episode_id": e["episode_id"], "headline": e.get("headline")})
    for tag, items in by_tag.items():
        d = BY_PATTERN / tag
        d.mkdir(parents=True, exist_ok=True)
        (d / "index.json").write_text(json.dumps(items, indent=2))
    print(f"rolled up {len(by_tag)} pattern(s) -> {BY_PATTERN}")


def main() -> None:
    p = argparse.ArgumentParser(description="Hindsight discovery fan-out.")
    p.add_argument("cmd", choices=["generate", "aggregate", "distill", "rollup"])
    p.add_argument("--episode", default=None)
    args = p.parse_args()
    if args.cmd == "generate":
        if not args.episode:
            p.error("generate needs --episode")
        generate(args.episode)
    elif args.cmd == "aggregate":
        aggregate(args.episode)
    elif args.cmd == "distill":
        distill()
    else:
        rollup()


if __name__ == "__main__":
    main()
