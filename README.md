# goldreplay

**Parallel Claude subagents trade gold (XAUUSD) with a provable no-look-forward bias — one bar at a time.**

A fleet of Claude Code subagents each walk *one* gold trading session forward in 15-minute
steps, deciding when to open a basket, how to steer it, and when to close — **committing every
decision before the next bar is revealed.** They cannot peek at the future, by construction. The
realized P&L is computed server-side and benchmarked against the deterministic strategy rule and
the *real* live EA that traded the same sessions.

This is a port of a no-lookahead walk-forward agent harness onto the reverse-engineered
"Quantum Queen" XAUUSD grid strategy (873 real baskets as ground truth).

---

## The no-lookahead architecture (the technology)

The agent never holds state. The full session lives server-side; the agent advances only by
committing a decision for the current bar:

```
        ┌──────────────────────── server-side state (full session, M1) ───────────────────────┐
        │   bars[0..N]   ptr ──┐                                                                │
        └──────────────────────┼─────────────────────────────────────────────────────────────┘
                               │ reveals ONLY bars[ptr]
                               ▼
                       ┌───────────────┐     commit decision      ┌──────────────────────────┐
                       │  Claude agent │ ───────(step)──────────▶ │ apply · advance grid on   │
                       │  (one bar)    │                          │ M1 · ptr += 1 · P&L server│
                       └───────────────┘ ◀──reveals next bar───── └──────────────────────────┘
```

Lookahead is gated **three independent ways**:

1. **State-machine reveal** — `replay step` records your decision for `bars[ptr]`, then prints
   `bars[ptr+1]`. You physically cannot advance without committing, and never receive a future bar.
2. **Prefix-only enrichment** — every field on decision bar *i* (`mom_4h`, `rng_pos`, `rsi`,
   `atr_1h`, cum move, distance to the 60-min hi/lo) is computed from M1 timestamps **≤ that bar's
   close**. The 4h/60-min/200-bar lookbacks reach into *pre-episode* bars (legitimately past data),
   never future ones.
3. **Prompt guard** — the prompt forbids reading `data/price/*.parquet` or the feature cache.

All three are checked by `tests/test_no_lookahead.py`, including a **prefix-determinism proof**:
recompute each shown feature from raw M1 truncated at that bar's timestamp and assert it is
identical — i.e. no future bar can change a value the agent saw.

---

## Two subagent modes (mirrored from the source harness)

- **`replay` — prospective, no-lookahead (the trader).** The agent *is* the trader: it walks the
  session and commits real open/close decisions. This is the bias-free evidence.
- **`research` — hindsight discovery.** The agent sees the whole session at once (by design) and
  labels notable patterns with hypothetical outcomes. This is *discovery*, not evidence.
- **`distill`** folds both into a lessons playbook, explicitly weighting the realized no-lookahead
  replay expectancy **above** the hindsight estimates.

---

## The killer three-way comparison

On the *same* sessions, line up three independent traders. Metric is **$/oz** (account-size
independent — absolute $ isn't comparable across a $1M engine, a ~$1k live account, and a
risk-sized agent):

| episode | source | n_baskets | total_per_oz | win_rate | avg_hold_min | dirs |
|---|---|---|---|---|---|---|
| 2024-06-07.evening | AGENT | 1 | 1.27 | 1.0 | 3.0 | s |
| 2024-06-07.evening | ENGINE | 6 | 8.65 | 1.0 | 22.2 | s,s,s,s,s,s |
| 2024-06-07.evening | LIVE_EA | 6 | 8.03 | 1.0 | 567.7 | s,s,s,s,s,s |

(See `reports/_examples/` for committed sample outputs so you can read results without running agents.
ENGINE ≈ LIVE_EA per-oz validates the reverse-engineered rule; the agent — with zero ability to peek —
read the same downtrend and shorted it too.)

---

## Quickstart

```bash
uv sync                                          # or: pip install -e .
python -m goldreplay.episodes build              # build the episode catalog (one-time)
python -m goldreplay.replay prep --episode random-active
# → then in Claude Code chat: "spawn the gold replay agents"  (uses .claude/skills/gold-replay)
```

Manual one-episode demo (what an agent does, scripted):

```bash
EID=2024-06-07.evening
python -m goldreplay.replay start --episode $EID
python -m goldreplay.replay step  --episode $EID --decision '{"open":{"direction":"short","lot":"auto","tp_per_oz":1.0,"manage":"grid","setup":"evening_trend_breakout","thesis":"4h down, breaking session low","invalidation":2330}}'
# ... step with {"observation":null} until DONE ...
python -m goldreplay.replay summary   --date 2024-06-07
python -m goldreplay.replay aggregate --all
python -m goldreplay.replay report
python -m goldreplay.compare --episodes $EID
python tests/test_no_lookahead.py                # prove the guarantee
```

## CLI

```
episodes build|list                              episode catalog (date.session slices)
replay   prep|manifest --episode <spec>          spec: <eid>|random|random-active|latest
replay   start|status  --episode <eid>
replay   step          --episode <eid> --decision '<json>'
replay   summary --date <YYYY-MM-DD> | aggregate --all | report
research generate|aggregate --episode <spec> | distill | rollup
compare  --episodes <eid,...> | --all-replayed
```

## Layout

```
src/goldreplay/   replay.py (★ no-lookahead engine)  enrich.py (prefix-only bars)
                  episodes.py  compare.py (★ three-way)  research.py  engine.py/engine_baseline.py
                  align.py/features.py/config.py/basket.py/sizing.py  (vendored EA math)
data/price/       XAUUSD_M1.parquet  (~790k M1 bars, 2024-04 → 2026-06)
data/ground_truth/ baskets.parquet (873 real EA baskets)  trades_all.parquet
findings/         lessons/  replay_trades.*  replay_expectancy.md  three_way_comparison.md
reports/          per-episode prompts + results;  _examples/ committed samples
tests/            test_no_lookahead.py
```

## Data provenance

Vendored from a separate reverse-engineering project: gold M1 from Dukascopy; broker server time
solved empirically as **UTC+3** (`align.py`); 873 real Quantum Queen baskets as the ground-truth
column. The deterministic rule (`engine.py`) and the agent see the *exact same* reverse-engineered
features, so neither can drift from the strategy being studied.

## Limitations (honest)

- The no-lookahead guarantee is **architectural**, not cryptographic — an agent is *told* not to
  read the parquet; the prefix-determinism test proves what it *sees* contains no future, and the
  state machine proves it can only advance by committing.
- Default `manage:"grid"` means the deterministic grid places legs + the $1/oz aggregate take-profit
  between the agent's decisions (exactly as the real EA works). The agent *arms, steers, and closes*
  — it doesn't place every leg by hand. `manage:"manual"` gives full hand control (hard mode).
- 15-minute decision cadence loses sub-bar fill timing; fills/MFE/MAE are still computed on the full
  M1 path server-side.
- Absolute $ is not comparable across the three traders — use the $/oz column.
- Single instrument (XAUUSD), sample window 2024-04 → 2026-06.
