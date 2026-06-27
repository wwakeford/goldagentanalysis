# goldreplay

**Spawn one Claude agent per trading day. Hand each a strategy idea and a shared tag vocabulary. Let it walk the day forward like a real trader — with no look-ahead bias — noting and tagging what it sees, so the knowledge accumulates and feeds the next wave of agents.**

---

## What this is

**The idea.** We start with a hunch about how gold trades — a strategy idea (e.g. *"in an active
session, a strong 4-hour trend that breaks out tends to continue"*). Instead of hard-coding that
hunch into a rigid backtest, we hand it to **Claude agents and let them trade it like real
traders** — under lab conditions where they cannot cheat.

**One agent per day.** We slice the gold history into trading days (each day's active session) and
**spawn one Claude agent per day**, all in parallel. Each agent is handed the same brief:

- **What we're looking for** — the strategy idea in plain language (the session + trend + breakout
  setup to hunt).
- **A set of tags** — a shared vocabulary of named patterns (`evening_trend_breakout`,
  `grid_recovery`, `counter_trend_trap`, …) so every agent describes what it sees in the *same
  words*. Agents may coin a new tag when they spot something not on the list.

**It walks the day forward — no look-ahead.** The agent doesn't get the whole day at once. It
receives the session **one 15-minute bar at a time**, and to see the next bar it must first commit
a decision for the current one. It is *physically prevented from seeing the future* — exactly the
constraint a real trader lives under, and the one almost every AI backtest secretly violates.

**It behaves like a real trader.** As it walks the day it **notes observations** ("4h trend turned
down, breaking the session low"), **tags** each to the shared vocabulary, **opens a trade** when
its setup actually triggers, manages it, and **closes** it — committing every decision live, before
the next bar exists.

**What comes out.** Each agent leaves a structured, tagged record of its day plus its trades scored
on real (no-peeking) profit. Because every agent used the same tags, all those days become
**comparable, searchable knowledge** — and a later `distill` agent reads the whole pile to learn
which tagged setups genuinely paid off, sharpening the brief for the next wave of agents. The
day-by-day results are also benchmarked against the *real* bot that traded those days, to prove the
simulation is faithful.

**Why it matters.** It turns a vague trading hunch into a fleet of disciplined, honest analysts
that each live one day blind, write down what they saw in a common language, and hand that
knowledge forward — so the system *accumulates* verified insight instead of overfitting to the past.

This is a port of a no-lookahead walk-forward agent harness onto the reverse-engineered
"Quantum Queen" XAUUSD grid strategy (873 real baskets as ground truth).

---

## How the no-look-ahead guarantee is enforced

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
live/             live MT5 deployment of the SAME strategy (run_live_quantum.py + quantum/)
```

## Going live

The replay harness is the *research* side — prove the edge bias-free. The deploy side
lives in [`live/`](live/README.md): the **same** Quantum Queen grid wired to a live MT5
broker via a parity-tested Broker / margin-guard / kill-switch / reconcile chassis, with
the decision + feature math **imported** from the canonical `quantumstrategy` repo so it
cannot drift from what was validated here. Three gates (feature parity, replay parity,
margin / no-DQ) must be green before go-live — see [`live/README.md`](live/README.md).

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
