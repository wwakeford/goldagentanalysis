---
name: gold-replay
description: Spawn the bar-by-bar no-lookahead walk-forward replay agents for one or more gold (XAUUSD) trading sessions. Resolves which session the user means (a specific episode like 2025-06-12.evening, a random EA-active session, or the latest), builds the per-episode prompts, then fans out one no-lookahead gold-trader Agent per episode. Use when the user asks to "spawn gold agents", "replay a gold session", "walk forward gold bar by bar", or run the gold replay — for a certain, random, or latest episode.
---

# Gold walk-forward replay (no-lookahead trader)

Drives the decision-gated replay engine (`src/goldreplay/replay.py`) so each episode (a
`date.session` slice of gold) is walked forward 15 minutes at a time by an Agent that never
sees future bars. The agent opens a basket when a setup triggers, lets the deterministic grid
work (or manages it by hand), and closes it — committing each decision before the next bar is
revealed. P&L / R / MFE / MAE are computed server-side; an un-closed basket is auto-flattened at
the session end and flagged `forced`. The no-lookahead guarantee holds by construction.

## Step 0 — resolve the session + prep (one command)

Map what the user said to an `--episode` spec and run `prep`. It builds the per-episode prompt(s)
+ `_replay_manifest.json` and prints the resolved episode(s) with the live-EA ground truth.

| User says | spec |
|-----------|------|
| a specific session ("June 12 evening", "2025-06-12.evening") | `2025-06-12.evening` |
| "a random session" / "some gold day" / nothing specific | `random-active` |
| "the latest" / "most recent" | `latest` |

```
python -m goldreplay.replay prep --episode <spec>
```

`random-active` / `latest` pick from catalog episodes the **live EA actually traded** (so the
three-way comparison has a ground-truth column) that also have a clean 4h feature warmup. Run
`python -m goldreplay.episodes build` first if `episodes/catalog.csv` is missing.

## Step 1 — confirm before fan-out

Show the user the **resolved episode(s)**, the **M1 bar count**, and **whether/what the live EA
traded** from the `prep` output, then wait for go-ahead. Each agent self-drives ~24 `step` calls.

## Step 2 — fan out agents

On go-ahead: read `reports/<eid>/_replay_manifest.json`. Spawn one **general-purpose Agent per
row IN PARALLEL** (single message, multiple Agent tool calls). Pass the contents of each row's
`prompt_path` as the agent prompt **verbatim** — do not paraphrase. Each agent self-drives its own
`start` → `step` loop and writes its result to `result_path` (`reports/<eid>/<eid>.replay.json`).
- If an agent errors mid-loop it can resume: `replay status --episode <eid>` reprints the current
  bar, then continue `step`-ing. State lives in `reports/<eid>/<eid>.replay.state.json`.
- **Do NOT let agents read `data/price/*.parquet`** — the prompt forbids it; that reintroduces lookahead.

## Step 3 — roll up + compare

```
python -m goldreplay.replay summary --date <date>      # the episode's agent baskets vs EA P&L
python -m goldreplay.replay aggregate --all            # flat trade table
python -m goldreplay.replay report                     # replay_expectancy.md (no-lookahead)
python -m goldreplay.compare --episodes <eid,...>      # AGENT vs ENGINE vs LIVE_EA ($/oz)
```

Lead with the **three-way comparison**: did the no-lookahead agent match the direction and the
sign of the deterministic rule and the real EA on the same session? Then relay which setups armed
and any `propose:*` patterns.

## Notes
- Each `<eid>.replay.json` holds `observations`, the managed-trade `baskets` (with server-computed
  `pnl_usd`/`pnl_per_oz`/`r_multiple`/`mfe_per_oz`/`mae_per_oz`/`hold_min`/`vs_1oz_tp`), and the
  `ea_ground_truth` for the session.
- For hindsight discovery instead of prospective trading, use the `research generate` fan-out.
- This is the bias-free way to have an LLM trade gold: unlike the hindsight `research` prompt, the
  agent cannot peek ahead.
