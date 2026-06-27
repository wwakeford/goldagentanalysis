# goldreplay/live — Quantum Queen, deployed

goldreplay *researches* the Quantum Queen recovery grid bias-free — one Claude agent
per day, walking gold forward bar-by-bar with no look-ahead (see the top-level
[README](../README.md)). **This directory is the other half of that story: where the
*same* validated strategy runs against a live MT5 broker.**

Run the **Quantum Queen recovery grid** (higher-return, both-directional, validated on
2yr of data) live for **Quanthack** (70% return / 15% DD / 10% Sharpe / 5% risk
discipline; **forced liquidation = instant DQ**), on a parity-tested execution chassis.

## What this is

`live/` swaps a decision core (the no-lookahead replay trader → the deterministic
Quantum recovery grid) behind a Broker / runner / margin-guard / kill-switch / reconcile
chassis. The Quantum decision + feature math are **imported** from the canonical
`quantumstrategy` repo (not copied) so they cannot drift from the backtest.

- `quantum/quantum_features.py` — incremental live features (causal ATR). **Gate 1.**
- `quantum/quantum_engine.py` — per-bar decision state machine (mirror of backtest engine).
- `quantum/live_quantum.py` — `QuantumRunner` (bar-cadence) + replay parity. **Gate 2.**
- `quantum/gate3_margin.py` — margin / no-force-liquidation validation. **Gate 3.**
- `quantum/backfill.py` — M1 history warm-up (broker `copy_rates`, Dukascopy fallback).
- `run_live_quantum.py` — live entrypoint (env-wired, backfill, `--check`).
- `broker.py` / `sizing.py` — execution chassis (`SimBroker` any-OS, `MT5Broker` Windows;
  margin-guard lot sizing with `MARGIN_CALL_LEVEL = 0.30`, the comp's DQ line).

## Validate before the box (all three must be green)
```
cd live
python -m quantum.quantum_features      # Gate 1: features == causal oracle (~1e-12)
python -m quantum.live_quantum          # Gate 2: replay == causal backtest (~1e-9)
python -m quantum.gate3_margin 100      # Gate 3: DQ-avoided all slices; ride-safe @ risk
python run_live_quantum.py --check      # sizing + warm-up preview (no connect)
```
Reference status: **Gate 1 PASS, Gate 2 PASS, Gate 3 PASS** (operating risk **3.0%** +
**regime gate**: min margin level ~6.3 vs the 0.30 DQ line = 21× clear, no bailout;
full-sample +91% causal, no DQ). The **regime gate** (don't open a basket against the
confirmed HTF MA200 trend) cut the downtrend-trapped-long tail: full-sample win
83.7%→91.5%, flip-losses 69→25.

## Provision the Windows box
London Windows VPS, MT5 terminal, Python 3.11/3.12 x64, "Allow algorithmic trading".
Copy **two** repos to the box:
- this `goldreplay` repo, and
- the `quantumstrategy` repo (needs `backtest/` + `scripts/`), and set `QUANTUM_REPO`
  to its path. (`data/price/` is only needed for the Dukascopy parity oracle, not live.)

```powershell
py -m venv .venv
.venv\Scripts\pip install -r live\requirements-live.txt
$env:QUANTUM_REPO = "C:\path\to\quantumstrategy"
```

## Configure + go live (PowerShell)
```powershell
$env:MT5_LOGIN="..."; $env:MT5_PASSWORD="..."; $env:MT5_SERVER="Broker-Server"
$env:SYMBOL="XAUUSD"            # confirm the exact comp symbol string
$env:LEVERAGE="100"            # account margin leverage — CONFIRM with the comp
$env:RISK_FRAC="0.030"          # return dial (shipped); no-DQ frontier to ~0.048 (Gate 3)
$env:REGIME_GATE_MARGIN="20"    # block baskets opposing HTF trend by >$20/oz (0=symmetric)
cd live
..\.venv\Scripts\python run_live_quantum.py --check   # preview sizing + warm-up
..\.venv\Scripts\python run_live_quantum.py           # live
```
Startup **backfills ~2400 M1 bars** (broker `copy_rates`, Dukascopy fallback) so the htf
trend-flip tail-protector is **armed from the first live bar** (~1.7 trading days of
history). Confirm the banner shows `flip ARMED`. A `KILL` file in `data/live_logs/`
flattens + halts at any time; restart re-adopts the open basket via `reconcile()` (no
double-trade).

## Risk dial / monitoring
- `RISK_FRAC` is the main return/risk knob. **0.030 shipped** (min margin ~6–12 on the
  worst regime); 0.045 still no-DQ (min ~3, more DD); ≥0.06 risks the voluntary bailout.
  Dial DOWN to 0.018 (min margin ~15–19) for a more conservative DD profile.
  `REGIME_GATE_MARGIN` (default 20): higher = looser gate (more counter-trend baskets);
  0 = symmetric (no gate).
- Watch `data/live_logs/events.jsonl`: `size` (lot + binding), `open`/`add`/`close`
  (with `reason`: tp/flip), `hard_floor`/`halt` (the voluntary bailout).
- **Confirm with organisers:** exact account leverage, the symbol string, and any
  order-rate limits (Quantum fires ≤1 decision/min — friendly, but confirm).

## Known live-vs-backtest gap (honest)
The backtest's ATR was mildly lookahead; the live engine uses a **causal** ATR.
Re-validated: the edge is unchanged (full-sample +45.1% causal vs +43.0% lookahead,
same Sharpe/RD, no DQ). No action needed — flagged for transparency.
