"""The shipped, validated Quantum Queen config — the live-forward-tuned winner.

Fields from quantumstrategy/backtest BASELINE plus the 06-20 fresh-data re-opt
(sweep_week.py, data to 06-19): both-directional ride-to-recovery (no hard stop, no
time stop) + htf trend-flip exit with an ATR-floored MAE gate + vol-scaled take-profit
+ **entry_strictness 2.25** (the regime gate makes looser entries safe — they stay
trend-aligned; doubled the live-forward return at lower DD) + **regime gate** (don't open
against the confirmed HTF trend — cuts the downtrend-trapped-long tail) + **risk 3.0%**
(the return dial). NOTE: strictness <2.0 hits a backtest DQ cliff; 2.25 stays clear of it
and is live-margin-validated.

`risk_per_basket_frac` is the return dial. Fresh frontier: no-DQ to ~4.8% backtest;
live MT5 margin model holds min level ~6.3 (21x the 0.30 DQ line) at 3.0%. 3.0% chosen
as the moderate point (~2x the old 1.8% weekly return, healthy margin vs the cliff).
Everything else is frozen at the validated values; do not tune live without re-running
the backtest gates.
"""
from __future__ import annotations

from . import _paths  # noqa: F401
from backtest.config import BacktestConfig


def build_live_config(
    initial_capital: float = 1_000_000.0,
    risk_per_basket_frac: float = 0.024,
    entry_strictness: float = 2.0,
    vol_scale_k: float = 0.90,
    account_leverage: float = 100.0,  # base default; the COMPETITION 1:30 is applied by
                                      # build_comp_config / comp_sim (validated by gate3@30).
                                      # Kept at 100 here so gate2 execution-parity isn't
                                      # confounded by a margin-model DQ on one side only.
    regime_gate: bool = True,
    regime_gate_margin: float = 20.0,
    max_depth: int = 4,
    **overrides,
) -> BacktestConfig:
    """Construct the shipped winner BacktestConfig (overridable for sweeps/env).

    TIER-1 ALIGNMENT (2026-06-21): tightened defaults to match the live signal's
    observed behaviour (real account 2234595 over 23 weeks of 2026 — 96% weekly
    win rate, no catastrophic week, scaled to $1M ~+0.4%/wk):
      - max_depth 6 -> 4  (live signal never grids 6 deep; RISK_FINDINGS confirms
        the holdout-validated cap that halves tail metrics with no edge loss)
      - entry_strictness 2.25 -> 2.0  (matches the EA's basket cadence; sweep_week
        showed s2.0 dominates s2.25 on every recent slice without DQ)
      - risk_per_basket_frac 0.030 -> 0.024  (compounds with the smaller grid for
        comparable per-basket $ risk vs the shipped 6-deep config)
    """
    cfg = BacktestConfig(
        initial_capital=initial_capital,
        account_leverage=account_leverage,
        # --- both-dir ride-to-recovery: NO hard stop, NO time stop ---
        long_only=False,
        hard_stop=False,
        max_hold_hours=0.0,
        # --- TIER-1: cap grid depth at 4 (matches live signal behaviour) ---
        max_depth=max_depth,
        # --- htf trend-flip exit with ATR-floored MAE gate (the tail killer) ---
        trend_flip_exit=True,
        flip_mode="htf",
        flip_htf_lookback=2160,
        flip_max_mae_usd=60.0,
        flip_mom_margin_usd=10.0,
        flip_mae_atr_mult=4.0,
        flip_persist_bars=30,
        flip_min_age_min=60.0,
        # --- entry selectivity ---
        entry_strictness=entry_strictness,
        # --- vol-scaled take-profit (drift correction: gold/ATR doubled) ---
        tp_mode="vol_scaled",
        vol_scale_k=vol_scale_k,
        vol_clamp=(1.0, 12.0),
        # --- regime entry gate: don't open against a confirmed HTF trend ---
        # (the downtrend-trapped-long tail killer; lookback MUST == flip_htf_lookback
        # because the live FeatureRow carries a single ma200_lag at flip_htf_lookback)
        regime_gate=regime_gate,
        regime_gate_lookback=2160,
        regime_gate_margin=regime_gate_margin,
        # --- sizing return dial ---
        risk_per_basket_frac=risk_per_basket_frac,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# --------------------------------------------------------------------------- #
# Competition (Quanthack 21-27 Jun) — phased, schedule-aware config.
#
# The schedule is a survive-then-maximize tournament: 3 daily elimination cuts
# (22:00, rank-based top-N advance) then a 48h Finals for the Top 100. research/
# comp_sim.py scored the shipped strategy over every 2026 comp-shaped week at the
# COMPETITION margin model (1:30 per official rules, force-liq <= 0.30) and found:
#   * margin-call DQ (0.30) STILL never happens even at 1:30 (100% survival to 4.5%
#     risk — max_depth=4 keeps notional small, so even an 80% equity DD doesn't breach
#     30% margin). DQ is NOT the binding constraint.
#   * BUT at 1:30 the live margin-guard bailout (hard_floor 0.40) now fires above
#     ~0.018 risk on sustained adverse CONTINUOUS regimes (gate3 @30 FULL slice bails
#     out at 0.030). The continuous-safe operating ceiling dropped ~0.048 (@1:100) ->
#     ~0.018 (@1:30). Short flat-start 5-day comp windows hold margin >0.40 even at
#     0.045, but a worse-than-2026 comp week could bail out -> anchor qualifying <=0.018.
#   * DRAWDOWN is the real binding constraint (leverage-INDEPENDENT: sizing is
#     risk-$-based, not leverage-based), scaling ~linearly with risk_per_basket_frac
#     (worst week 2026-03-01: 25% DD @1.2% -> 81% @4.5%). It tanks Return-rank + DD-rank
#     at the cut snapshots (floating-inclusive marks).
# So the optimal posture is phased: size LOW in qualifying (you only need top-N, not
# #1 — minimize the chance a violent day posts an eliminating drawdown snapshot), then
# raise risk in the Finals where return dominates over a short, margin-safe window.
# The binding edge lever is SNAPSHOT-DRAWDOWN management (tighter TP to bank floating
# gains, pre-cut trim of floating losers) — see research/comp_score.py + comp_ab.py.
# (Full-flatten / cat_stop round-cut were rejected: they forfeit the ride-to-recovery
# edge that keeps the weekly median positive — 06-20 sweep.)
QUALIFYING_RISK = 0.030   # R1-R3 (hard stop bounds the tail, so qualifying runs hotter than old 0.018)
FINALS_RISK = 0.045       # 48h Finals: return-max
COMP_LEVERAGE = 30.0      # 1:30 per official comp rules (was 100; "confirmed 1:100" unverified — terminal check pending)
COMP_STOP_OUT = 0.30      # force-liquidation / DQ at margin level <= 0.30

# --- Posture presets (the validated hard-stop configs from research/comp_ab.py @1:30) ---
# A hard stop is the tail killer: comp_ab over 75 weeks (2025-26) showed it cuts the
# worst-week snapshot DD from 60-81% to 16-41% with 0 margin-DQ. Two deployable shapes:
#   * hs30  (coupled):  size+freeze+stop all @30$/oz  -> 0/75 eliminating weeks (DD>25), upside capped
#   * dec20_45 (decoupled): size@20 (big lots = full trend upside), ride to a 45$/oz catastrophe stop
#     -> recovers nostop's +31%/+46% good-week upside while halving the tail
# BALANCED  = survive the cuts (hs30 qualifying) then swing in finals (dec20_45).
# AGGRESSIVE = swing through the cuts too (dec20_45 qualifying, dec15_45 finals).
_HS30 = dict(hard_stop=True, max_adverse_usd=30.0, hard_stop_usd=0.0)
_DEC = lambda size, stop: dict(hard_stop=True, max_adverse_usd=float(size), hard_stop_usd=float(stop))
COMP_POSTURE = "balanced"   # default; set "aggressive" (or pass posture=) to swing through the cuts
_POSTURE = {
    "balanced": {
        "qualifying": {"risk_per_basket_frac": QUALIFYING_RISK, **_HS30},
        "finals":     {"risk_per_basket_frac": FINALS_RISK, **_DEC(20, 45)},
    },
    "aggressive": {
        "qualifying": {"risk_per_basket_frac": QUALIFYING_RISK, **_DEC(20, 45)},
        "finals":     {"risk_per_basket_frac": FINALS_RISK, **_DEC(15, 45)},
    },
}

# Round-cut timestamps (UTC). 22:00 London (BST) -> 21:00 UTC  [verify before launch].
import pandas as pd  # noqa: E402

COMP_START = pd.Timestamp("2026-06-21 21:00")   # Sun 22:00 London
FINALS_START = pd.Timestamp("2026-06-24 21:00")  # Wed 22:00 London (R3 cut -> Finals)
FINALS_END = pd.Timestamp("2026-06-26 21:00")    # Fri 22:00 London


def phase_for_time(now: "pd.Timestamp | None" = None) -> str:
    """'qualifying' (R1-R3) or 'finals' (>= FINALS_START). Pre-start -> 'qualifying'."""
    now = pd.Timestamp(now) if now is not None else COMP_START
    return "finals" if now >= FINALS_START else "qualifying"


def build_comp_config(phase: str = "qualifying", posture: str | None = None,
                      **overrides) -> BacktestConfig:
    """Phased competition config: 1:30 / DQ 0.30, hard-stop tail control.

    posture: 'balanced' (default; hs30 qualifying -> dec20_45 finals) or 'aggressive'
    (dec20_45 qualifying -> dec15_45 finals). The hard-stop params + risk come from the
    validated _POSTURE presets; all other levers stay frozen at build_live_config values.

    Operationally the live bot switches phase by restarting with phase='finals' at
    FINALS_START (reconcile adopts any open basket), or calls phase_for_time() per bar.
    """
    posture = posture or COMP_POSTURE
    params = dict(_POSTURE[posture][phase])
    params.update(overrides)   # caller overrides win (sweeps / live tweaks)
    cfg = build_live_config(account_leverage=COMP_LEVERAGE, **params)
    cfg.stop_out_frac = COMP_STOP_OUT
    return cfg
