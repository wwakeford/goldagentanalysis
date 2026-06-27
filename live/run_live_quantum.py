"""Live entrypoint for the Quantum Queen recovery grid — run ON THE WINDOWS BOX.

Wires the shipped (validated, causal-feature) config + no-stop-aware margin sizing +
QuantumRunner to an MT5 account, reading credentials/knobs from environment variables.
Backfills ~2400+ M1 bars at startup so the htf trend-flip tail-protector is armed from
the first live bar, prints a sizing + warm-up preview, then trades once per closed bar.

Usage (PowerShell on the Windows box):
    $env:MT5_LOGIN="12345678"; $env:MT5_PASSWORD="..."; $env:MT5_SERVER="Broker-Server"
    python run_live_quantum.py --check     # validate cfg + sizing + warmup, NO connect
    python run_live_quantum.py             # live, dynamic margin-safe sizing off equity

Env knobs:
    MT5_LOGIN / MT5_PASSWORD / MT5_SERVER / MT5_PATH, SYMBOL (XAUUSD)
    PHASE (qualifying|finals; default = phase_for_time()), POSTURE (balanced|aggressive)
    RISK_FRAC / ENTRY_STRICTNESS / VOL_SCALE_K / REGIME_GATE_MARGIN (override the preset)
    LEVERAGE (30; account margin leverage per official comp rules — CONFIRM in the terminal)
    MARGIN_BUFFER (2.5), MAX_RIDE_MAE (60 $/oz; ~the hard-stop distance), HARD_FLOOR (0.40), SOFT_FLOOR (0.80)
    QUANTUM_REPO (path to the copied quantumstrategy repo: backtest/ + scripts/)
A KILL file in data/live_logs/ flattens + halts at any time.

DEPLOYS THE VALIDATED COMPETITION CONFIG (build_comp_config): phased hard-stop tail
control, 1:30 margin model, DQ 0.30. balanced = hs30 qualifying -> dec20_45 finals.
"""
from __future__ import annotations

import os
import sys

from quantum.config_live import build_comp_config, phase_for_time, COMP_POSTURE, COMP_LEVERAGE
from quantum.live_quantum import QuantumRunner, QRunnerConfig, _PPL
from quantum import backfill as backfill_mod


def _f(name, default):
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else default


def build_cfg():
    """The validated competition config (hard-stop tail control + phased risk @1:30).

    PHASE selects qualifying (hs30) vs finals (dec20_45); POSTURE selects balanced vs
    aggressive. Env vars override individual preset fields for live tweaks.
    """
    phase = os.environ.get("PHASE") or phase_for_time()
    posture = os.environ.get("POSTURE", COMP_POSTURE)
    overrides = {}
    for env, fld in (("RISK_FRAC", "risk_per_basket_frac"),
                     ("ENTRY_STRICTNESS", "entry_strictness"),
                     ("VOL_SCALE_K", "vol_scale_k"),
                     ("REGIME_GATE_MARGIN", "regime_gate_margin")):
        if os.environ.get(env):
            overrides[fld] = _f(env, None)
    cfg = build_comp_config(phase, posture=posture, **overrides)
    cfg._phase, cfg._posture = phase, posture   # for the preview banner
    return cfg


def build_rcfg():
    # LEVERAGE defaults to the comp 1:30; MAX_RIDE_MAE ~the hard-stop distance (the basket
    # closes at the stop, so reserving margin for a 150$/oz ride would needlessly shrink lots).
    return QRunnerConfig(
        size_mode="margin",
        leverage=_f("LEVERAGE", COMP_LEVERAGE),
        margin_buffer=_f("MARGIN_BUFFER", 2.5),
        max_ride_mae=_f("MAX_RIDE_MAE", 60.0),
        hard_floor=_f("HARD_FLOOR", 0.40),
        soft_floor=_f("SOFT_FLOOR", 0.80),
    )


def _sizing_preview(cfg, rcfg, equity=1_000_000.0, price=4300.0):
    """Mirror QuantumRunner._size_basket arithmetic for a no-connect preview."""
    import math
    from backtest.sizing import leg_lot
    eng = leg_lot(cfg, price)
    mpl = price * _PPL / rcfg.leverage
    t_cap = equity / (rcfg.max_ride_mae * _PPL + rcfg.margin_buffer * mpl)
    leg_cap = t_cap / max(1, cfg.max_depth)
    lot = max(cfg.lot_step, math.floor(min(eng, leg_cap) / cfg.lot_step) * cfg.lot_step)
    total = lot * cfg.max_depth
    used_full = total * mpl
    # margin level after a max_ride_mae full-depth adverse excursion
    eq_exc = equity - rcfg.max_ride_mae * total * _PPL
    ml_exc = eq_exc / used_full if used_full else float("inf")
    gross_full = total * price * _PPL / _PPL  # = total*price*100; leverage = gross/equity
    lev_exposure = (total * price * 100.0) / equity
    return dict(eng_lot=eng, leg_cap=leg_cap, lot=lot, total_lots=total,
                used_full=used_full, ml_at_excursion=ml_exc, lev_exposure=lev_exposure,
                binding=("margin" if leg_cap < eng else "strategy"))


def main():
    check = "--check" in sys.argv
    cfg = build_cfg()
    rcfg = build_rcfg()
    symbol = os.environ.get("SYMBOL", "XAUUSD")

    print("=== Quantum Queen live (goldreplay/live) ===")
    print(f"COMP: phase={getattr(cfg,'_phase','?')}  posture={getattr(cfg,'_posture','?')}  "
          f"hard_stop={cfg.hard_stop}(size@{cfg.max_adverse_usd:.0f}/stop@"
          f"{(cfg.hard_stop_usd or cfg.max_adverse_usd):.0f}$oz)  acct_lev=1:{cfg.account_leverage:.0f}  "
          f"stop_out={cfg.stop_out_frac}")
    print(f"config: risk={cfg.risk_per_basket_frac}  entry_strictness={cfg.entry_strictness}  "
          f"tp={cfg.tp_mode}(k={cfg.vol_scale_k},clamp={cfg.vol_clamp})  "
          f"flip=htf(lb={cfg.flip_htf_lookback},mae>={cfg.flip_max_mae_usd}|{cfg.flip_mae_atr_mult}xATR)  "
          f"regime_gate={'on@$%.0f' % cfg.regime_gate_margin if cfg.regime_gate else 'off'}  "
          f"max_depth={cfg.max_depth}")
    print(f"runner: leverage={rcfg.leverage}  margin_buffer={rcfg.margin_buffer}  "
          f"max_ride_mae={rcfg.max_ride_mae}  soft_floor={rcfg.soft_floor}  "
          f"hard_floor={rcfg.hard_floor} (comp force-liq=0.30)")
    p = _sizing_preview(cfg, rcfg)
    print(f"\nsizing @ $1M / ${4300:.0f}: lot/leg={p['lot']:.2f}  full-depth={p['total_lots']:.2f} lots  "
          f"used_margin=${p['used_full']:,.0f}  binding={p['binding']}")
    print(f"  after a {rcfg.max_ride_mae:.0f}$/oz full-depth adverse ride: margin level="
          f"{p['ml_at_excursion']:.2f}  exposure-leverage={p['lev_exposure']:.1f}x (comp penalises >28x)")
    need = cfg.flip_htf_lookback + 240
    print(f"  htf flip arms after ~{need} M1 bars (~{need/1440:.1f} trading days) of warm-up")

    if check:
        print("\n--check: config + sizing validated. NOT connecting / trading.")
        return 0

    from broker import make_broker
    broker = make_broker(
        "mt5", symbol=symbol, leverage=rcfg.leverage,
        login=int(os.environ["MT5_LOGIN"]) if os.environ.get("MT5_LOGIN") else None,
        password=os.environ.get("MT5_PASSWORD"),
        server=os.environ.get("MT5_SERVER"),
        path=os.environ.get("MT5_PATH"),
    )
    runner = QuantumRunner(broker, cfg, rcfg)

    print("\nbackfilling M1 history for feature warm-up ...")
    hist = backfill_mod.backfill(broker, cfg)
    seeded = runner.seed_features(hist)
    armed = runner.feats.update  # noop ref; check arming via last seeded row
    print(f"  seeded {seeded} bars from {hist.attrs.get('source')}  "
          f"(need {need}; flip {'ARMED' if seeded >= need else 'NOT yet armed — trading, flip dormant'})")

    acc = broker.account()
    print(f"\nconnected: equity=${acc.equity:,.0f}  starting live loop (once per closed M1 bar)\n")
    runner.run_live()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
