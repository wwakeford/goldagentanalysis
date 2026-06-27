"""Gate 3 — margin-floor / no-force-liquidation validation in the live stack.

Drives the QuantumRunner in live ``margin`` sizing mode through the SimBroker over
the worst historical regime (DOWN_2026: 2026-01-28 -> cutoff — the steamroller slice
that previously force-liquidated at risk >=1.5%) and the full sample, tracking the
MINIMUM margin level while in a position. The comp force-liquidates (instant DQ) at
margin level 0.30; we require the live stack to stay above the runner's hard_floor
(0.40, itself above 0.30) at the chosen sizing, and to never self-flatten.

Reports a small sizing frontier so the return dial can be turned with eyes open.
"""
from __future__ import annotations

import sys

from . import _paths  # noqa: F401
from price_align import load_m1
from quantum.config_live import build_live_config
from quantum.quantum_features import causal_feature_frame
from quantum.live_quantum import QuantumRunner, QRunnerConfig
from broker import SimBroker

SLICES = {
    "FULL": (None, None),
    "DOWN_2026": ("2026-01-28", None),
    "HOLDOUT": ("2026-05-01", None),
}


def _slice(frame, lo, hi):
    f = frame
    if lo is not None:
        f = f[f["timestamp"] >= lo]
    if hi is not None:
        f = f[f["timestamp"] < hi]
    return f.reset_index(drop=True)


def run(leverage: float = 30.0, hard_floor: float = 0.40,
        operating_risk: float = 0.030, risks=(0.018, 0.030, 0.045, 0.060),
        max_ride_mae: float = 150.0, margin_buffer: float = 2.5,
        cfg_overrides: dict | None = None) -> bool:
    """Two guarantees, reported separately:
      * DQ-avoided  — the broker margin level never reaches the comp's 0.30 line
        (no force-liquidation) on ANY slice at ANY tested sizing. This is the hard,
        non-negotiable guarantee; the voluntary hard_floor bailout (0.40 > 0.30)
        backstops it.
      * ride-safe   — at the OPERATING sizing the bailout is never even needed
        (margin stays comfortably above hard_floor), so the full ride-to-recovery
        behaviour is preserved.
    Gate passes iff DQ-avoided everywhere AND ride-safe at the operating sizing.
    """
    m = load_m1()
    dq_ok = True
    ride_ok = True
    print(f"leverage={leverage:.0f}  hard_floor={hard_floor}  max_ride_mae={max_ride_mae}  "
          f"margin_buffer={margin_buffer}  operating_risk={operating_risk}  "
          f"(comp force-liq line = 0.30)\n")
    print(f"{'risk':>6} {'slice':>10} {'ret%':>9} {'maxDD%':>7} {'minMargLvl':>11} "
          f"{'bailout':>8} {'DQ-avoided':>11}")
    if cfg_overrides:
        print(f"cfg_overrides: {cfg_overrides}\n")
    for risk in risks:
        cfg = build_live_config(risk_per_basket_frac=risk, **(cfg_overrides or {}))
        frame_all = causal_feature_frame(cfg, m)
        for name, (lo, hi) in SLICES.items():
            frame = _slice(frame_all, lo, hi)
            broker = SimBroker(balance=cfg.initial_capital, leverage=leverage,
                               commission_per_lot_side=0.0)
            rcfg = QRunnerConfig(size_mode="margin", leverage=leverage,
                                 hard_floor=hard_floor, max_ride_mae=max_ride_mae,
                                 margin_buffer=margin_buffer, log=False)
            out = QuantumRunner(broker, cfg, rcfg).run_replay(frame, track=True)
            ret = (out["equity"] - cfg.initial_capital) / cfg.initial_capital
            ml = out["min_margin_level"]
            bailout = out["halted"]
            # bailout fires AT hard_floor, above 0.30, so DQ is avoided as long as the
            # min observed margin level didn't punch through 0.30 in a single bar.
            dq_avoided = ml > 0.30
            dq_ok &= dq_avoided
            if abs(risk - operating_risk) < 1e-9:
                ride_ok &= (not bailout) and (ml > hard_floor)
            mark = " <-OPERATING" if abs(risk - operating_risk) < 1e-9 else ""
            print(f"{risk:>6.3f} {name:>10} {ret*100:>9.2f} {out['max_drawdown']*100:>7.2f} "
                  f"{ml:>11.2f} {str(bailout):>8} {'YES' if dq_avoided else 'NO':>11}{mark}")
        print()
    passed = dq_ok and ride_ok
    print(f"DQ-avoided on all slices/sizings: {'YES' if dq_ok else 'NO'}")
    print(f"ride-safe at operating risk {operating_risk}: {'YES' if ride_ok else 'NO'}")
    print("GATE 3:", "PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    lev = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    raise SystemExit(0 if run(leverage=lev) else 1)


if __name__ == "__main__":
    lev = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
    raise SystemExit(0 if run(leverage=lev) else 1)
