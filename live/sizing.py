"""Position sizing / margin guard for the live competition grid.

The competition ranks on raw equity (aggression pays) but eliminates you the
instant **margin level <= 30%** -- and an averaging grid's whole failure mode is
exactly that. So we never let the leaderboard tempt us past the cliff: size the
per-leg lot so that, with the grid at FULL depth (`max_legs`) and price gone a
worst-case adverse distance against it, margin level still sits far above 30%.

Two independent caps, we take the tighter (smaller lot):

  1. **Margin-level floor (hard safety).** Even after a `margin_excursion_adr` *
     ADR adverse move at full depth, margin level >= `buffer` (default 1.5 = 150%,
     5x the 30% death line). The excursion is set wider than the disaster stop so
     a stop that slips or gaps still can't margin-call us.
  2. **Loss-budget cap (the return/risk dial).** The realised loss when the ADR
     disaster stop fires at full depth is <= `loss_budget_frac` of equity
     (default 0.10 = balanced posture). This is the knob you turn for more/less
     aggression; the margin floor is non-negotiable.

MT5 margin model for XAUUSD: 1 lot = 100 oz, so notional/lot = price * 100 and
required margin/lot = price * 100 / leverage. A $1 (1 point) move on 1 lot = $100
(== ``StrategyParams.usd_per_point_per_lot``). Margin level = equity / used_margin.

Nothing here imports MetaTrader5 -- pure arithmetic, runs and is tested on any OS.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from strategy import StrategyParams

MARGIN_CALL_LEVEL = 0.30   # comp rule: margin level <= 30% => instant elimination


@dataclass
class Sizing:
    lot_per_leg: float          # lot size for EACH grid leg (rounded to lot_step)
    total_lots: float           # lot_per_leg * max_legs (full-depth exposure)
    used_margin_full: float     # margin consumed at full depth ($)
    worst_loss: float           # $ loss if disaster stop fires at full depth
    worst_loss_pct: float       # worst_loss / equity
    margin_level_at_excursion: float  # margin level after the safety excursion
    binding: str                # which cap set the lot ("margin" | "loss_budget")


def margin_per_lot(price: float, leverage: float,
                   usd_per_point_per_lot: float = 100.0) -> float:
    """Required margin for 1 lot of XAUUSD. contract_size (oz) == ppl == 100."""
    return price * usd_per_point_per_lot / leverage


def margin_level(equity: float, used_margin: float) -> float:
    """Equity / used margin (the quantity the comp checks against 30%). No open
    position -> infinite (no margin used), reported as inf."""
    return math.inf if used_margin <= 0 else equity / used_margin


def lot_per_leg(equity: float, price: float, adr: float, p: StrategyParams,
                leverage: float = 30.0, buffer: float = 1.5,
                margin_excursion_adr: float = 1.0, loss_budget_frac: float = 0.10,
                lot_step: float = 0.01, min_lot: float = 0.01,
                max_lot: float | None = None) -> Sizing:
    """Largest per-leg lot that satisfies BOTH the margin floor and loss budget.

    Let N = max_legs and T = N * L (total lots at full depth). Margin used at full
    depth is T * mpl (mpl = margin_per_lot). For an adverse excursion of E points
    from the weighted average, floating loss = E * T * ppl, so:

      margin floor:  (equity - E_m * T * ppl) / (T * mpl) >= buffer
                     => T <= equity / (E_m * ppl + buffer * mpl)
      loss budget:   E_s * T * ppl <= loss_budget_frac * equity
                     => T <= loss_budget_frac * equity / (E_s * ppl)

    with E_m = margin_excursion_adr * ADR (safety) and E_s = adr_stop_k * ADR
    (the disaster-stop distance, the *expected* worst basket). We take min(T),
    split across N legs, floor to the lot step.
    """
    ppl = p.usd_per_point_per_lot
    n = max(1, p.max_legs)
    mpl = margin_per_lot(price, leverage, ppl)

    e_margin = margin_excursion_adr * adr
    t_margin = equity / (e_margin * ppl + buffer * mpl)

    # disaster-stop distance; if no ADR stop configured, fall back to the margin
    # excursion so the loss budget still binds something sane.
    e_stop = (p.adr_stop_k * adr) if p.adr_stop_k > 0 else e_margin
    t_loss = (loss_budget_frac * equity) / (e_stop * ppl) if e_stop > 0 else math.inf

    t = min(t_margin, t_loss)
    binding = "margin" if t_margin <= t_loss else "loss_budget"

    # split across legs, floor to lot step, clamp
    raw_l = t / n
    steps = math.floor(raw_l / lot_step)
    l = max(min_lot, steps * lot_step)
    if max_lot is not None:
        l = min(l, max_lot)
    l = round(l, 2)

    total = l * n
    used_full = total * mpl
    worst = e_stop * total * ppl
    eq_at_exc = equity - e_margin * total * ppl
    return Sizing(
        lot_per_leg=l, total_lots=round(total, 2), used_margin_full=used_full,
        worst_loss=worst, worst_loss_pct=worst / equity if equity else 0.0,
        margin_level_at_excursion=margin_level(eq_at_exc, used_full),
        binding=binding,
    )


if __name__ == "__main__":
    # Stress sweep: across plausible equity / gold price / ADR, the chosen lot
    # must keep margin level after the SAFETY excursion above the buffer and far
    # above the 30% death line -- including a deliberate gold-crash ADR.
    p = StrategyParams.competition()
    print(f"max_legs={p.max_legs} adr_stop_k={p.adr_stop_k}  (buffer=150%, "
          f"loss_budget=10%, excursion=1.0xADR)\n")
    print(f"{'equity':>10} {'price':>7} {'adr':>5} {'lot/leg':>8} "
          f"{'totlots':>8} {'usedM$':>10} {'worst$':>9} {'worst%':>7} "
          f"{'mlvl@exc':>9} {'binding':>11}")
    worst_ml = math.inf
    for equity in (1_000_000, 1_200_000, 800_000):
        for price in (4300.0, 3000.0, 5000.0):
            for adr in (40.0, 80.0, 150.0):   # 150 = crash-day ADR
                s = lot_per_leg(equity, price, adr, p)
                worst_ml = min(worst_ml, s.margin_level_at_excursion)
                print(f"{equity:>10,} {price:>7.0f} {adr:>5.0f} "
                      f"{s.lot_per_leg:>8.2f} {s.total_lots:>8.2f} "
                      f"{s.used_margin_full:>10,.0f} {s.worst_loss:>9,.0f} "
                      f"{s.worst_loss_pct:>6.1%} "
                      f"{s.margin_level_at_excursion:>9.2f} {s.binding:>11}")
                assert s.margin_level_at_excursion >= 1.5 - 1e-6, \
                    f"margin buffer breached: {s.margin_level_at_excursion}"
                assert s.worst_loss_pct <= 0.1001, f"loss budget breached: {s.worst_loss_pct}"
    print(f"\nmin margin level after 1.0xADR full-depth excursion: {worst_ml:.2f} "
          f"(death line = {MARGIN_CALL_LEVEL}) -> "
          f"{'SAFE' if worst_ml >= 1.5 - 1e-6 else 'CHECK'}")
