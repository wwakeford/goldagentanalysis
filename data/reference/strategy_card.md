# Quantum Queen XAUUSD — entry rule (condensed reference)

Reverse-engineered from 873 live baskets (2 MT5 accounts, 2024-04 → 2026-06). This is
a **reference** the replay agent is given as context — not a mandate. The agent may deviate.

**One line:** during an active **session window**, when the **4h trend** is strong enough and
price **breaks to the trend-side extreme of the last 60 min**, open the first leg **in the
direction of the 4h trend**. Grid-add on adverse moves; close the whole basket at an aggregate
take-profit.

- **Direction** = sign of 4h momentum (`mom_4h`). 97% of live baskets match; never fight the 4h trend.
- **Session windows** (broker time, UTC+3) and their thresholds:
  | window | broker hours | `|mom_4h|` floor | breakout `rng_pos` |
  |---|---|---|---|
  | evening (dominant) | 18–23 | ~$20 | ≥ 0.80 long / ≤ 0.20 short |
  | midnight | ~3 | ~$18 | ≥ 0.75 (≈100% long here) |
  | morning | 8–11 | ~$10 | ≥ 0.82 (more two-sided) |
- **Grid**: add a leg on ~$1.90 adverse move, lots mostly flat, depth median shallow (cap 6 here).
- **Close**: the WHOLE basket at an aggregate take-profit ≈ **$1/oz from VWAP** (≈ $100/lot),
  median hold ~9 min. 98% of live baskets close green.
- **Tail risk**: a basket armed **against** the higher-timeframe trend that price never reverts to —
  it grinds the grid to its max adverse / stop. Recognising and NOT arming this is the edge.
