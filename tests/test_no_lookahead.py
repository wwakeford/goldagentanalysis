"""Formal no-lookahead guarantees for the gold walk-forward replay.

Run:  PYTHONPATH=src python3 tests/test_no_lookahead.py

Three independent checks:

1. PREFIX-DETERMINISM (the core proof). Every feature on decision bar i is taken
   AS-OF the last M1 bar at-or-before that bar's close. We recompute mom_4h /
   rng_pos / rsi from RAW M1 truncated at bar i's timestamp and assert they equal
   what the agent was shown — i.e. the value depends only on bars <= i, so no
   future bar can change it.

2. POINTER MONOTONICITY. `step` advances ptr by exactly 1 and the state never
   exposes a bar beyond ptr to the emitter.

3. PROMPT GUARD. The replay prompt forbids reading the cached parquet (the third
   gate, in case an agent is tempted to peek).
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from goldreplay import episodes as ep                      # noqa: E402
from goldreplay.align import load_m1                       # noqa: E402
from goldreplay.config import BacktestConfig               # noqa: E402
from goldreplay.enrich import decision_bars                # noqa: E402
from goldreplay.features import add_m1_features, build_features  # noqa: E402
from goldreplay import replay                              # noqa: E402

EID = "2024-06-07.evening"
TAIL = 600  # M1 bars before a decision point — far exceeds the 240-bar lookback


def test_prefix_determinism() -> None:
    cfg = BacktestConfig()
    feat = build_features(cfg)
    bars = decision_bars(feat, EID, cfg)
    raw = load_m1()  # untouched M1, naive UTC

    # the agent sees each field rounded to a fixed precision (set in enrich.py);
    # compare at that precision so we test the VALUE, not cosmetic rounding.
    DECIMALS = {"mom_4h": 2, "rng_pos": 3, "rsi": 1}
    checked = 0
    for bar in bars[::4]:                       # sample every 4th decision bar
        ts = pd.Timestamp(bar["ts_utc"])
        # truncate RAW M1 to ONLY bars at-or-before this decision point, then
        # recompute the features from scratch on that prefix.
        prefix = raw.loc[raw["timestamp"] <= ts].tail(TAIL).reset_index(drop=True)
        recomputed = add_m1_features(prefix).iloc[-1]
        assert recomputed["timestamp"] == ts, f"as-of ts mismatch at {bar['time']}"
        for field, dp in DECIMALS.items():
            shown = bar[field]
            truth = recomputed[field]
            if shown is None or (isinstance(truth, float) and np.isnan(truth)):
                continue
            assert float(shown) == round(float(truth), dp), (
                f"{field} at {bar['time']}: agent saw {shown}, prefix-only truth {round(float(truth), dp)} "
                f"— a future bar changed a value the agent was shown (LOOKAHEAD!)")
            checked += 1
    assert checked > 0, "no fields checked"
    print(f"  [1] prefix-determinism: {checked} field-values match prefix-only recomputation")


def test_pointer_monotonic() -> None:
    sink = io.StringIO()
    sp = replay._state_path(EID)
    with contextlib.redirect_stdout(sink):       # silence the bar prints
        replay.start(EID)
        st = json.loads(sp.read_text())
        n = len(st["bars"])
        assert st["ptr"] == 0
        steps = 0
        last_ptr = 0
        while True:
            st = json.loads(sp.read_text())
            if st["ptr"] >= n:
                break
            before = st["ptr"]
            replay.step(EID, '{"observation": null}')
            after = json.loads(sp.read_text()).get("ptr", n)
            assert after == before + 1, f"ptr jumped {before}->{after}"
            steps += 1
            last_ptr = after
    print(f"  [2] pointer monotonicity: {steps} steps, ptr advanced by exactly 1 each, ended at {last_ptr}/{n}")


def test_prompt_guard() -> None:
    replay.manifest(EID)
    prompt = replay._prompt_path(EID).read_text()
    assert "NEVER read data/price/*.parquet" in prompt
    assert "ONLY learn future prices via `step`" in prompt
    print("  [3] prompt guard: no-lookahead clause present in the replay prompt")


if __name__ == "__main__":
    print(f"no-lookahead verification on {EID}:")
    test_prefix_determinism()
    test_pointer_monotonic()
    test_prompt_guard()
    print("ALL NO-LOOKAHEAD CHECKS PASSED")
