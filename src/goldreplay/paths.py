"""Central path constants for the goldreplay repo."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

DATA = ROOT / "data"
PRICE_DIR = DATA / "price"
M1_PARQUET = PRICE_DIR / "XAUUSD_M1.parquet"
FEATURES_CACHE = PRICE_DIR / "_features_cache.parquet"

GROUND_TRUTH = DATA / "ground_truth"
BASKETS_PARQUET = GROUND_TRUTH / "baskets.parquet"
TRADES_PARQUET = GROUND_TRUTH / "trades_all.parquet"

REFERENCE = DATA / "reference"
STRATEGY_CARD = REFERENCE / "strategy_card.md"

EPISODES_DIR = ROOT / "episodes"
CATALOG_CSV = EPISODES_DIR / "catalog.csv"

REPORTS = ROOT / "reports"

FINDINGS = ROOT / "findings"
LESSONS_DIR = FINDINGS / "lessons"
LESSONS_INDEX = LESSONS_DIR / "index.json"
REPLAY_TRADES = FINDINGS / "replay_trades"          # .parquet / .csv suffixed at write
REPLAY_EXPECTANCY = FINDINGS / "replay_expectancy.md"
THREE_WAY_MD = FINDINGS / "three_way_comparison.md"
BY_TICKER = FINDINGS / "by_ticker"
BY_PATTERN = FINDINGS / "by_pattern"
