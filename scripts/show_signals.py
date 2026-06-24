#!/usr/bin/env python3
"""
show_signals.py — demo of the SignalEngine with synthetic multi-timeframe data.

Usage:
    python scripts/show_signals.py
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from signals.engine import SignalEngine


# ── Synthetic data ────────────────────────────────────────────────────────────

def _make_ohlcv(n: int, trend: float, seed: int, freq: str = "15min",
                noise: float = 0.3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq=freq)
    close = 100.0 + trend * np.arange(n) + rng.normal(0, noise, n).cumsum()
    high  = close + rng.uniform(0.2, 0.8, n)
    low   = close - rng.uniform(0.2, 0.8, n)
    open_ = close - rng.normal(0, 0.2, n)
    vol   = rng.integers(1000, 5000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


SCENARIOS = [
    {
        "name":   "Strong Uptrend (AAPL-like)",
        "symbol": "AAPL",
        "trend":  +0.35,
        "seed":   1,
    },
    {
        "name":   "Strong Downtrend (bearish)",
        "symbol": "META",
        "trend":  -0.30,
        "seed":   2,
    },
    {
        "name":   "Mild Uptrend",
        "symbol": "MSFT",
        "trend":  +0.08,
        "seed":   3,
    },
    {
        "name":   "Choppy / No Signal",
        "symbol": "XYZ",
        "trend":  +0.01,
        "seed":   4,
    },
]


# ── Formatting ────────────────────────────────────────────────────────────────

_COLS = 72

def _hr(char="─"):
    return char * _COLS

def _header(text: str, char="═"):
    pad = (_COLS - len(text) - 2) // 2
    return f"{char * pad} {text} {char * pad}"


def _print_signal(sig):
    arrow = "▲" if sig.direction == "BUY" else "▼"
    bar = "█" * int(sig.score / 5)
    print(f"  {arrow} {sig.direction:<4}  score={sig.score:5.1f}  {bar}")
    print(f"       timeframe={sig.timeframe}  ts={str(sig.timestamp)[:16]}")
    print(f"       confirmations ({len(sig.confirmations)}):")
    for c in sig.confirmations:
        sign = "+" if c.direction == "bullish" else "-"
        print(f"         [{sign}{c.weight:4.1f}]  {c.source:<25}  {c.detail}")
    print(f"       breakdown:")
    for cat, vals in sig.breakdown.items():
        bull_v = vals.get("bullish", 0)
        bear_v = vals.get("bearish", 0)
        if sig.direction == "BUY":
            print(f"         {cat:<12}  bull={bull_v:.1f}  bear={bear_v:.1f}")
        else:
            print(f"         {cat:<12}  bear={bear_v:.1f}  bull={bull_v:.1f}")


def _print_raw(raw: dict, symbol: str):
    print(f"  Raw per-timeframe scores for {symbol}:")
    for tf, data in raw["by_tf"].items():
        print(f"    {tf:>4}  bull={data['bullish']:5.1f}  bear={data['bearish']:5.1f}")
    print(f"  Combined  bull={raw['combined_bull']:5.1f}  bear={raw['combined_bear']:5.1f}")
    print(f"  MTF bonus bull={raw['mtf_bonus_bull']:4.0f}   bear={raw['mtf_bonus_bear']:4.0f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    engine = SignalEngine(threshold=75.0, min_confirmations=3)
    print()
    print(_header("Signal Engine Demo"))
    print(_hr())
    print(f"  threshold={engine.threshold}   min_confirmations={engine.min_confirmations}")
    print(_hr())

    for scenario in SCENARIOS:
        symbol = scenario["symbol"]
        trend  = scenario["trend"]
        seed   = scenario["seed"]
        print()
        print(_header(scenario["name"], char="─"))

        # Each higher TF multiplies the per-bar trend so price travels
        # a similar total distance, keeping EMA stacks aligned.
        frames = {
            "5m":  _make_ohlcv(500, trend,         seed,     "5min",  noise=0.15),
            "15m": _make_ohlcv(400, trend * 3,     seed + 1, "15min", noise=0.13),
            "30m": _make_ohlcv(350, trend * 6,     seed + 2, "30min", noise=0.12),
            "1H":  _make_ohlcv(350, trend * 12,    seed + 3, "1h",    noise=0.11),
            "4H":  _make_ohlcv(300, trend * 48,    seed + 4, "4h",    noise=0.10),
            "D":   _make_ohlcv(300, trend * 192,   seed + 5, "D",     noise=0.08),
            "W":   _make_ohlcv(260, trend * 960,   seed + 6, "W",     noise=0.06),
        }

        raw     = engine.raw_scores(frames, symbol)
        signals = engine.scan(frames, symbol)

        _print_raw(raw, symbol)
        print()

        if signals:
            print(f"  SIGNALS FIRED ({len(signals)}):")
            for sig in signals:
                _print_signal(sig)
        else:
            print("  No signals met threshold.")

    print()
    print(_hr("═"))
    print()


if __name__ == "__main__":
    main()
