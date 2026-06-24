#!/usr/bin/env python3
"""
Show all indicators running on a realistic SPY-like price series.
Tries to load real data from SQLite first; falls back to a synthetic
random-walk series seeded for reproducibility.

Usage:
    python scripts/show_indicators.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from indicators.combined import run_all


def _load_from_db(symbol: str = "SPY") -> pd.DataFrame | None:
    """Return OHLCV DataFrame from SQLite, or None if empty."""
    try:
        from data.database import fetch_bars, DB_PATH
        rows = fetch_bars(symbol, limit=500, db_path=DB_PATH)
        if len(rows) < 210:
            return None
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").set_index("timestamp")
        return df[["open", "high", "low", "close", "volume"]]
    except Exception:
        return None


def _synthetic_df(n: int = 400, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic SPY-like random-walk price series.
    Daily returns are normally distributed with slight upward drift.
    """
    rng     = np.random.default_rng(seed)
    returns = rng.normal(loc=0.0003, scale=0.010, size=n)  # ~7% annual drift, ~16% vol
    close   = 450.0 * np.cumprod(1 + returns)

    daily_range = close * rng.uniform(0.004, 0.012, size=n)  # 0.4–1.2% daily range
    high   = close + daily_range * rng.uniform(0.3, 0.7, size=n)
    low    = close - daily_range * rng.uniform(0.3, 0.7, size=n)
    open_  = low + (high - low) * rng.uniform(0, 1, size=n)
    volume = rng.integers(50_000_000, 120_000_000, size=n).astype(float)

    idx = pd.date_range("2023-01-03", periods=n, freq="B")  # business days
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _bar(label: str, value: float, lo: float, hi: float, width: int = 20) -> str:
    """ASCII progress bar showing where a value sits in [lo, hi]."""
    pct   = max(0.0, min(1.0, (value - lo) / (hi - lo))) if hi != lo else 0.5
    filled = int(pct * width)
    bar   = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {value:6.2f}"


def main() -> None:
    # ── Load data ──────────────────────────────────────────────────────────────
    df = _load_from_db("SPY")
    if df is not None:
        source = "SQLite (real SPY data)"
    else:
        df     = _synthetic_df(n=400)
        source = "synthetic random-walk (SPY-like, seed=42)"

    # ── Run indicators ─────────────────────────────────────────────────────────
    result = run_all(df)
    last   = result.iloc[-1]
    prev   = result.iloc[-2]
    tail   = result.tail(5)

    # ── Print report ──────────────────────────────────────────────────────────
    SEP = "=" * 65
    print(f"\n{SEP}")
    print(f"  Indicator Report  —  {source}")
    print(f"  {len(df)} input bars  →  {len(result)} bars after warmup")
    print(f"  Latest date: {result.index[-1].strftime('%Y-%m-%d')}")
    print(SEP)

    price = last["close"]
    print(f"\n  {'PRICE':20s}  ${price:>10.3f}")

    print(f"\n  ── Trend (EMAs) ──────────────────────────────────────")
    for p in [8, 21, 50, 200]:
        e     = last[f"ema_{p}"]
        above = "▲ above EMA" if price > e else "▼ below EMA"
        print(f"  {'EMA ' + str(p):20s}  ${e:>10.3f}   {above}")

    print(f"\n  ── Momentum ──────────────────────────────────────────")
    rsi_val = last["rsi_14"]
    signal  = "OVERBOUGHT" if rsi_val > 70 else ("OVERSOLD" if rsi_val < 30 else "neutral")
    print(f"  {'RSI(14)':20s}  {_bar('', rsi_val, 0, 100)}   {signal}")

    macd_v  = last["macd"]
    macd_s  = last["macd_sig"]
    macd_h  = last["macd_hist"]
    cross   = "bullish cross ▲" if macd_v > macd_s else "bearish cross ▼"
    print(f"  {'MACD':20s}  {macd_v:>+8.4f}")
    print(f"  {'MACD Signal':20s}  {macd_s:>+8.4f}   {cross}")
    print(f"  {'MACD Histogram':20s}  {macd_h:>+8.4f}")

    print(f"\n  ── Volatility (Bollinger Bands & ATR) ────────────────")
    bb_u  = last["bb_upper"]
    bb_m  = last["bb_middle"]
    bb_l  = last["bb_lower"]
    bb_pct = last["bb_pct"]
    print(f"  {'BB Upper':20s}  ${bb_u:>10.3f}")
    print(f"  {'BB Middle (SMA20)':20s}  ${bb_m:>10.3f}")
    print(f"  {'BB Lower':20s}  ${bb_l:>10.3f}")
    print(f"  {'BB %B (position)':20s}  {_bar('', bb_pct, 0, 1)}   (0=lower, 1=upper)")
    print(f"  {'ATR(14)':20s}  ${last['atr_14']:>10.3f}   daily volatility estimate")

    print(f"\n  ── Volume ────────────────────────────────────────────")
    print(f"  {'VWAP(20)':20s}  ${last['vwap']:>10.3f}   "
          f"{'price ABOVE VWAP ▲' if price > last['vwap'] else 'price below VWAP ▼'}")

    print(f"\n  ── Trend Strength (ADX) ──────────────────────────────")
    adx_v  = last["adx"]
    trend  = "STRONG trend" if adx_v > 25 else ("weak trend" if adx_v > 20 else "ranging/choppy")
    print(f"  {'ADX(14)':20s}  {_bar('', adx_v, 0, 60)}   {trend}")
    print(f"  {'+DI (bullish)':20s}  {last['adx_plus']:>8.2f}")
    print(f"  {'-DI (bearish)':20s}  {last['adx_minus']:>8.2f}   "
          f"{'bulls winning ▲' if last['adx_plus'] > last['adx_minus'] else 'bears winning ▼'}")

    print(f"\n{SEP}")
    print(f"  Last 5 rows (selected columns)")
    print(SEP)
    cols = ["close", "ema_8", "ema_21", "rsi_14", "macd", "bb_upper", "bb_lower", "atr_14", "adx"]
    pd.set_option("display.float_format", "{:.3f}".format)
    pd.set_option("display.max_columns", 10)
    pd.set_option("display.width", 120)
    print(tail[cols].to_string())
    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
