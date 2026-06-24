"""
Technical indicators implemented in pure pandas/numpy.

Each function accepts a DataFrame with columns:
    open, high, low, close, volume
and a DatetimeIndex.

All functions return either a pd.Series or pd.DataFrame
with clearly named columns.
"""
import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ── Trend ─────────────────────────────────────────────────────────────────────

def ema(df: pd.DataFrame, period: int) -> pd.Series:
    """Exponential Moving Average of close prices."""
    return df["close"].ewm(span=period, adjust=False, min_periods=period).mean().rename(f"ema_{period}")


# ── Momentum ──────────────────────────────────────────────────────────────────

def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Relative Strength Index (0–100).
    > 70 = overbought, < 30 = oversold.
    Uses Wilder's smoothing (equivalent to EMA with α = 1/period).
    """
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).fillna(0)   # day-1 diff is NaN → treat as no change
    loss  = (-delta).clip(lower=0).fillna(0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - 100 / (1 + rs)
    # avg_loss == 0 means no losses at all → RSI = 100 (all gains)
    result = result.fillna(100.0)
    return result.rename(f"rsi_{period}")


def macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    MACD indicator.
    Returns DataFrame with columns: macd, signal, histogram.
    """
    ema_fast   = df["close"].ewm(span=fast,   adjust=False).mean()
    ema_slow   = df["close"].ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return pd.DataFrame({
        "macd":      macd_line,
        "signal":    signal_line,
        "histogram": histogram,
    })


# ── Volatility ────────────────────────────────────────────────────────────────

def bollinger_bands(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.DataFrame:
    """
    Bollinger Bands.
    Returns DataFrame with columns: bb_upper, bb_middle, bb_lower, bb_width, bb_pct.
    bb_pct = where close sits within the band (0 = lower, 1 = upper).
    """
    middle = df["close"].rolling(period).mean()
    std    = df["close"].rolling(period).std(ddof=0)
    upper  = middle + std_dev * std
    lower  = middle - std_dev * std
    band_range = (upper - lower).replace(0, np.nan)
    return pd.DataFrame({
        "bb_upper":  upper,
        "bb_middle": middle,
        "bb_lower":  lower,
        "bb_width":  band_range / middle,          # normalised band width
        "bb_pct":    (df["close"] - lower) / band_range,  # %B
    })


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average True Range.
    True Range = max(H-L, |H-prev_C|, |L-prev_C|).
    Uses Wilder's smoothing.
    """
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean().rename(f"atr_{period}")


# ── Volume ────────────────────────────────────────────────────────────────────

def vwap(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Rolling VWAP over `period` bars.
    For daily bars this is a multi-day rolling average.
    Typical price = (high + low + close) / 3.
    """
    tp   = (df["high"] + df["low"] + df["close"]) / 3
    tpv  = tp * df["volume"]
    return (
        tpv.rolling(period).sum() / df["volume"].rolling(period).sum()
    ).rename("vwap")


# ── Trend strength ────────────────────────────────────────────────────────────

def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Average Directional Index.
    Returns DataFrame with columns: adx, adx_plus (+DI), adx_minus (-DI).
    ADX > 25 = trending market, < 20 = ranging/choppy.
    """
    prev_high  = df["high"].shift(1)
    prev_low   = df["low"].shift(1)
    prev_close = df["close"].shift(1)

    # True range
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Directional movement
    up_move   = df["high"] - prev_high
    down_move = prev_low   - df["low"]

    plus_dm  = np.where((up_move > down_move) & (up_move > 0),   up_move,   0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    # Wilder smoothing
    alpha    = 1 / period
    atr_w    = pd.Series(plus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean()  # reuse shape
    atr_w    = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_di  = 100 * pd.Series(plus_dm,  index=df.index).ewm(alpha=alpha, adjust=False).mean() / atr_w
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean() / atr_w

    dx  = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx_val = dx.ewm(alpha=alpha, adjust=False).mean().clip(0, 100)  # defined range [0,100]

    return pd.DataFrame({
        "adx":       adx_val,
        "adx_plus":  plus_di,
        "adx_minus": minus_di,
    })
