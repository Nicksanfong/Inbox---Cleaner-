import logging

import pandas as pd

from indicators.base import (
    ema, rsi, macd, bollinger_bands, atr, vwap, adx
)

log = logging.getLogger(__name__)

EMA_PERIODS = [8, 21, 50, 200]


def run_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run every indicator on a price DataFrame and return a single combined DataFrame.

    Input df must have columns: open, high, low, close, volume
    with a DatetimeIndex.

    Returns the original OHLCV columns plus all indicator columns,
    sorted by date (oldest first).
    """
    if df.empty:
        raise ValueError("DataFrame is empty — cannot compute indicators.")

    required = {"open", "high", "low", "close", "volume"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")

    out = df.copy()

    # EMAs
    for p in EMA_PERIODS:
        out[f"ema_{p}"] = ema(df, p)

    # RSI
    out["rsi_14"] = rsi(df)

    # MACD
    _macd = macd(df)
    out["macd"]      = _macd["macd"]
    out["macd_sig"]  = _macd["signal"]
    out["macd_hist"] = _macd["histogram"]

    # Bollinger Bands
    _bb = bollinger_bands(df)
    out["bb_upper"]  = _bb["bb_upper"]
    out["bb_middle"] = _bb["bb_middle"]
    out["bb_lower"]  = _bb["bb_lower"]
    out["bb_width"]  = _bb["bb_width"]
    out["bb_pct"]    = _bb["bb_pct"]

    # ATR
    out["atr_14"] = atr(df)

    # VWAP (rolling 20-bar)
    out["vwap"] = vwap(df)

    # ADX
    _adx = adx(df)
    out["adx"]       = _adx["adx"]
    out["adx_plus"]  = _adx["adx_plus"]
    out["adx_minus"] = _adx["adx_minus"]

    indicator_cols = [c for c in out.columns if c not in {"open", "high", "low", "close", "volume"}]
    rows_before = len(out)
    out = out.dropna(subset=indicator_cols)
    log.info(
        "run_all: %d input rows → %d usable rows after indicator warmup",
        rows_before, len(out),
    )
    return out
