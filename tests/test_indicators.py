"""
Unit tests for the indicator module.

Each test is built around a price series whose expected output
can be derived mathematically, so the assertions are "known values"
rather than just "something came back."
"""
import numpy as np
import pandas as pd
import pytest

from indicators.base import ema, rsi, macd, bollinger_bands, atr, vwap, adx
from indicators.combined import run_all


# ── Helpers ───────────────────────────────────────────────────────────────────

def _const_df(price: float = 100.0, n: int = 60) -> pd.DataFrame:
    """All bars equal — simplest possible known-value dataset."""
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": price, "high": price, "low": price, "close": price, "volume": 1_000_000},
        index=idx,
    )


def _rising_df(start: float = 100.0, step: float = 1.0, n: int = 250) -> pd.DataFrame:
    """Linearly rising close prices — EMA lags, RSI stays high."""
    idx    = pd.date_range("2024-01-01", periods=n, freq="D")
    closes = [start + i * step for i in range(n)]
    return pd.DataFrame({
        "open":   [c - 0.1 for c in closes],
        "high":   [c + 0.5 for c in closes],
        "low":    [c - 0.5 for c in closes],
        "close":  closes,
        "volume": 1_000_000,
    }, index=idx)


def _falling_df(start: float = 300.0, step: float = 1.0, n: int = 60) -> pd.DataFrame:
    """Linearly falling close prices — RSI stays low."""
    idx    = pd.date_range("2024-01-01", periods=n, freq="D")
    closes = [start - i * step for i in range(n)]
    return pd.DataFrame({
        "open":   [c + 0.1 for c in closes],
        "high":   [c + 0.5 for c in closes],
        "low":    [c - 0.5 for c in closes],
        "close":  closes,
        "volume": 1_000_000,
    }, index=idx)


# ── EMA ───────────────────────────────────────────────────────────────────────

def test_ema_constant_series_equals_price():
    """EMA of a constant series must converge to that constant."""
    df     = _const_df(price=150.0, n=60)
    result = ema(df, period=8)
    # After enough bars the EMA is equal to the constant (within float tolerance)
    assert abs(result.iloc[-1] - 150.0) < 1e-6


def test_ema_rising_lags_price():
    """On a rising series the EMA should be below the latest close."""
    df = _rising_df(n=100)
    for p in [8, 21, 50]:
        result = ema(df, period=p)
        assert result.iloc[-1] < df["close"].iloc[-1], f"EMA({p}) should lag rising close"


def test_ema_longer_period_lags_more():
    """A longer-period EMA lags more (is lower on a rising series)."""
    df    = _rising_df(n=100)
    ema8  = ema(df, period=8).iloc[-1]
    ema21 = ema(df, period=21).iloc[-1]
    ema50 = ema(df, period=50).iloc[-1]
    assert ema8 > ema21 > ema50


def test_ema_returns_named_series():
    result = ema(_const_df(), period=21)
    assert result.name == "ema_21"


# ── RSI ───────────────────────────────────────────────────────────────────────

def test_rsi_all_gains_near_100():
    """When every day is an up day, RSI should approach 100."""
    df = _rising_df(n=60)
    result = rsi(df, period=14)
    assert result.iloc[-1] > 90, f"Expected RSI > 90 on rising series, got {result.iloc[-1]:.2f}"


def test_rsi_all_losses_near_0():
    """When every day is a down day, RSI should approach 0."""
    df = _falling_df(n=60)
    result = rsi(df, period=14)
    assert result.iloc[-1] < 10, f"Expected RSI < 10 on falling series, got {result.iloc[-1]:.2f}"


def test_rsi_range_0_to_100():
    """RSI must always be between 0 and 100."""
    np.random.seed(99)
    closes = 100 + np.cumsum(np.random.randn(200))
    df = pd.DataFrame({"close": closes, "open": closes, "high": closes + 0.5,
                        "low": closes - 0.5, "volume": 1_000_000},
                       index=pd.date_range("2024-01-01", periods=200))
    result = rsi(df, period=14).dropna()
    assert (result >= 0).all() and (result <= 100).all()


def test_rsi_returns_named_series():
    result = rsi(_rising_df(), period=14)
    assert result.name == "rsi_14"


# ── MACD ──────────────────────────────────────────────────────────────────────

def test_macd_constant_series_is_zero():
    """On a constant series all EMAs are equal, so MACD and histogram are 0."""
    df     = _const_df(n=60)
    result = macd(df)
    assert abs(result["macd"].iloc[-1])      < 1e-6
    assert abs(result["histogram"].iloc[-1]) < 1e-6


def test_macd_rising_series_positive():
    """Fast EMA > slow EMA on a rising series → positive MACD."""
    df     = _rising_df(n=100)
    result = macd(df)
    assert result["macd"].iloc[-1] > 0


def test_macd_columns():
    result = macd(_rising_df())
    assert set(result.columns) == {"macd", "signal", "histogram"}


def test_macd_histogram_equals_macd_minus_signal():
    """Histogram must always equal MACD line minus signal line."""
    df     = _rising_df(n=100)
    result = macd(df).dropna()
    diff   = (result["macd"] - result["signal"] - result["histogram"]).abs()
    assert diff.max() < 1e-9


# ── Bollinger Bands ───────────────────────────────────────────────────────────

def test_bb_constant_series_bands_equal():
    """On a constant series std=0, so upper == middle == lower."""
    df     = _const_df(n=60)
    result = bollinger_bands(df, period=20)
    last   = result.iloc[-1]
    assert abs(last["bb_upper"]  - last["bb_middle"]) < 1e-6
    assert abs(last["bb_lower"]  - last["bb_middle"]) < 1e-6


def test_bb_upper_above_lower():
    """Upper band must always be above lower band on a non-constant series."""
    df     = _rising_df(n=100)
    result = bollinger_bands(df).dropna()
    assert (result["bb_upper"] > result["bb_lower"]).all()


def test_bb_middle_is_rolling_mean():
    """Middle band is a 20-period simple moving average."""
    df     = _rising_df(n=100)
    result = bollinger_bands(df, period=20)
    manual = df["close"].rolling(20).mean()
    diff   = (result["bb_middle"] - manual).dropna().abs()
    assert diff.max() < 1e-9


def test_bb_columns():
    result = bollinger_bands(_rising_df())
    assert {"bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_pct"}.issubset(result.columns)


# ── ATR ───────────────────────────────────────────────────────────────────────

def test_atr_constant_ohlc_is_zero():
    """On a flat constant series True Range is 0 → ATR converges to 0."""
    df     = _const_df(n=60)
    result = atr(df, period=14)
    assert abs(result.iloc[-1]) < 1e-6


def test_atr_positive_on_volatile_series():
    """ATR must be positive when there is price movement."""
    df     = _rising_df(n=60)
    result = atr(df, period=14)
    assert result.iloc[-1] > 0


def test_atr_returns_named_series():
    result = atr(_rising_df())
    assert result.name == "atr_14"


# ── VWAP ─────────────────────────────────────────────────────────────────────

def test_vwap_constant_equals_price():
    """VWAP of a constant series equals that constant price."""
    df     = _const_df(price=200.0, n=40)
    result = vwap(df, period=20)
    assert abs(result.iloc[-1] - 200.0) < 1e-6


def test_vwap_positive_on_real_series():
    df     = _rising_df(n=60)
    result = vwap(df, period=20).dropna()
    assert (result > 0).all()


def test_vwap_returns_named_series():
    result = vwap(_const_df())
    assert result.name == "vwap"


# ── ADX ───────────────────────────────────────────────────────────────────────

def test_adx_strongly_trending_above_25():
    """A strongly trending series should produce ADX > 25."""
    df     = _rising_df(start=100, step=2.0, n=150)  # steep trend
    result = adx(df, period=14)
    assert result["adx"].iloc[-1] > 25, \
        f"Expected ADX > 25 on strong trend, got {result['adx'].iloc[-1]:.2f}"


def test_adx_range_0_to_100():
    """ADX is always between 0 and 100."""
    df     = _rising_df(n=150)
    result = adx(df).dropna()
    assert (result["adx"] >= 0).all() and (result["adx"] <= 100).all()


def test_adx_plus_positive_on_uptrend():
    """+DI (bullish) should exceed -DI (bearish) on an uptrend."""
    df     = _rising_df(n=100)
    result = adx(df, period=14).dropna()
    # For the bulk of the uptrend, +DI > -DI
    assert (result["adx_plus"] > result["adx_minus"]).mean() > 0.8


def test_adx_columns():
    result = adx(_rising_df())
    assert set(result.columns) == {"adx", "adx_plus", "adx_minus"}


# ── run_all ───────────────────────────────────────────────────────────────────

def test_run_all_returns_all_columns():
    df     = _rising_df(n=250)
    result = run_all(df)
    expected = {
        "ema_8", "ema_21", "ema_50", "ema_200",
        "rsi_14", "macd", "macd_sig", "macd_hist",
        "bb_upper", "bb_middle", "bb_lower",
        "atr_14", "vwap", "adx",
    }
    assert expected.issubset(result.columns)


def test_run_all_no_nans_in_output():
    """After warmup rows are dropped, no indicator column should have NaN."""
    df     = _rising_df(n=250)
    result = run_all(df)
    indicator_cols = [c for c in result.columns if c not in ("open", "high", "low", "close", "volume")]
    assert result[indicator_cols].isna().sum().sum() == 0, "Unexpected NaNs in indicator output"


def test_run_all_raises_on_empty_df():
    with pytest.raises(ValueError, match="empty"):
        run_all(pd.DataFrame())


def test_run_all_raises_on_missing_columns():
    df = pd.DataFrame({"close": [1, 2, 3]})
    with pytest.raises(ValueError, match="missing columns"):
        run_all(df)
