"""
Tests for signals/scoring.py and signals/engine.py
"""
import numpy as np
import pandas as pd
import pytest

from signals.scoring import score_bar
from signals.engine import SignalEngine
from signals.models import Confirmation, Signal


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_row(**kwargs) -> pd.Series:
    """Build a minimal indicator row with sensible defaults."""
    defaults = dict(
        close=100.0,
        ema_8=100.0, ema_21=100.0, ema_50=100.0, ema_200=100.0,
        macd=0.0, macd_sig=0.0, macd_hist=0.0,
        adx=15.0, adx_plus=10.0, adx_minus=10.0,
        rsi_14=50.0,
        bb_pct=0.5, bb_upper=105.0, bb_lower=95.0,
        vwap=100.0,
    )
    defaults.update(kwargs)
    return pd.Series(defaults)


def _bull_row() -> pd.Series:
    """Strong uptrend row — should produce a high bullish score."""
    return _make_row(
        close=115.0,
        ema_8=113.0, ema_21=110.0, ema_50=105.0, ema_200=100.0,
        macd=0.5, macd_sig=0.3, macd_hist=0.2,
        adx=30.0, adx_plus=28.0, adx_minus=12.0,
        rsi_14=62.0,
        bb_pct=0.6, bb_upper=120.0, bb_lower=100.0,
        vwap=112.0,
    )


def _bear_row() -> pd.Series:
    """Strong downtrend row — should produce a high bearish score."""
    return _make_row(
        close=85.0,
        ema_8=87.0, ema_21=90.0, ema_50=95.0, ema_200=100.0,
        macd=-0.5, macd_sig=-0.3, macd_hist=-0.2,
        adx=30.0, adx_plus=12.0, adx_minus=28.0,
        rsi_14=38.0,
        bb_pct=0.4, bb_upper=100.0, bb_lower=80.0,
        vwap=88.0,
    )


def _make_ohlcv(n: int = 100, trend: float = 0.1, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    close = 100.0 + trend * np.arange(n) + rng.normal(0, 0.5, n).cumsum()
    high   = close + rng.uniform(0.2, 1.0, n)
    low    = close - rng.uniform(0.2, 1.0, n)
    open_  = close - rng.normal(0, 0.3, n)
    volume = rng.integers(1000, 5000, n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                          "close": close, "volume": volume}, index=idx)


# ── score_bar ─────────────────────────────────────────────────────────────────

class TestScoreBarBasic:
    def test_returns_required_keys(self):
        row = _make_row()
        result = score_bar(row, [])
        assert set(result.keys()) >= {"bullish", "bearish",
                                       "bull_confirmations", "bear_confirmations",
                                       "breakdown"}

    def test_scores_in_range(self):
        for row in (_bull_row(), _bear_row(), _make_row()):
            r = score_bar(row, [])
            assert 0.0 <= r["bullish"] <= 100.0
            assert 0.0 <= r["bearish"] <= 100.0

    def test_confirmations_are_confirmation_objects(self):
        r = score_bar(_bull_row(), [])
        for c in r["bull_confirmations"] + r["bear_confirmations"]:
            assert isinstance(c, Confirmation)

    def test_breakdown_has_all_categories(self):
        r = score_bar(_bull_row(), [])
        assert set(r["breakdown"].keys()) >= {"Trend", "Momentum", "Volatility", "Patterns"}


class TestScoreBarBullish:
    def test_bull_row_scores_higher_bullish_than_bearish(self):
        r = score_bar(_bull_row(), [])
        assert r["bullish"] > r["bearish"]

    def test_full_ema_stack_fires(self):
        r = score_bar(_bull_row(), [])
        sources = {c.source for c in r["bull_confirmations"]}
        assert "full_ema_stack" in sources

    def test_above_ema_200_fires(self):
        r = score_bar(_bull_row(), [])
        sources = {c.source for c in r["bull_confirmations"]}
        assert "above_ema_200" in sources

    def test_macd_bull_cross_fires(self):
        r = score_bar(_bull_row(), [])
        sources = {c.source for c in r["bull_confirmations"]}
        assert "macd_bull_cross" in sources

    def test_strong_adx_fires_bullish(self):
        row = _make_row(adx=30.0, adx_plus=25.0, adx_minus=10.0)
        r = score_bar(row, [])
        sources = {c.source for c in r["bull_confirmations"]}
        assert "adx_strong" in sources

    def test_rsi_oversold_fires(self):
        row = _make_row(rsi_14=25.0)
        r = score_bar(row, [])
        sources = {c.source for c in r["bull_confirmations"]}
        assert "rsi_oversold" in sources

    def test_rsi_recovering_fires(self):
        row = _make_row(rsi_14=38.0)
        r = score_bar(row, [])
        sources = {c.source for c in r["bull_confirmations"]}
        assert "rsi_recovering" in sources

    def test_at_bb_lower_fires(self):
        row = _make_row(bb_pct=0.10)
        r = score_bar(row, [])
        sources = {c.source for c in r["bull_confirmations"]}
        assert "at_bb_lower" in sources

    def test_above_vwap_fires(self):
        row = _make_row(close=101.0, vwap=100.0)
        r = score_bar(row, [])
        sources = {c.source for c in r["bull_confirmations"]}
        assert "above_vwap" in sources


class TestScoreBarBearish:
    def test_bear_row_scores_higher_bearish_than_bullish(self):
        r = score_bar(_bear_row(), [])
        assert r["bearish"] > r["bullish"]

    def test_full_ema_stack_bear_fires(self):
        r = score_bar(_bear_row(), [])
        sources = {c.source for c in r["bear_confirmations"]}
        assert "full_ema_stack" in sources

    def test_rsi_overbought_fires(self):
        row = _make_row(rsi_14=75.0)
        r = score_bar(row, [])
        sources = {c.source for c in r["bear_confirmations"]}
        assert "rsi_overbought" in sources

    def test_rsi_fading_fires(self):
        row = _make_row(rsi_14=62.0)
        r = score_bar(row, [])
        sources = {c.source for c in r["bear_confirmations"]}
        assert "rsi_fading" in sources

    def test_at_bb_upper_fires(self):
        row = _make_row(bb_pct=0.90)
        r = score_bar(row, [])
        sources = {c.source for c in r["bear_confirmations"]}
        assert "at_bb_upper" in sources

    def test_below_vwap_fires(self):
        row = _make_row(close=99.0, vwap=100.0)
        r = score_bar(row, [])
        sources = {c.source for c in r["bear_confirmations"]}
        assert "below_vwap" in sources


class TestScoreBarNeutral:
    def test_neutral_row_both_sides_score(self):
        # Mix: EMAs bullish, MACD bearish, RSI neutral, BB/VWAP mixed
        row = _make_row(
            close=105.0,
            ema_8=104.0, ema_21=103.0, ema_50=102.0, ema_200=101.0,  # bullish stack
            macd=-0.2, macd_sig=0.1, macd_hist=-0.3,                  # bearish MACD
            rsi_14=50.0,                                               # neutral RSI
            bb_pct=0.5, vwap=106.0,                                    # bearish VWAP
        )
        r = score_bar(row, [])
        # Both sides should score; neither should be zero
        assert r["bullish"] > 0
        assert r["bearish"] > 0

    def test_adx_weak_does_not_fire_adx_strong(self):
        row = _make_row(adx=20.0)
        r = score_bar(row, [])
        all_sources = {c.source for c in r["bull_confirmations"] + r["bear_confirmations"]}
        assert "adx_strong" not in all_sources


class TestScoreBarBreakdown:
    def test_trend_points_positive_on_bull(self):
        r = score_bar(_bull_row(), [])
        assert r["breakdown"]["Trend"]["bullish"] > 0

    def test_breakdown_sums_are_nonnegative(self):
        r = score_bar(_bull_row(), [])
        for cat, vals in r["breakdown"].items():
            assert vals["bullish"] >= 0
            assert vals["bearish"] >= 0


# ── SignalEngine ───────────────────────────────────────────────────────────────

class TestSignalEngineInit:
    def test_defaults(self):
        eng = SignalEngine()
        assert eng.threshold == 75.0
        assert eng.min_confirmations == 3

    def test_custom_params(self):
        eng = SignalEngine(threshold=60.0, min_confirmations=2)
        assert eng.threshold == 60.0
        assert eng.min_confirmations == 2


class TestSignalEngineScan:
    def test_returns_list(self):
        eng = SignalEngine()
        df = _make_ohlcv(200, trend=0.1, seed=1)
        result = eng.scan({"15m": df})
        assert isinstance(result, list)

    def test_empty_frames_returns_empty(self):
        eng = SignalEngine()
        assert eng.scan({}) == []

    def test_signal_fields(self):
        eng = SignalEngine(threshold=0.0, min_confirmations=1)
        df = _make_ohlcv(200, trend=0.2, seed=2)
        signals = eng.scan({"15m": df, "1H": df, "D": df})
        if signals:
            sig = signals[0]
            assert isinstance(sig, Signal)
            assert sig.direction in ("BUY", "SELL")
            assert 0.0 <= sig.score <= 100.0
            assert sig.timeframe in ("15m", "1H", "D")
            assert len(sig.confirmations) >= 1

    def test_signals_sorted_descending(self):
        eng = SignalEngine(threshold=0.0, min_confirmations=1)
        df = _make_ohlcv(200, trend=0.2, seed=3)
        signals = eng.scan({"15m": df, "1H": df, "D": df})
        scores = [s.score for s in signals]
        assert scores == sorted(scores, reverse=True)

    def test_no_signal_below_threshold(self):
        eng = SignalEngine(threshold=99.0, min_confirmations=1)
        df = _make_ohlcv(200, trend=0.05, seed=4)
        signals = eng.scan({"15m": df, "1H": df, "D": df})
        for s in signals:
            assert s.score >= 99.0

    def test_buy_signal_on_strong_uptrend(self):
        eng = SignalEngine(threshold=50.0, min_confirmations=2)
        df = _make_ohlcv(300, trend=0.3, seed=5)
        signals = eng.scan({"15m": df, "1H": df, "D": df})
        directions = {s.direction for s in signals}
        assert "BUY" in directions

    def test_sell_signal_on_strong_downtrend(self):
        eng = SignalEngine(threshold=50.0, min_confirmations=2)
        df = _make_ohlcv(300, trend=-0.3, seed=6)
        signals = eng.scan({"15m": df, "1H": df, "D": df})
        directions = {s.direction for s in signals}
        assert "SELL" in directions

    def test_min_confirmations_enforced(self):
        eng = SignalEngine(threshold=0.0, min_confirmations=100)
        df = _make_ohlcv(200, trend=0.2, seed=7)
        signals = eng.scan({"15m": df, "1H": df, "D": df})
        assert signals == []

    def test_single_timeframe_works(self):
        eng = SignalEngine(threshold=0.0, min_confirmations=1)
        df = _make_ohlcv(200, trend=0.2, seed=8)
        signals = eng.scan({"1H": df})
        assert isinstance(signals, list)

    def test_missing_timeframe_skipped_gracefully(self):
        eng = SignalEngine(threshold=0.0, min_confirmations=1)
        df = _make_ohlcv(200, trend=0.2, seed=9)
        signals = eng.scan({"15m": df, "D": df})  # no 1H
        assert isinstance(signals, list)


class TestSignalEngineRawScores:
    def test_returns_expected_keys(self):
        eng = SignalEngine()
        df = _make_ohlcv(200, trend=0.1, seed=10)
        result = eng.raw_scores({"15m": df, "1H": df})
        assert set(result.keys()) >= {
            "by_tf", "combined_bull", "combined_bear",
            "mtf_bonus_bull", "mtf_bonus_bear",
        }

    def test_empty_frames_returns_empty(self):
        eng = SignalEngine()
        assert eng.raw_scores({}) == {}

    def test_by_tf_has_timeframe_entries(self):
        eng = SignalEngine()
        df = _make_ohlcv(200, trend=0.1, seed=11)
        result = eng.raw_scores({"15m": df, "D": df})
        assert "15m" in result["by_tf"]
        assert "D"   in result["by_tf"]

    def test_combined_scores_in_range(self):
        eng = SignalEngine()
        df = _make_ohlcv(200, trend=0.15, seed=12)
        result = eng.raw_scores({"15m": df, "1H": df, "D": df})
        assert 0.0 <= result["combined_bull"] <= 100.0
        assert 0.0 <= result["combined_bear"] <= 100.0

    def test_mtf_bonus_valid_values(self):
        eng = SignalEngine()
        df = _make_ohlcv(300, trend=0.3, seed=13)
        result = eng.raw_scores({"15m": df, "1H": df, "D": df})
        assert result["mtf_bonus_bull"] in (0.0, 6.0, 10.0)
        assert result["mtf_bonus_bear"] in (0.0, 6.0, 10.0)

    def test_strong_uptrend_gets_mtf_bonus(self):
        eng = SignalEngine()
        df = _make_ohlcv(500, trend=0.5, seed=14)
        result = eng.raw_scores({"15m": df, "1H": df, "D": df})
        assert result["mtf_bonus_bull"] > 0


class TestMtfBonus:
    """Isolated tests for the MTF bonus logic."""

    def test_all_three_agree_gives_10(self):
        eng = SignalEngine(threshold=0.0, min_confirmations=1)
        df_up = _make_ohlcv(500, trend=0.5, seed=20)
        result = eng.raw_scores({"15m": df_up, "1H": df_up, "D": df_up})
        assert result["mtf_bonus_bull"] == 10.0

    def test_single_timeframe_no_bonus(self):
        eng = SignalEngine()
        df = _make_ohlcv(200, trend=0.2, seed=21)
        result = eng.raw_scores({"1H": df})
        assert result["mtf_bonus_bull"] == 0.0
        assert result["mtf_bonus_bear"] == 0.0
