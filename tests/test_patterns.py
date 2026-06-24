"""
Tests for the pattern recognition module.

Each test constructs the *minimum* price series that should trigger the
pattern and asserts that (a) at least one result is returned, (b) the
direction is correct, and (c) confidence is in [0, 1].
"""
import numpy as np
import pandas as pd
import pytest

from patterns.models import PatternResult
from patterns.candlestick import (
    scan_doji, scan_hammer, scan_shooting_star, scan_pin_bar,
    scan_bullish_engulfing, scan_bearish_engulfing,
    scan_morning_star, scan_evening_star,
)
from patterns.chart import (
    scan_double_top, scan_double_bottom,
    scan_head_and_shoulders, scan_inv_head_and_shoulders,
    scan_ascending_triangle, scan_descending_triangle,
    scan_symmetrical_triangle,
    scan_trendline_break_up, scan_trendline_break_down,
)
from patterns.detector import scan_all, scan_candlestick, scan_chart


# ── DataFrame builder helpers ─────────────────────────────────────────────────

def _make_df(rows: list[dict], start: str = "2024-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(rows), freq="D")
    return pd.DataFrame(rows, index=idx)


def _candle(o, h, l, c, v=1_000_000):
    return {"open": o, "high": h, "low": l, "close": c, "volume": v}


def _flat_series(n: int = 50, base: float = 100.0) -> pd.DataFrame:
    """Steady price series — no pattern should fire."""
    rows = [_candle(base, base + 0.5, base - 0.5, base) for _ in range(n)]
    return _make_df(rows)


# ── Shared assertion ──────────────────────────────────────────────────────────

def _assert_pattern(results, pattern_name, direction):
    matches = [r for r in results if r.pattern == pattern_name]
    assert matches, f"No '{pattern_name}' detected"
    for r in matches:
        assert r.direction == direction, f"Expected direction={direction!r}, got {r.direction!r}"
        assert 0.0 <= r.confidence <= 1.0, f"Confidence {r.confidence} out of [0,1]"
        assert isinstance(r.timestamp, pd.Timestamp)
        assert r.bar_index >= 0


# ── PatternResult dataclass ───────────────────────────────────────────────────

class TestPatternResult:
    def test_repr(self):
        ts = pd.Timestamp("2024-06-01")
        r = PatternResult("doji", "neutral", 0.85, ts, 5)
        assert "doji" in repr(r)
        assert "0.85" in repr(r)

    def test_defaults(self):
        ts = pd.Timestamp("2024-06-01")
        r = PatternResult("doji", "neutral", 0.85, ts, 5)
        assert r.start_index == -1
        assert r.notes == ""


# ── Doji ─────────────────────────────────────────────────────────────────────

class TestDoji:
    def _doji_df(self):
        # Body = 0.01, range = 2.0 → ratio 0.005 < 0.10
        rows = [_candle(100.00, 101.0, 99.0, 100.01)]
        return _make_df(rows)

    def test_detects_doji(self):
        results = scan_doji(self._doji_df())
        _assert_pattern(results, "doji", "neutral")

    def test_no_doji_on_large_body(self):
        # Body = 3.0, range = 4.0 → ratio 0.75 > 0.10
        df = _make_df([_candle(98.0, 102.0, 98.0, 101.0)])
        assert scan_doji(df) == []

    def test_zero_range_skipped(self):
        df = _make_df([_candle(100.0, 100.0, 100.0, 100.0)])
        assert scan_doji(df) == []

    def test_confidence_between_0_and_1(self):
        results = scan_doji(self._doji_df())
        assert all(0.0 <= r.confidence <= 1.0 for r in results)


# ── Hammer ────────────────────────────────────────────────────────────────────

class TestHammer:
    def _hammer_df(self):
        # Small body at top: o=100, c=100.5 (body=0.5), lower wick=5, upper wick=0
        # range=5.5, body/range=0.09 < 0.35, lower=5 >= 2*0.5=1 ✓
        rows = [_candle(100.0, 100.5, 95.0, 100.5)]
        return _make_df(rows)

    def test_detects_hammer(self):
        results = scan_hammer(self._hammer_df())
        _assert_pattern(results, "hammer", "bullish")

    def test_no_hammer_large_body(self):
        df = _make_df([_candle(100.0, 102.0, 98.0, 102.0)])
        assert scan_hammer(df) == []

    def test_no_hammer_short_wick(self):
        # lower wick = 0.1, body = 0.5 → wick < 2×body
        df = _make_df([_candle(100.0, 100.5, 99.9, 100.5)])
        assert scan_hammer(df) == []


# ── Shooting Star ─────────────────────────────────────────────────────────────

class TestShootingStar:
    def _ss_df(self):
        # Small body at bottom: o=100, c=100.5 (body=0.5), upper wick=5
        # range=5.5, lower wick=0
        rows = [_candle(100.0, 105.5, 100.0, 100.5)]
        return _make_df(rows)

    def test_detects_shooting_star(self):
        results = scan_shooting_star(self._ss_df())
        _assert_pattern(results, "shooting_star", "bearish")

    def test_no_ss_short_upper_wick(self):
        df = _make_df([_candle(100.0, 100.8, 99.5, 100.5)])
        assert scan_shooting_star(df) == []


# ── Pin Bar ───────────────────────────────────────────────────────────────────

class TestPinBar:
    def test_bullish_pin_bar(self):
        # Long lower wick: lower=6, body=0.5 → 6 >= 3*0.5 ✓, lower > upper
        df = _make_df([_candle(100.0, 100.5, 94.0, 100.5)])
        results = scan_pin_bar(df)
        _assert_pattern(results, "pin_bar", "bullish")

    def test_bearish_pin_bar(self):
        # Long upper wick: upper=6, body=0.5
        df = _make_df([_candle(100.0, 106.5, 100.0, 100.5)])
        results = scan_pin_bar(df)
        _assert_pattern(results, "pin_bar", "bearish")

    def test_no_pin_bar_equal_wicks(self):
        # Small wicks both sides
        df = _make_df([_candle(100.0, 101.0, 99.0, 100.5)])
        assert scan_pin_bar(df) == []


# ── Bullish Engulfing ─────────────────────────────────────────────────────────

class TestBullishEngulfing:
    def _df(self):
        rows = [
            _candle(102.0, 102.5, 99.5, 100.0),  # bearish: o=102 > c=100
            _candle(99.0,  103.5, 98.5, 103.5),  # bullish: engulfs prev body
        ]
        return _make_df(rows)

    def test_detects(self):
        results = scan_bullish_engulfing(self._df())
        _assert_pattern(results, "bullish_engulfing", "bullish")

    def test_no_detection_if_not_engulfing(self):
        rows = [
            _candle(102.0, 102.5, 99.5, 100.0),
            _candle(100.5, 102.5, 99.5, 101.5),  # bullish but doesn't engulf
        ]
        df = _make_df(rows)
        assert scan_bullish_engulfing(df) == []


# ── Bearish Engulfing ─────────────────────────────────────────────────────────

class TestBearishEngulfing:
    def _df(self):
        rows = [
            _candle(100.0, 103.5, 99.5, 103.0),  # bullish
            _candle(104.0, 104.5, 98.5, 99.0),   # bearish, engulfs
        ]
        return _make_df(rows)

    def test_detects(self):
        results = scan_bearish_engulfing(self._df())
        _assert_pattern(results, "bearish_engulfing", "bearish")

    def test_no_detection_on_single_candle(self):
        df = _make_df([_candle(100.0, 103.0, 99.0, 102.0)])
        assert scan_bearish_engulfing(df) == []


# ── Morning Star ──────────────────────────────────────────────────────────────

class TestMorningStar:
    def _df(self):
        # Day1: large bearish (range=10, body=8)
        # Day2: tiny body star
        # Day3: large bullish closing above midpoint of Day1
        rows = [
            _candle(110.0, 110.5, 100.0, 102.0),  # bearish: mid=(110+102)/2=106
            _candle(101.5, 102.0, 100.5, 101.8),  # tiny star
            _candle(102.0, 112.0, 101.5, 109.0),  # bullish, close=109 > 106
        ]
        return _make_df(rows)

    def test_detects(self):
        results = scan_morning_star(self._df())
        _assert_pattern(results, "morning_star", "bullish")

    def test_minimum_3_bars_required(self):
        df = _make_df([_candle(110.0, 110.5, 100.0, 102.0),
                       _candle(102.0, 112.0, 101.5, 109.0)])
        assert scan_morning_star(df) == []


# ── Evening Star ──────────────────────────────────────────────────────────────

class TestEveningStar:
    def _df(self):
        # Day1: large bullish (o=100, c=108, mid=104)
        # Day2: tiny star
        # Day3: large bearish closing below midpoint of Day1
        rows = [
            _candle(100.0, 108.5, 99.5, 108.0),  # bullish, mid=(100+108)/2=104
            _candle(108.5, 109.0, 107.5, 108.2),  # tiny star
            _candle(108.0, 108.5, 97.5, 100.0),  # bearish, close=100 < 104
        ]
        return _make_df(rows)

    def test_detects(self):
        results = scan_evening_star(self._df())
        _assert_pattern(results, "evening_star", "bearish")


# ── Double Top ────────────────────────────────────────────────────────────────

class TestDoubleTop:
    def _df(self):
        """Two peaks ~105 separated by a valley ~95."""
        prices = (
            [100.0] * 5 +
            [102.0, 104.0, 105.0, 104.0, 102.0] +   # peak 1 at index 7
            [100.0, 98.0, 95.0, 98.0, 100.0] +       # valley
            [102.0, 104.0, 105.1, 104.0, 102.0] +   # peak 2 at index 17
            [100.0] * 3
        )
        rows = [_candle(p - 0.3, p + 0.5, p - 0.5, p) for p in prices]
        return _make_df(rows)

    def test_detects(self):
        df = self._df()
        results = scan_double_top(df, peak_order=3, min_gap=5, max_gap=40)
        _assert_pattern(results, "double_top", "bearish")

    def test_confidence_valid(self):
        df = self._df()
        results = scan_double_top(df, peak_order=3, min_gap=5, max_gap=40)
        assert all(0.0 <= r.confidence <= 1.0 for r in results)


# ── Double Bottom ─────────────────────────────────────────────────────────────

class TestDoubleBottom:
    def _df(self):
        prices = (
            [100.0] * 5 +
            [98.0, 96.0, 95.0, 96.0, 98.0] +   # trough 1
            [100.0, 102.0, 105.0, 102.0, 100.0] +
            [98.0, 96.0, 95.1, 96.0, 98.0] +   # trough 2
            [100.0] * 3
        )
        rows = [_candle(p - 0.3, p + 0.5, p - 0.5, p) for p in prices]
        return _make_df(rows)

    def test_detects(self):
        df = self._df()
        results = scan_double_bottom(df, trough_order=3, min_gap=5, max_gap=40)
        _assert_pattern(results, "double_bottom", "bullish")


# ── Head and Shoulders ────────────────────────────────────────────────────────

class TestHeadAndShoulders:
    def _df(self):
        # L shoulder ~104, Head ~110, R shoulder ~104
        prices = (
            [100.0] * 3 +
            [102.0, 103.0, 104.0, 103.0, 102.0] +   # left shoulder (peak ~104)
            [100.0, 101.0] +
            [105.0, 108.0, 110.0, 108.0, 105.0] +   # head (peak ~110)
            [101.0, 100.0] +
            [102.0, 103.0, 104.1, 103.0, 102.0] +   # right shoulder (~104)
            [100.0] * 3
        )
        rows = [_candle(p - 0.3, p + 0.5, p - 0.5, p) for p in prices]
        return _make_df(rows)

    def test_detects(self):
        df = self._df()
        results = scan_head_and_shoulders(df, peak_order=3, min_gap=3, max_gap=15)
        _assert_pattern(results, "head_and_shoulders", "bearish")


# ── Inverse Head and Shoulders ────────────────────────────────────────────────

class TestInvHeadAndShoulders:
    def _df(self):
        prices = (
            [100.0] * 3 +
            [98.0, 97.0, 96.0, 97.0, 98.0] +    # left shoulder
            [100.0, 99.0] +
            [95.0, 92.0, 90.0, 92.0, 95.0] +    # head
            [99.0, 100.0] +
            [98.0, 97.0, 95.9, 97.0, 98.0] +    # right shoulder
            [100.0] * 3
        )
        rows = [_candle(p - 0.3, p + 0.5, p - 0.5, p) for p in prices]
        return _make_df(rows)

    def test_detects(self):
        df = self._df()
        results = scan_inv_head_and_shoulders(df, trough_order=3, min_gap=3, max_gap=15)
        _assert_pattern(results, "inv_head_and_shoulders", "bullish")


# ── Triangles ─────────────────────────────────────────────────────────────────

def _triangle_df(n: int = 50, resist_slope: float = 0.0, support_slope: float = 0.0,
                 base: float = 100.0, osc_amp: float = 1.5, period: int = 4) -> pd.DataFrame:
    """
    Build a DataFrame where highs trend along resist_slope and lows along support_slope,
    with a sine-wave oscillation so that _find_peaks / _find_troughs can locate actual
    local extrema within the window.
    """
    x = np.arange(n, dtype=float)
    wave   = osc_amp * np.sin(2 * np.pi * x / period)
    highs  = base + 2.0 + resist_slope * x + wave
    lows   = base - 2.0 + support_slope * x - wave
    opens  = (highs + lows) / 2 - 0.2
    closes = (highs + lows) / 2 + 0.2
    volume = np.full(n, 1_000_000.0)
    idx    = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volume},
        index=idx,
    )


class TestAscendingTriangle:
    def test_detects(self):
        # Flat resistance (slope=0), rising support — oscillation creates detectable peaks/troughs
        df = _triangle_df(n=50, resist_slope=0.0, support_slope=0.1, base=100.0)
        results = scan_ascending_triangle(df, window=30, flat_slope_pct=0.002)
        assert len(results) > 0
        _assert_pattern(results, "ascending_triangle", "bullish")


class TestDescendingTriangle:
    def test_detects(self):
        # Falling resistance, flat support
        df = _triangle_df(n=50, resist_slope=-0.1, support_slope=0.0, base=100.0)
        results = scan_descending_triangle(df, window=30, flat_slope_pct=0.002)
        assert len(results) > 0
        _assert_pattern(results, "descending_triangle", "bearish")


class TestSymmetricalTriangle:
    def test_detects(self):
        # Falling resistance, rising support — equal magnitude slopes → symmetrical
        df = _triangle_df(n=50, resist_slope=-0.1, support_slope=0.1, base=100.0)
        results = scan_symmetrical_triangle(df, window=30, flat_slope_pct=0.001)
        assert len(results) > 0
        _assert_pattern(results, "symmetrical_triangle", "neutral")


# ── Trendline Breaks ──────────────────────────────────────────────────────────

class TestTrendlineBreakUp:
    def _df(self):
        n = 45
        x = np.arange(n, dtype=float)
        # Descending highs WITH oscillation so _find_peaks can detect peaks along the downtrend
        wave  = 1.5 * np.sin(2 * np.pi * x / 4)
        highs  = 110.0 - 0.2 * x + wave
        closes = highs - 3.5          # well below highs throughout
        opens  = closes - 0.5
        lows   = closes - 1.5
        # Decisive breakout above the falling trendline at bar 30
        closes[30] = 120.0
        opens[30]  = 110.0
        highs[30]  = 121.0
        lows[30]   = 109.0
        volume = np.full(n, 1_000_000.0)
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        return pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volume},
            index=idx,
        )

    def test_detects(self):
        df = self._df()
        results = scan_trendline_break_up(df, lookback=20)
        _assert_pattern(results, "trendline_break_up", "bullish")


class TestTrendlineBreakDown:
    def _df(self):
        n = 45
        x = np.arange(n, dtype=float)
        # Rising lows WITH oscillation so _find_troughs can detect troughs along the uptrend
        wave  = 1.5 * np.sin(2 * np.pi * x / 4)
        lows   = 90.0 + 0.2 * x - wave   # troughs where sin=+1
        closes = lows + 3.5               # well above lows throughout
        opens  = closes + 0.5
        highs  = closes + 1.5
        # Decisive breakdown below the rising trendline at bar 30
        closes[30] = 75.0
        opens[30]  = 92.0
        lows[30]   = 74.0
        highs[30]  = 93.0
        volume = np.full(n, 1_000_000.0)
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        return pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volume},
            index=idx,
        )

    def test_detects(self):
        df = self._df()
        results = scan_trendline_break_down(df, lookback=20)
        _assert_pattern(results, "trendline_break_down", "bearish")


# ── Detector (scan_all / scan_candlestick / scan_chart) ───────────────────────

class TestDetector:
    def _mixed_df(self):
        """A series that contains a known doji and a bullish engulfing."""
        rows = [
            _candle(102.0, 102.5, 99.5, 100.0),  # bearish
            _candle(99.0,  103.5, 98.5, 103.5),  # bullish engulfing
            _candle(103.4, 104.0, 103.0, 103.45), # doji (body≈0.05, range=1)
        ]
        return _make_df(rows)

    def test_scan_all_returns_list(self):
        df = self._mixed_df()
        results = scan_all(df)
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, PatternResult)

    def test_scan_all_sorted_by_bar_index(self):
        df = self._mixed_df()
        results = scan_all(df)
        if len(results) >= 2:
            indices = [r.bar_index for r in results]
            assert indices == sorted(indices)

    def test_scan_all_empty_df(self):
        empty = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        assert scan_all(empty) == []

    def test_scan_candlestick_only_candlestick_patterns(self):
        df = self._mixed_df()
        results = scan_candlestick(df)
        chart_names = {
            "double_top", "double_bottom", "head_and_shoulders",
            "inv_head_and_shoulders", "ascending_triangle",
            "descending_triangle", "symmetrical_triangle",
            "trendline_break_up", "trendline_break_down",
        }
        for r in results:
            assert r.pattern not in chart_names

    def test_scan_chart_only_chart_patterns(self):
        df = self._mixed_df()
        results = scan_chart(df)
        candle_names = {
            "doji", "hammer", "shooting_star", "pin_bar",
            "bullish_engulfing", "bearish_engulfing",
            "morning_star", "evening_star",
        }
        for r in results:
            assert r.pattern not in candle_names

    def test_all_confidences_in_range(self):
        """Run on a synthetic 200-bar random walk — no confidence out of [0,1]."""
        rng = np.random.default_rng(0)
        n = 200
        close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
        rng2 = np.random.default_rng(1)
        spread = close * rng2.uniform(0.003, 0.008, n)
        high   = close + spread * rng2.uniform(0.3, 0.7, n)
        low    = close - spread * rng2.uniform(0.3, 0.7, n)
        open_  = low + (high - low) * rng2.uniform(0, 1, n)
        volume = np.full(n, 1_000_000.0)
        idx    = pd.date_range("2024-01-01", periods=n, freq="D")
        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=idx,
        )
        results = scan_all(df)
        for r in results:
            assert 0.0 <= r.confidence <= 1.0, (
                f"{r.pattern} returned confidence={r.confidence}"
            )
