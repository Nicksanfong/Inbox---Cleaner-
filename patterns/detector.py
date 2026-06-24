"""
Master pattern scanner — runs all candlestick and chart patterns on one call.
"""
import pandas as pd

from patterns.models import PatternResult
from patterns.candlestick import (
    scan_doji,
    scan_hammer,
    scan_shooting_star,
    scan_pin_bar,
    scan_bullish_engulfing,
    scan_bearish_engulfing,
    scan_morning_star,
    scan_evening_star,
)
from patterns.chart import (
    scan_double_top,
    scan_double_bottom,
    scan_head_and_shoulders,
    scan_inv_head_and_shoulders,
    scan_ascending_triangle,
    scan_descending_triangle,
    scan_symmetrical_triangle,
    scan_trendline_break_up,
    scan_trendline_break_down,
)

_CANDLESTICK_SCANNERS = [
    scan_doji,
    scan_hammer,
    scan_shooting_star,
    scan_pin_bar,
    scan_bullish_engulfing,
    scan_bearish_engulfing,
    scan_morning_star,
    scan_evening_star,
]

_CHART_SCANNERS = [
    scan_double_top,
    scan_double_bottom,
    scan_head_and_shoulders,
    scan_inv_head_and_shoulders,
    scan_ascending_triangle,
    scan_descending_triangle,
    scan_symmetrical_triangle,
    scan_trendline_break_up,
    scan_trendline_break_down,
]


def scan_all(df: pd.DataFrame) -> list[PatternResult]:
    """
    Run every pattern scanner on *df* and return a combined list sorted by
    bar_index ascending (then by confidence descending for ties).

    Input df must have columns: open, high, low, close, volume
    with a DatetimeIndex.
    """
    if df.empty:
        return []

    results: list[PatternResult] = []
    for scanner in _CANDLESTICK_SCANNERS + _CHART_SCANNERS:
        try:
            results.extend(scanner(df))
        except Exception:
            pass

    results.sort(key=lambda r: (r.bar_index, -r.confidence))
    return results


def scan_candlestick(df: pd.DataFrame) -> list[PatternResult]:
    """Run only the single- and multi-candle patterns."""
    results: list[PatternResult] = []
    for scanner in _CANDLESTICK_SCANNERS:
        results.extend(scanner(df))
    results.sort(key=lambda r: (r.bar_index, -r.confidence))
    return results


def scan_chart(df: pd.DataFrame) -> list[PatternResult]:
    """Run only the chart / structural patterns."""
    results: list[PatternResult] = []
    for scanner in _CHART_SCANNERS:
        results.extend(scanner(df))
    results.sort(key=lambda r: (r.bar_index, -r.confidence))
    return results
