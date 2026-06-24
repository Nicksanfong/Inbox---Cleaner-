"""
Candlestick pattern detection.

Each scan_* function accepts an OHLCV DataFrame (DatetimeIndex) and
returns a list of PatternResult objects — one per detected occurrence.

Confidence is always in [0, 1]:
  1.0 = textbook-perfect pattern
  0.0 = detection threshold (anything below is not returned)
"""
import numpy as np
import pandas as pd

from patterns.models import PatternResult

# ── Internal geometry helpers ─────────────────────────────────────────────────

def _arrays(df: pd.DataFrame):
    return (
        df["open"].values.astype(float),
        df["high"].values.astype(float),
        df["low"].values.astype(float),
        df["close"].values.astype(float),
        df.index,
    )


def _body(o, c):       return abs(c - o)
def _upper_wick(o, h, c): return h - max(o, c)
def _lower_wick(o, l, c): return min(o, c) - l
def _rng(h, l):        return h - l


# ── Single-candle patterns ────────────────────────────────────────────────────

def scan_doji(df: pd.DataFrame, body_pct: float = 0.10) -> list[PatternResult]:
    """
    Doji — open ≈ close (body < body_pct × total range).
    Signals indecision; more significant after a trend.
    """
    opens, highs, lows, closes, idx = _arrays(df)
    results = []
    for i in range(len(df)):
        r = _rng(highs[i], lows[i])
        if r == 0:
            continue
        ratio = _body(opens[i], closes[i]) / r
        if ratio < body_pct:
            conf = round(1.0 - ratio / body_pct, 4)
            results.append(PatternResult(
                pattern="doji", direction="neutral", confidence=conf,
                timestamp=idx[i], bar_index=i,
                notes=f"body={ratio:.3f}×range",
            ))
    return results


def scan_hammer(df: pd.DataFrame, body_pct: float = 0.35,
                shadow_mult: float = 2.0) -> list[PatternResult]:
    """
    Hammer — small body at top of range, long lower wick (≥ shadow_mult × body),
    tiny upper wick. Bullish reversal.
    """
    opens, highs, lows, closes, idx = _arrays(df)
    results = []
    for i in range(len(df)):
        r = _rng(highs[i], lows[i])
        if r == 0:
            continue
        body  = _body(opens[i], closes[i])
        lower = _lower_wick(opens[i], lows[i], closes[i])
        upper = _upper_wick(opens[i], highs[i], closes[i])
        if body == 0:
            continue
        if (body / r) > body_pct:
            continue
        if lower < shadow_mult * body:
            continue
        if upper > 0.15 * r:          # upper wick must be small
            continue
        shadow_score = min(lower / body, 6.0) / 6.0
        body_score   = 1.0 - (body / r) / body_pct
        conf = round(0.5 * shadow_score + 0.5 * body_score, 4)
        results.append(PatternResult(
            pattern="hammer", direction="bullish", confidence=conf,
            timestamp=idx[i], bar_index=i,
            notes=f"lower_wick={lower:.3f}, body={body:.3f}",
        ))
    return results


def scan_shooting_star(df: pd.DataFrame, body_pct: float = 0.35,
                       shadow_mult: float = 2.0) -> list[PatternResult]:
    """
    Shooting Star — small body at bottom of range, long upper wick,
    tiny lower wick. Bearish reversal.
    """
    opens, highs, lows, closes, idx = _arrays(df)
    results = []
    for i in range(len(df)):
        r = _rng(highs[i], lows[i])
        if r == 0:
            continue
        body  = _body(opens[i], closes[i])
        upper = _upper_wick(opens[i], highs[i], closes[i])
        lower = _lower_wick(opens[i], lows[i], closes[i])
        if body == 0:
            continue
        if (body / r) > body_pct:
            continue
        if upper < shadow_mult * body:
            continue
        if lower > 0.15 * r:
            continue
        shadow_score = min(upper / body, 6.0) / 6.0
        body_score   = 1.0 - (body / r) / body_pct
        conf = round(0.5 * shadow_score + 0.5 * body_score, 4)
        results.append(PatternResult(
            pattern="shooting_star", direction="bearish", confidence=conf,
            timestamp=idx[i], bar_index=i,
            notes=f"upper_wick={upper:.3f}, body={body:.3f}",
        ))
    return results


def scan_pin_bar(df: pd.DataFrame, wick_mult: float = 3.0) -> list[PatternResult]:
    """
    Pin Bar — wick ≥ wick_mult × body in either direction.
    Direction is set by which wick dominates (long lower = bullish rejection,
    long upper = bearish rejection).
    """
    opens, highs, lows, closes, idx = _arrays(df)
    results = []
    for i in range(len(df)):
        r = _rng(highs[i], lows[i])
        if r == 0:
            continue
        body  = max(_body(opens[i], closes[i]), 1e-8)
        upper = _upper_wick(opens[i], highs[i], closes[i])
        lower = _lower_wick(opens[i], lows[i], closes[i])
        max_wick = max(upper, lower)
        if max_wick < wick_mult * body:
            continue
        direction = "bullish" if lower > upper else "bearish"
        conf = round(min(max_wick / body, 10.0) / 10.0, 4)
        results.append(PatternResult(
            pattern="pin_bar", direction=direction, confidence=conf,
            timestamp=idx[i], bar_index=i,
            notes=f"wick={max_wick:.3f}, body={body:.3f}",
        ))
    return results


# ── Two-candle patterns ───────────────────────────────────────────────────────

def scan_bullish_engulfing(df: pd.DataFrame) -> list[PatternResult]:
    """
    Bullish Engulfing — bearish candle followed by a bullish candle
    whose body completely engulfs the previous body.
    """
    opens, highs, lows, closes, idx = _arrays(df)
    results = []
    for i in range(1, len(df)):
        # Day 1 must be bearish
        if closes[i-1] >= opens[i-1]:
            continue
        # Day 2 must be bullish
        if closes[i] <= opens[i]:
            continue
        prev_body_top = opens[i-1]   # open > close for bearish
        prev_body_bot = closes[i-1]
        curr_body_top = closes[i]    # close > open for bullish
        curr_body_bot = opens[i]
        if curr_body_top > prev_body_top and curr_body_bot < prev_body_bot:
            body_prev = prev_body_top - prev_body_bot
            body_curr = curr_body_top - curr_body_bot
            conf = round(min(body_curr / body_prev, 3.0) / 3.0, 4)
            results.append(PatternResult(
                pattern="bullish_engulfing", direction="bullish", confidence=conf,
                timestamp=idx[i], bar_index=i, start_index=i-1,
                notes=f"engulf_ratio={body_curr/body_prev:.2f}×",
            ))
    return results


def scan_bearish_engulfing(df: pd.DataFrame) -> list[PatternResult]:
    """
    Bearish Engulfing — bullish candle followed by a bearish candle
    whose body completely engulfs the previous body.
    """
    opens, highs, lows, closes, idx = _arrays(df)
    results = []
    for i in range(1, len(df)):
        if closes[i-1] <= opens[i-1]:   # Day 1 must be bullish
            continue
        if closes[i] >= opens[i]:        # Day 2 must be bearish
            continue
        prev_top = closes[i-1]
        prev_bot = opens[i-1]
        curr_top = opens[i]
        curr_bot = closes[i]
        if curr_top > prev_top and curr_bot < prev_bot:
            body_prev = prev_top - prev_bot
            body_curr = curr_top - curr_bot
            conf = round(min(body_curr / body_prev, 3.0) / 3.0, 4)
            results.append(PatternResult(
                pattern="bearish_engulfing", direction="bearish", confidence=conf,
                timestamp=idx[i], bar_index=i, start_index=i-1,
                notes=f"engulf_ratio={body_curr/body_prev:.2f}×",
            ))
    return results


# ── Three-candle patterns ─────────────────────────────────────────────────────

def scan_morning_star(df: pd.DataFrame, star_body_pct: float = 0.25) -> list[PatternResult]:
    """
    Morning Star — large bearish, small-body star (gap down), large bullish.
    Day 3 must close above the midpoint of Day 1's body. Bullish reversal.
    """
    opens, highs, lows, closes, idx = _arrays(df)
    results = []
    for i in range(2, len(df)):
        o1, c1 = opens[i-2], closes[i-2]
        o2, c2 = opens[i-1], closes[i-1]
        o3, c3 = opens[i],   closes[i]

        # Day 1: large bearish
        r1 = _rng(highs[i-2], lows[i-2])
        if r1 == 0 or c1 >= o1:
            continue
        if _body(o1, c1) < 0.4 * r1:
            continue

        # Day 2: small body (star)
        r2 = _rng(highs[i-1], lows[i-1])
        if r2 == 0:
            continue
        if _body(o2, c2) > star_body_pct * r2 and r2 > 0.01 * c1:
            continue

        # Day 3: large bullish, closes above midpoint of day 1
        midpoint = (o1 + c1) / 2
        if c3 <= o3 or c3 <= midpoint:
            continue
        r3 = _rng(highs[i], lows[i])
        if _body(o3, c3) < 0.4 * r3:
            continue

        body3 = _body(o3, c3)
        body1 = _body(o1, c1)
        conf  = round(min(body3 / body1, 1.5) / 1.5, 4)
        results.append(PatternResult(
            pattern="morning_star", direction="bullish", confidence=conf,
            timestamp=idx[i], bar_index=i, start_index=i-2,
            notes=f"day3_close={c3:.2f} > midpoint={midpoint:.2f}",
        ))
    return results


def scan_evening_star(df: pd.DataFrame, star_body_pct: float = 0.25) -> list[PatternResult]:
    """
    Evening Star — large bullish, small-body star (gap up), large bearish.
    Day 3 must close below the midpoint of Day 1's body. Bearish reversal.
    """
    opens, highs, lows, closes, idx = _arrays(df)
    results = []
    for i in range(2, len(df)):
        o1, c1 = opens[i-2], closes[i-2]
        o2, c2 = opens[i-1], closes[i-1]
        o3, c3 = opens[i],   closes[i]

        # Day 1: large bullish
        r1 = _rng(highs[i-2], lows[i-2])
        if r1 == 0 or c1 <= o1:
            continue
        if _body(o1, c1) < 0.4 * r1:
            continue

        # Day 2: small body (star)
        r2 = _rng(highs[i-1], lows[i-1])
        if r2 == 0:
            continue
        if _body(o2, c2) > star_body_pct * r2 and r2 > 0.01 * c1:
            continue

        # Day 3: large bearish, closes below midpoint of day 1
        midpoint = (o1 + c1) / 2
        if c3 >= o3 or c3 >= midpoint:
            continue
        r3 = _rng(highs[i], lows[i])
        if _body(o3, c3) < 0.4 * r3:
            continue

        body3 = _body(o3, c3)
        body1 = _body(o1, c1)
        conf  = round(min(body3 / body1, 1.5) / 1.5, 4)
        results.append(PatternResult(
            pattern="evening_star", direction="bearish", confidence=conf,
            timestamp=idx[i], bar_index=i, start_index=i-2,
            notes=f"day3_close={c3:.2f} < midpoint={midpoint:.2f}",
        ))
    return results
