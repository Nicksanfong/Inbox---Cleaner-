"""
Chart pattern detection.

Each scan_* function accepts an OHLCV DataFrame (DatetimeIndex) and
returns a list of PatternResult objects — one per detected occurrence.

Patterns covered:
  double_top / double_bottom — two comparable peaks/troughs with a valley/peak between
  head_and_shoulders         — three-peak reversal (bearish)
  inv_head_and_shoulders     — three-trough reversal (bullish)
  ascending_triangle         — flat resistance, rising support
  descending_triangle        — flat support, falling resistance
  symmetrical_triangle       — converging highs and lows
  trendline_break_up         — price closes above a falling resistance trendline
  trendline_break_down       — price closes below a rising support trendline
"""
import numpy as np
import pandas as pd

from patterns.models import PatternResult

# ── helpers ───────────────────────────────────────────────────────────────────

def _find_peaks(arr: np.ndarray, order: int = 3) -> np.ndarray:
    """Return indices of local maxima where arr[i] > all arr[i±1..order]."""
    peaks = []
    for i in range(order, len(arr) - order):
        window = arr[i - order: i + order + 1]
        if arr[i] == window.max() and arr[i] > arr[i - 1] and arr[i] > arr[i + 1]:
            peaks.append(i)
    return np.array(peaks, dtype=int)


def _find_troughs(arr: np.ndarray, order: int = 3) -> np.ndarray:
    """Return indices of local minima where arr[i] < all arr[i±1..order]."""
    troughs = []
    for i in range(order, len(arr) - order):
        window = arr[i - order: i + order + 1]
        if arr[i] == window.min() and arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
            troughs.append(i)
    return np.array(troughs, dtype=int)


def _linreg(x: np.ndarray, y: np.ndarray):
    """Return (slope, intercept) of a least-squares fit."""
    if len(x) < 2:
        return 0.0, float(y[0]) if len(y) else 0.0
    m = np.polyfit(x.astype(float), y.astype(float), 1)
    return float(m[0]), float(m[1])


# ── Double Top / Double Bottom ────────────────────────────────────────────────

def scan_double_top(
    df: pd.DataFrame,
    peak_order: int = 5,
    similarity_pct: float = 0.02,
    min_gap: int = 10,
    max_gap: int = 60,
) -> list[PatternResult]:
    """
    Double Top — two peaks at similar price levels with a valley between them.
    Bearish reversal signal. Confidence based on peak similarity and valley depth.
    """
    highs = df["high"].values.astype(float)
    closes = df["close"].values.astype(float)
    idx = df.index
    results = []

    peaks = _find_peaks(highs, order=peak_order)
    for j in range(1, len(peaks)):
        i1, i2 = peaks[j - 1], peaks[j]
        gap = i2 - i1
        if gap < min_gap or gap > max_gap:
            continue
        p1, p2 = highs[i1], highs[i2]
        avg_peak = (p1 + p2) / 2
        if avg_peak == 0:
            continue
        diff_pct = abs(p1 - p2) / avg_peak
        if diff_pct > similarity_pct:
            continue
        # valley must be meaningfully below the peaks
        valley = closes[i1:i2].min()
        valley_depth = (avg_peak - valley) / avg_peak
        if valley_depth < 0.02:
            continue
        similarity_score = 1.0 - diff_pct / similarity_pct
        depth_score = min(valley_depth / 0.05, 1.0)
        conf = round(0.6 * similarity_score + 0.4 * depth_score, 4)
        results.append(PatternResult(
            pattern="double_top", direction="bearish", confidence=conf,
            timestamp=idx[i2], bar_index=i2, start_index=i1,
            notes=f"peaks={p1:.2f},{p2:.2f} valley={valley:.2f}",
        ))
    return results


def scan_double_bottom(
    df: pd.DataFrame,
    trough_order: int = 5,
    similarity_pct: float = 0.02,
    min_gap: int = 10,
    max_gap: int = 60,
) -> list[PatternResult]:
    """
    Double Bottom — two troughs at similar price levels with a peak between them.
    Bullish reversal signal.
    """
    lows = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)
    idx = df.index
    results = []

    troughs = _find_troughs(lows, order=trough_order)
    for j in range(1, len(troughs)):
        i1, i2 = troughs[j - 1], troughs[j]
        gap = i2 - i1
        if gap < min_gap or gap > max_gap:
            continue
        t1, t2 = lows[i1], lows[i2]
        avg_trough = (t1 + t2) / 2
        if avg_trough == 0:
            continue
        diff_pct = abs(t1 - t2) / avg_trough
        if diff_pct > similarity_pct:
            continue
        peak = closes[i1:i2].max()
        peak_height = (peak - avg_trough) / avg_trough
        if peak_height < 0.02:
            continue
        similarity_score = 1.0 - diff_pct / similarity_pct
        height_score = min(peak_height / 0.05, 1.0)
        conf = round(0.6 * similarity_score + 0.4 * height_score, 4)
        results.append(PatternResult(
            pattern="double_bottom", direction="bullish", confidence=conf,
            timestamp=idx[i2], bar_index=i2, start_index=i1,
            notes=f"troughs={t1:.2f},{t2:.2f} peak={peak:.2f}",
        ))
    return results


# ── Head and Shoulders ────────────────────────────────────────────────────────

def scan_head_and_shoulders(
    df: pd.DataFrame,
    peak_order: int = 4,
    shoulder_similarity: float = 0.04,
    min_gap: int = 5,
    max_gap: int = 40,
) -> list[PatternResult]:
    """
    Head and Shoulders — left shoulder, higher head, right shoulder at similar height.
    Bearish reversal. Confidence based on shoulder symmetry and head prominence.
    """
    highs = df["high"].values.astype(float)
    idx = df.index
    results = []

    peaks = _find_peaks(highs, order=peak_order)
    for k in range(2, len(peaks)):
        il, ih, ir = peaks[k - 2], peaks[k - 1], peaks[k]
        if (ih - il) < min_gap or (ih - il) > max_gap:
            continue
        if (ir - ih) < min_gap or (ir - ih) > max_gap:
            continue
        sl, head, sr = highs[il], highs[ih], highs[ir]
        if head <= sl or head <= sr:
            continue
        avg_shoulder = (sl + sr) / 2
        if avg_shoulder == 0:
            continue
        shoulder_diff = abs(sl - sr) / avg_shoulder
        if shoulder_diff > shoulder_similarity:
            continue
        head_prominence = (head - avg_shoulder) / avg_shoulder
        if head_prominence < 0.02:
            continue
        symmetry_score = 1.0 - shoulder_diff / shoulder_similarity
        prominence_score = min(head_prominence / 0.05, 1.0)
        gap_score = 1.0 - abs((ih - il) - (ir - ih)) / max(ih - il, ir - ih)
        conf = round(0.4 * symmetry_score + 0.4 * prominence_score + 0.2 * gap_score, 4)
        results.append(PatternResult(
            pattern="head_and_shoulders", direction="bearish", confidence=conf,
            timestamp=idx[ir], bar_index=ir, start_index=il,
            notes=f"L={sl:.2f} H={head:.2f} R={sr:.2f}",
        ))
    return results


def scan_inv_head_and_shoulders(
    df: pd.DataFrame,
    trough_order: int = 4,
    shoulder_similarity: float = 0.04,
    min_gap: int = 5,
    max_gap: int = 40,
) -> list[PatternResult]:
    """
    Inverse Head and Shoulders — left shoulder, lower head, right shoulder at similar depth.
    Bullish reversal.
    """
    lows = df["low"].values.astype(float)
    idx = df.index
    results = []

    troughs = _find_troughs(lows, order=trough_order)
    for k in range(2, len(troughs)):
        il, ih, ir = troughs[k - 2], troughs[k - 1], troughs[k]
        if (ih - il) < min_gap or (ih - il) > max_gap:
            continue
        if (ir - ih) < min_gap or (ir - ih) > max_gap:
            continue
        sl, head, sr = lows[il], lows[ih], lows[ir]
        if head >= sl or head >= sr:
            continue
        avg_shoulder = (sl + sr) / 2
        if avg_shoulder == 0:
            continue
        shoulder_diff = abs(sl - sr) / avg_shoulder
        if shoulder_diff > shoulder_similarity:
            continue
        head_depth = (avg_shoulder - head) / avg_shoulder
        if head_depth < 0.02:
            continue
        symmetry_score = 1.0 - shoulder_diff / shoulder_similarity
        depth_score = min(head_depth / 0.05, 1.0)
        gap_score = 1.0 - abs((ih - il) - (ir - ih)) / max(ih - il, ir - ih)
        conf = round(0.4 * symmetry_score + 0.4 * depth_score + 0.2 * gap_score, 4)
        results.append(PatternResult(
            pattern="inv_head_and_shoulders", direction="bullish", confidence=conf,
            timestamp=idx[ir], bar_index=ir, start_index=il,
            notes=f"L={sl:.2f} H={head:.2f} R={sr:.2f}",
        ))
    return results


# ── Triangles ─────────────────────────────────────────────────────────────────

def _scan_triangle(
    df: pd.DataFrame,
    window: int = 30,
    min_touches: int = 3,
) -> list[tuple]:
    """
    Shared helper — returns (start, end, resist_slope, support_slope, conf) tuples
    over rolling windows.
    """
    highs = df["high"].values.astype(float)
    lows = df["low"].values.astype(float)
    n = len(df)
    results = []
    for end in range(window, n):
        start = end - window
        x = np.arange(window)
        h = highs[start:end]
        l = lows[start:end]
        peak_idx = _find_peaks(h, order=2)
        trough_idx = _find_troughs(l, order=2)
        if len(peak_idx) < 2 or len(trough_idx) < 2:
            continue
        rs, ri = _linreg(peak_idx, h[peak_idx])
        ss, si = _linreg(trough_idx, l[trough_idx])
        results.append((start, end - 1, rs, ss, ri, si))
    return results


def scan_ascending_triangle(
    df: pd.DataFrame,
    window: int = 30,
    flat_slope_pct: float = 0.001,
) -> list[PatternResult]:
    """
    Ascending Triangle — flat resistance (near-zero slope), rising support (positive slope).
    Bullish continuation.
    """
    idx = df.index
    results = []
    for start, end, rs, ss, ri, si in _scan_triangle(df, window):
        price_scale = float(df["close"].iloc[start:end + 1].mean())
        if price_scale == 0:
            continue
        rs_norm = abs(rs) / price_scale
        ss_norm = ss / price_scale
        if rs_norm > flat_slope_pct and ss_norm > flat_slope_pct / 2:
            continue
        if ss_norm <= 0:
            continue
        flat_score = max(0.0, 1.0 - rs_norm / flat_slope_pct)
        rise_score = min(ss_norm / (flat_slope_pct * 2), 1.0)
        conf = round(0.5 * flat_score + 0.5 * rise_score, 4)
        results.append(PatternResult(
            pattern="ascending_triangle", direction="bullish", confidence=conf,
            timestamp=idx[end], bar_index=end, start_index=start,
            notes=f"resist_slope={rs:.5f} support_slope={ss:.5f}",
        ))
    return results


def scan_descending_triangle(
    df: pd.DataFrame,
    window: int = 30,
    flat_slope_pct: float = 0.001,
) -> list[PatternResult]:
    """
    Descending Triangle — flat support, falling resistance. Bearish continuation.
    """
    idx = df.index
    results = []
    for start, end, rs, ss, ri, si in _scan_triangle(df, window):
        price_scale = float(df["close"].iloc[start:end + 1].mean())
        if price_scale == 0:
            continue
        rs_norm = rs / price_scale
        ss_norm = abs(ss) / price_scale
        if ss_norm > flat_slope_pct and rs_norm < -flat_slope_pct / 2:
            continue
        if rs_norm >= 0:
            continue
        flat_score = max(0.0, 1.0 - ss_norm / flat_slope_pct)
        fall_score = min(abs(rs_norm) / (flat_slope_pct * 2), 1.0)
        conf = round(0.5 * flat_score + 0.5 * fall_score, 4)
        results.append(PatternResult(
            pattern="descending_triangle", direction="bearish", confidence=conf,
            timestamp=idx[end], bar_index=end, start_index=start,
            notes=f"resist_slope={rs:.5f} support_slope={ss:.5f}",
        ))
    return results


def scan_symmetrical_triangle(
    df: pd.DataFrame,
    window: int = 30,
    flat_slope_pct: float = 0.001,
) -> list[PatternResult]:
    """
    Symmetrical Triangle — resistance falling, support rising (converging).
    Neutral — breakout direction decides bias.
    """
    idx = df.index
    results = []
    for start, end, rs, ss, ri, si in _scan_triangle(df, window):
        price_scale = float(df["close"].iloc[start:end + 1].mean())
        if price_scale == 0:
            continue
        rs_norm = rs / price_scale
        ss_norm = ss / price_scale
        if rs_norm >= 0 or ss_norm <= 0:
            continue
        if abs(rs_norm) < flat_slope_pct / 4 or ss_norm < flat_slope_pct / 4:
            continue
        symmetry = 1.0 - abs(abs(rs_norm) - ss_norm) / (abs(rs_norm) + ss_norm)
        strength = min((abs(rs_norm) + ss_norm) / (flat_slope_pct * 2), 1.0)
        conf = round(0.5 * symmetry + 0.5 * strength, 4)
        results.append(PatternResult(
            pattern="symmetrical_triangle", direction="neutral", confidence=conf,
            timestamp=idx[end], bar_index=end, start_index=start,
            notes=f"resist_slope={rs:.5f} support_slope={ss:.5f}",
        ))
    return results


# ── Trendline Breaks ──────────────────────────────────────────────────────────

def scan_trendline_break_up(
    df: pd.DataFrame,
    lookback: int = 20,
    min_touches: int = 2,
) -> list[PatternResult]:
    """
    Trendline Break Up — price closes above a descending resistance line
    fitted through recent swing highs.
    """
    highs = df["high"].values.astype(float)
    closes = df["close"].values.astype(float)
    idx = df.index
    results = []

    for i in range(lookback, len(df)):
        window_h = highs[i - lookback: i]
        peak_idx = _find_peaks(window_h, order=2)
        if len(peak_idx) < min_touches:
            continue
        slope, intercept = _linreg(peak_idx, window_h[peak_idx])
        if slope >= 0:
            continue
        projected = slope * lookback + intercept
        if closes[i] > projected:
            break_pct = (closes[i] - projected) / projected if projected != 0 else 0
            conf = round(min(break_pct / 0.01, 1.0), 4)
            results.append(PatternResult(
                pattern="trendline_break_up", direction="bullish", confidence=conf,
                timestamp=idx[i], bar_index=i, start_index=i - lookback,
                notes=f"close={closes[i]:.2f} > trendline={projected:.2f}",
            ))
    return results


def scan_trendline_break_down(
    df: pd.DataFrame,
    lookback: int = 20,
    min_touches: int = 2,
) -> list[PatternResult]:
    """
    Trendline Break Down — price closes below an ascending support line
    fitted through recent swing lows.
    """
    lows = df["low"].values.astype(float)
    closes = df["close"].values.astype(float)
    idx = df.index
    results = []

    for i in range(lookback, len(df)):
        window_l = lows[i - lookback: i]
        trough_idx = _find_troughs(window_l, order=2)
        if len(trough_idx) < min_touches:
            continue
        slope, intercept = _linreg(trough_idx, window_l[trough_idx])
        if slope <= 0:
            continue
        projected = slope * lookback + intercept
        if closes[i] < projected:
            break_pct = (projected - closes[i]) / projected if projected != 0 else 0
            conf = round(min(break_pct / 0.01, 1.0), 4)
            results.append(PatternResult(
                pattern="trendline_break_down", direction="bearish", confidence=conf,
                timestamp=idx[i], bar_index=i, start_index=i - lookback,
                notes=f"close={closes[i]:.2f} < trendline={projected:.2f}",
            ))
    return results
