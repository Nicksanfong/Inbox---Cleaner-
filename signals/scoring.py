"""
Confluence scorer — converts one indicator row + detected patterns into
directional scores (0–100 each for bullish and bearish).

Score architecture
──────────────────
  Trend      EMA stack, MACD line & histogram, ADX / DI     0 – 65 pts
  Momentum   RSI zone (oversold / recovering / overbought)  0 – 20 pts
  Volatility Bollinger Band position, VWAP                  0 – 21 pts
  Patterns   Candlestick + chart patterns (capped at 25)    0 – 25 pts
  Multi-TF   Added by engine after comparing timeframes     0 – 10 pts
                                                            ──────────
  Theoretical max                                           ~141 → capped 100

Calibration: a strong 8/9-indicator uptrend scores ~65 pts per TF; with
3-TF alignment bonus (+10) the combined score reaches ~75 — exactly the
default threshold.  Requires multi-timeframe agreement for most signals to fire.
"""
from __future__ import annotations

import pandas as pd

from patterns.models import PatternResult
from signals.models import Confirmation

# ── Pattern catalogue ─────────────────────────────────────────────────────────

_PATTERN_WEIGHT: dict[str, float] = {
    # bullish
    "bullish_engulfing":        10,
    "morning_star":             10,
    "inv_head_and_shoulders":    9,
    "double_bottom":             9,
    "ascending_triangle":        8,
    "trendline_break_up":        8,
    "hammer":                    6,
    "pin_bar":                   5,
    # bearish
    "bearish_engulfing":        10,
    "evening_star":             10,
    "head_and_shoulders":        9,
    "double_top":                9,
    "descending_triangle":       8,
    "trendline_break_down":      8,
    "shooting_star":             6,
    # neutral (direction set per result)
    "symmetrical_triangle":      3,
    "doji":                      3,
}

_BULLISH_PATTERNS = frozenset({
    "bullish_engulfing", "morning_star", "inv_head_and_shoulders",
    "double_bottom", "ascending_triangle", "trendline_break_up",
    "hammer", "pin_bar",
})
_BEARISH_PATTERNS = frozenset({
    "bearish_engulfing", "evening_star", "head_and_shoulders",
    "double_top", "descending_triangle", "trendline_break_down",
    "shooting_star",
})

_MAX_PATTERN_PTS = 25.0


# ── Public function ───────────────────────────────────────────────────────────

def score_bar(
    row: pd.Series,
    bar_patterns: list[PatternResult],
) -> dict:
    """
    Score a single indicator bar for bullish AND bearish confluence.

    Parameters
    ----------
    row          : one row from a run_all() DataFrame (all indicator columns present)
    bar_patterns : PatternResult objects whose bar_index == this bar's index

    Returns
    -------
    {
        "bullish"            : float,          # 0–100
        "bearish"            : float,          # 0–100
        "bull_confirmations" : list[Confirmation],
        "bear_confirmations" : list[Confirmation],
        "breakdown"          : dict,           # category → {"bullish": pts, "bearish": pts}
    }
    """
    bull: list[Confirmation] = []
    bear: list[Confirmation] = []

    def _b(lst, source, direction, weight, detail):
        if weight > 0:
            lst.append(Confirmation(source=source, direction=direction,
                                    weight=round(weight, 2), detail=detail))

    close = float(row.get("close", 0))

    # ── Trend: EMA alignment ─────────────────────────────────────────────────
    e8   = float(row.get("ema_8",   close))
    e21  = float(row.get("ema_21",  close))
    e50  = float(row.get("ema_50",  close))
    e200 = float(row.get("ema_200", close))

    if close > e200:
        _b(bull, "above_ema_200",   "bullish", 11, f"close ${close:.2f} > EMA200 ${e200:.2f}")
    else:
        _b(bear, "below_ema_200",   "bearish", 11, f"close ${close:.2f} < EMA200 ${e200:.2f}")

    if e21 > e50:
        _b(bull, "ema_21_50_bull",  "bullish",  7, f"EMA21 ${e21:.2f} > EMA50 ${e50:.2f}")
    else:
        _b(bear, "ema_21_50_bear",  "bearish",  7, f"EMA21 ${e21:.2f} < EMA50 ${e50:.2f}")

    if e8 > e21:
        _b(bull, "ema_8_21_bull",   "bullish",  6, f"EMA8 ${e8:.2f} > EMA21 ${e21:.2f}")
    else:
        _b(bear, "ema_8_21_bear",   "bearish",  6, f"EMA8 ${e8:.2f} < EMA21 ${e21:.2f}")

    if close > e8 > e21 > e50 > e200:
        _b(bull, "full_ema_stack",  "bullish", 11, "Price > EMA8 > EMA21 > EMA50 > EMA200 — all aligned")
    elif close < e8 < e21 < e50 < e200:
        _b(bear, "full_ema_stack",  "bearish", 11, "Price < EMA8 < EMA21 < EMA50 < EMA200 — all aligned")

    # ── Trend: MACD ──────────────────────────────────────────────────────────
    macd_v = float(row.get("macd",      0))
    macd_s = float(row.get("macd_sig",  0))
    macd_h = float(row.get("macd_hist", 0))

    if macd_v > macd_s:
        _b(bull, "macd_bull_cross", "bullish", 11, f"MACD {macd_v:+.4f} > signal {macd_s:+.4f}")
    else:
        _b(bear, "macd_bear_cross", "bearish", 11, f"MACD {macd_v:+.4f} < signal {macd_s:+.4f}")

    if macd_h > 0:
        _b(bull, "macd_hist_pos",   "bullish",  6, f"MACD histogram +{macd_h:.4f} (momentum building)")
    else:
        _b(bear, "macd_hist_neg",   "bearish",  6, f"MACD histogram {macd_h:.4f} (momentum fading)")

    # ── Trend: ADX / Directional Index ───────────────────────────────────────
    adx       = float(row.get("adx",       0))
    adx_plus  = float(row.get("adx_plus",  0))
    adx_minus = float(row.get("adx_minus", 0))

    if adx_plus > adx_minus:
        _b(bull, "di_bull", "bullish",  6, f"+DI {adx_plus:.1f} > -DI {adx_minus:.1f} — buyers in control")
    else:
        _b(bear, "di_bear", "bearish",  6, f"-DI {adx_minus:.1f} > +DI {adx_plus:.1f} — sellers in control")

    if adx > 25:
        target = bull if adx_plus > adx_minus else bear
        dirn   = "bullish" if adx_plus > adx_minus else "bearish"
        _b(target, "adx_strong", dirn,  7, f"ADX {adx:.1f} > 25 — strong trend confirmed")

    # ── Momentum: RSI ────────────────────────────────────────────────────────
    rsi = float(row.get("rsi_14", 50))

    if rsi < 30:
        _b(bull, "rsi_oversold",   "bullish", 20, f"RSI {rsi:.1f} < 30 — deeply oversold, reversal likely")
    elif rsi < 45:
        _b(bull, "rsi_recovering", "bullish", 11, f"RSI {rsi:.1f} in 30–45 — bullish recovery zone")
    elif rsi > 70:
        _b(bear, "rsi_overbought", "bearish", 20, f"RSI {rsi:.1f} > 70 — deeply overbought, pullback likely")
    elif rsi > 55:
        _b(bear, "rsi_fading",     "bearish", 11, f"RSI {rsi:.1f} in 55–70 — bearish momentum zone")

    # ── Volatility: Bollinger Bands ──────────────────────────────────────────
    bb_pct = float(row.get("bb_pct", 0.5))
    bb_u   = float(row.get("bb_upper",  close + 1))
    bb_l   = float(row.get("bb_lower",  close - 1))

    if bb_pct < 0.20:
        _b(bull, "at_bb_lower", "bullish", 14,
           f"BB%B {bb_pct:.2f} — near lower band ${bb_l:.2f} (mean-reversion support)")
    elif bb_pct > 0.80:
        _b(bear, "at_bb_upper", "bearish", 14,
           f"BB%B {bb_pct:.2f} — near upper band ${bb_u:.2f} (mean-reversion resistance)")

    # ── Volatility: VWAP ─────────────────────────────────────────────────────
    vwap = float(row.get("vwap", close))
    if close > vwap:
        _b(bull, "above_vwap", "bullish", 7, f"close ${close:.2f} > VWAP ${vwap:.2f}")
    else:
        _b(bear, "below_vwap", "bearish", 7, f"close ${close:.2f} < VWAP ${vwap:.2f}")

    # ── Patterns ─────────────────────────────────────────────────────────────
    bull_pat_pts = 0.0
    bear_pat_pts = 0.0

    for p in bar_patterns:
        base_w = _PATTERN_WEIGHT.get(p.pattern, 4)
        pts    = base_w * p.confidence
        label  = p.pattern.replace("_", " ")
        if p.pattern in _BULLISH_PATTERNS or p.direction == "bullish":
            bull_pat_pts += pts
            _b(bull, f"pat_{p.pattern}", "bullish", pts,
               f"{label}  conf={p.confidence:.2f}")
        elif p.pattern in _BEARISH_PATTERNS or p.direction == "bearish":
            bear_pat_pts += pts
            _b(bear, f"pat_{p.pattern}", "bearish", pts,
               f"{label}  conf={p.confidence:.2f}")

    # ── Raw scores (pattern contribution capped at _MAX_PATTERN_PTS) ─────────
    def _raw(lst, pat_pts):
        non_pat = sum(c.weight for c in lst if not c.source.startswith("pat_"))
        return non_pat + min(pat_pts, _MAX_PATTERN_PTS)

    bull_score = min(_raw(bull, bull_pat_pts), 100.0)
    bear_score = min(_raw(bear, bear_pat_pts), 100.0)

    # ── Breakdown by category ─────────────────────────────────────────────────
    _trend_src = {
        "above_ema_200", "below_ema_200", "ema_21_50_bull", "ema_21_50_bear",
        "ema_8_21_bull",  "ema_8_21_bear",  "full_ema_stack",
        "macd_bull_cross", "macd_bear_cross", "macd_hist_pos", "macd_hist_neg",
        "di_bull", "di_bear", "adx_strong",
    }
    _mom_src  = {"rsi_oversold", "rsi_recovering", "rsi_overbought", "rsi_fading"}
    _vol_src  = {"at_bb_lower", "at_bb_upper", "above_vwap", "below_vwap"}

    def _cat(lst, keys):
        return round(sum(c.weight for c in lst if c.source in keys), 2)

    breakdown = {
        "Trend":      {"bullish": _cat(bull, _trend_src), "bearish": _cat(bear, _trend_src)},
        "Momentum":   {"bullish": _cat(bull, _mom_src),   "bearish": _cat(bear, _mom_src)},
        "Volatility": {"bullish": _cat(bull, _vol_src),   "bearish": _cat(bear, _vol_src)},
        "Patterns":   {
            "bullish": round(min(bull_pat_pts, _MAX_PATTERN_PTS), 2),
            "bearish": round(min(bear_pat_pts, _MAX_PATTERN_PTS), 2),
        },
    }

    return {
        "bullish":             bull_score,
        "bearish":             bear_score,
        "bull_confirmations":  bull,
        "bear_confirmations":  bear,
        "breakdown":           breakdown,
    }
