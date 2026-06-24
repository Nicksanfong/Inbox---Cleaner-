"""
Signal generation engine.

Call SignalEngine.scan(frames, symbol) where frames is a dict mapping
timeframe labels ("5m", "15m", "30m", "1H", "4H", "D", "W") to raw
OHLCV DataFrames.  Any subset of timeframes is valid; missing ones
are skipped automatically.

The engine:
  1. Runs indicators + patterns on each timeframe.
  2. Scores each bar with the confluence scorer.
  3. Applies a weighted multi-timeframe combination.
  4. Adds a bonus when multiple timeframes agree (max +10 pts).
  5. Enforces minimum confirmation count and score threshold.
  6. Returns Signal objects sorted by score descending.

Multi-timeframe weighting
─────────────────────────
  Weekly  (W)    25 %   ← macro trend / market regime
  Daily   (D)    20 %   ← primary trend
  4-hour  (4H)   18 %   ← swing structure
  Hourly  (1H)   15 %   ← intraday momentum
  30-min  (30m)  10 %   ← short-term momentum
  15-min  (15m)   7 %   ← entry refinement
  5-min   (5m)    5 %   ← entry trigger
                ─────
                100 %

  Agreement bonus (ratio of agreeing TFs):
    100 %  →  +10 pts
     ≥ 70 % →  + 6 pts
     ≥ 50 % →  + 3 pts
     < 50 % →  + 0 pts
"""
from __future__ import annotations

import logging

import pandas as pd

from indicators.combined import run_all
from patterns.detector import scan_all
from signals.models import Confirmation, Signal
from signals.scoring import score_bar

log = logging.getLogger(__name__)

# Timeframes listed highest → lowest; used for iteration order
_TF_ORDER = ["W", "D", "4H", "1H", "30m", "15m", "5m"]

_TF_WEIGHT: dict[str, float] = {
    "W":   0.25,
    "D":   0.20,
    "4H":  0.18,
    "1H":  0.15,
    "30m": 0.10,
    "15m": 0.07,
    "5m":  0.05,
}


def _mtf_bonus(agreements: int, n_tfs: int) -> float:
    """Points added when timeframes agree on direction."""
    if n_tfs < 2:
        return 0.0
    ratio = agreements / n_tfs
    if ratio == 1.0:    # all agree
        return 10.0
    if ratio >= 0.70:   # strong majority (≥ 70 %)
        return 6.0
    if ratio >= 0.50:   # simple majority
        return 3.0
    return 0.0


class SignalEngine:
    def __init__(self, threshold: float = 75.0, min_confirmations: int = 3):
        """
        Parameters
        ----------
        threshold         : minimum confluence score (0–100) to fire a signal
        min_confirmations : minimum number of distinct confirmation sources required
        """
        self.threshold         = threshold
        self.min_confirmations = min_confirmations

    # ── Public API ────────────────────────────────────────────────────────────

    def scan(
        self,
        frames: dict[str, pd.DataFrame],
        symbol: str = "UNKNOWN",
    ) -> list[Signal]:
        """
        Multi-timeframe confluence scan.

        Parameters
        ----------
        frames : {"5m": df, "15m": df, "30m": df, "1H": df,
                  "4H": df, "D": df, "W": df}  — raw OHLCV DataFrames.
                 Any subset of timeframes is valid; missing ones are skipped.
        symbol : ticker label (display only)

        Returns
        -------
        List of Signal objects (BUY and/or SELL) sorted by score descending.
        Only signals meeting both threshold and min_confirmations are returned.
        """
        tf_results = self._score_all_timeframes(frames, symbol)
        if not tf_results:
            return []

        signals = []
        for direction in ("bullish", "bearish"):
            sig = self._try_build(symbol, direction, tf_results)
            if sig is not None:
                signals.append(sig)

        return sorted(signals, key=lambda s: -s.score)

    def raw_scores(
        self,
        frames: dict[str, pd.DataFrame],
        symbol: str = "UNKNOWN",
    ) -> dict:
        """
        Return per-timeframe and combined scores WITHOUT threshold filtering.
        Useful for diagnostics and the show_signals demo.

        Returns
        -------
        {
            "by_tf":  {tf: {"bullish": float, "bearish": float, "timestamp": Timestamp}},
            "combined_bull": float,
            "combined_bear": float,
            "mtf_bonus_bull": float,
            "mtf_bonus_bear": float,
        }
        """
        tf_results = self._score_all_timeframes(frames, symbol)
        if not tf_results:
            return {}

        by_tf = {
            tf: {
                "bullish":   r["bullish"],
                "bearish":   r["bearish"],
                "timestamp": r["timestamp"],
            }
            for tf, r in tf_results.items()
        }

        def _combined(direction):
            total_w = sum(_TF_WEIGHT.get(tf, 0.25) for tf in tf_results)
            raw     = sum(_TF_WEIGHT.get(tf, 0.25) * r[direction]
                          for tf, r in tf_results.items())
            return (raw / total_w) if total_w > 0 else 0.0

        def _bonus(direction):
            agrees = sum(
                1 for r in tf_results.values()
                if (r["bullish"] >= r["bearish"]) == (direction == "bullish")
            )
            return _mtf_bonus(agrees, len(tf_results))

        return {
            "by_tf":          by_tf,
            "combined_bull":  round(_combined("bullish"), 1),
            "combined_bear":  round(_combined("bearish"), 1),
            "mtf_bonus_bull": _bonus("bullish"),
            "mtf_bonus_bear": _bonus("bearish"),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _score_all_timeframes(
        self,
        frames: dict[str, pd.DataFrame],
        symbol: str,
    ) -> dict[str, dict]:
        tf_results: dict[str, dict] = {}
        for tf in _TF_ORDER:
            if tf not in frames or frames[tf].empty:
                continue
            try:
                tf_results[tf] = self._score_single_tf(frames[tf])
            except Exception as exc:
                log.warning("SignalEngine [%s %s]: scoring failed — %s", symbol, tf, exc)
        return tf_results

    def _score_single_tf(self, df: pd.DataFrame) -> dict:
        """Run indicators, patterns, and scorer on the LAST bar of one timeframe."""
        result_df    = run_all(df)
        all_patterns = scan_all(result_df)
        last_idx     = len(result_df) - 1
        last_row     = result_df.iloc[-1]
        last_ts      = result_df.index[-1]
        bar_patterns = [p for p in all_patterns if p.bar_index == last_idx]
        scored       = score_bar(last_row, bar_patterns)
        scored["timestamp"] = last_ts
        return scored

    def _try_build(
        self,
        symbol:     str,
        direction:  str,                  # "bullish" | "bearish"
        tf_results: dict[str, dict],
    ) -> Signal | None:
        """Attempt to build a Signal for one direction; returns None if criteria not met."""
        n_tfs = len(tf_results)

        # How many timeframes agree with this direction?
        agreements = sum(
            1 for r in tf_results.values()
            if (r["bullish"] >= r["bearish"]) == (direction == "bullish")
        )
        bonus = _mtf_bonus(agreements, n_tfs)

        # Weighted average of per-TF scores for this direction
        total_w   = sum(_TF_WEIGHT.get(tf, 0.25) for tf in tf_results)
        weighted  = sum(
            _TF_WEIGHT.get(tf, 0.25) * r[direction]
            for tf, r in tf_results.items()
        ) / total_w if total_w > 0 else 0.0

        final_score = min(round(weighted + bonus, 1), 100.0)

        # Confirmations come from the shortest (entry) timeframe
        entry_tf = next(
            (tf for tf in reversed(_TF_ORDER) if tf in tf_results), None
        )
        if entry_tf is None:
            return None

        primary  = tf_results[entry_tf]
        conf_key = "bull_confirmations" if direction == "bullish" else "bear_confirmations"
        confs    = list(primary[conf_key])

        # Append MTF confirmation as a synthetic entry
        if bonus > 0:
            confs.append(Confirmation(
                source="mtf_alignment",
                direction=direction,
                weight=bonus,
                detail=(
                    f"{agreements}/{n_tfs} timeframe(s) aligned "
                    f"with {direction} direction → +{bonus:.0f} pts"
                ),
            ))

        # Enforce both criteria
        if len(confs) < self.min_confirmations:
            return None
        if final_score < self.threshold:
            return None

        # Build breakdown (from entry TF, plus MTF bonus line)
        breakdown = dict(primary["breakdown"])
        if bonus > 0:
            breakdown["Multi-TF"] = {direction: round(bonus, 1)}

        side = "BUY" if direction == "bullish" else "SELL"
        return Signal(
            symbol=symbol,
            direction=side,
            score=final_score,
            timeframe=entry_tf,
            timestamp=primary["timestamp"],
            confirmations=confs,
            breakdown=breakdown,
        )
