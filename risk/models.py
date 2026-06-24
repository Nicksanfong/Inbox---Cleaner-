"""Data classes for the risk management module."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RiskConfig:
    """Controls every risk decision for a strategy instance."""

    # Per-trade risk
    risk_pct:               float = 0.01     # fraction of account risked (1%)

    # Entry-quality gate
    min_rr:                 float = 2.0      # TP2 R-multiple must be ≥ this

    # Daily halt
    max_daily_drawdown_pct: float = 0.05     # halt if day's loss exceeds 5%

    # Portfolio caps
    max_open_positions:     int   = 5

    # Partial-exit scaling (R multiples and portion closed at each)
    tp_r_levels:            tuple = (1.0, 2.0, 3.0)
    tp_scale_pcts:          tuple = (0.40, 0.35, 0.25)  # must sum to 1.0

    # Trailing stop
    trailing_stop_enabled:  bool  = False
    trailing_pct:           float = 0.02     # trail distance as % of entry price

    # Correlation guard
    correlation_threshold:  float = 0.70     # reject if any open pos corr ≥ this

    # Execution
    allow_fractional_units: bool  = True     # False → round down to whole shares


@dataclass
class PositionSize:
    """Output of the position-sizing calculation for one trade."""

    symbol:            str
    direction:         str           # "long" | "short"
    entry_price:       float
    stop_price:        float
    stop_distance:     float         # abs(entry − stop)
    units:             float         # shares / contracts to open
    risk_amount:       float         # $ placed at risk (account × risk_pct)
    rr_ratio:          float         # R:R to the final TP
    take_profits:      list = field(default_factory=list)   # [TP1, TP2, TP3] prices
    tp_units:          list = field(default_factory=list)   # units to close at each TP
    breakeven_trigger: float = 0.0   # price at which to move stop to entry
    valid:             bool  = True
    rejection_reasons: list  = field(default_factory=list)

    def __repr__(self) -> str:
        if not self.valid:
            return f"PositionSize(REJECTED: {'; '.join(self.rejection_reasons)})"
        tp_str = "  ".join(
            f"TP{i + 1}=${v:.4g}({u:.4g}u)"
            for i, (v, u) in enumerate(zip(self.take_profits, self.tp_units))
        )
        return (
            f"PositionSize({self.symbol} {self.direction} "
            f"entry={self.entry_price:.4g} stop={self.stop_price:.4g} "
            f"units={self.units:.4g} risk=${self.risk_amount:.2f} "
            f"R:R=1:{self.rr_ratio:.1f}  {tp_str})"
        )


@dataclass
class OpenPosition:
    """Live state of one open trade — used for per-bar stop updates."""

    symbol:           str
    direction:        str            # "long" | "short"
    entry_price:      float
    current_stop:     float
    units_remaining:  float
    high_water_mark:  float = 0.0    # highest price seen (long) / lowest (short)
    at_breakeven:     bool  = False
    tp_hits:          list  = field(default_factory=lambda: [False, False, False])
