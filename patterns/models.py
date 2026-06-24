from dataclasses import dataclass, field


@dataclass
class PatternResult:
    """A single detected pattern on a price series."""
    pattern:    str            # snake_case name, e.g. "bullish_engulfing"
    direction:  str            # "bullish" | "bearish" | "neutral"
    confidence: float          # 0.0 – 1.0 (higher = cleaner textbook pattern)
    timestamp:  object         # pd.Timestamp of the bar where pattern completes
    bar_index:  int            # integer row index in the source DataFrame
    start_index: int = -1      # first bar involved (multi-bar patterns)
    notes:      str = ""       # human-readable detail

    def __repr__(self) -> str:
        return (
            f"PatternResult({self.pattern!r}, {self.direction}, "
            f"conf={self.confidence:.2f}, {str(self.timestamp)[:10]})"
        )
