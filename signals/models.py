from dataclasses import dataclass, field


@dataclass
class Confirmation:
    """One piece of evidence that contributed to a signal score."""
    source:    str    # machine-readable key, e.g. "above_ema_200"
    direction: str    # "bullish" | "bearish"
    weight:    float  # raw score contribution (pts)
    detail:    str    # human-readable explanation

    def __repr__(self) -> str:
        sign = "+" if self.direction == "bullish" else "-"
        return f"[{sign}{self.weight:.1f}] {self.source}: {self.detail}"


@dataclass
class Signal:
    """A generated BUY or SELL signal."""
    symbol:        str
    direction:     str            # "BUY" | "SELL"
    score:         float          # 0 – 100
    timeframe:     str            # entry timeframe label, e.g. "15m"
    timestamp:     object         # pd.Timestamp of the triggering bar
    confirmations: list = field(default_factory=list)   # list[Confirmation]
    breakdown:     dict = field(default_factory=dict)   # category → pts earned
    notes:         str = ""

    def __repr__(self) -> str:
        return (
            f"Signal({self.direction} {self.symbol}, score={self.score:.1f}, "
            f"tf={self.timeframe}, confs={len(self.confirmations)}, "
            f"{str(self.timestamp)[:16]})"
        )
