"""Data classes shared across the execution layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PendingOrder:
    """One entry order from placement through fill / rejection."""
    order_id:        str              # Alpaca order ID
    client_order_id: str              # our UUID (idempotency key)
    symbol:          str
    side:            str              # "buy" | "sell"
    direction:       str              # "long" | "short"
    qty:             float
    order_type:      str              # "limit" | "market"
    limit_price:     float | None
    stop_price:      float            # attached stop-loss price
    take_profits:    list = field(default_factory=list)  # [tp1, tp2, tp3]
    tp_qtys:         list = field(default_factory=list)  # units at each TP
    status:          str  = "new"
    filled_qty:      float = 0.0
    avg_fill_price:  float | None = None
    signal_score:    float = 0.0
    created_at:      str  = field(default_factory=_now)
    updated_at:      str  = field(default_factory=_now)
    attempts:        int  = 1        # incremented on market-fallback retry

    def __repr__(self) -> str:
        return (
            f"PendingOrder({self.symbol} {self.direction} {self.order_type}"
            f" qty={self.qty} @{self.limit_price} status={self.status}"
            f" attempts={self.attempts})"
        )


@dataclass
class ManagedPosition:
    """Live open position with local stop / TP state."""
    symbol:          str
    direction:       str              # "long" | "short"
    entry_price:     float
    qty:             float
    qty_remaining:   float
    stop_price:      float
    take_profits:    list = field(default_factory=list)   # [tp1, tp2, tp3]
    tp_qtys:         list = field(default_factory=list)
    tp_hits:         list = field(default_factory=lambda: [False, False, False])
    at_breakeven:    bool  = False
    high_water_mark: float = 0.0
    alpaca_order_id: str  = ""       # entry order ID on Alpaca
    opened_at:       str  = field(default_factory=_now)

    def __repr__(self) -> str:
        tps = "/".join("✓" if h else "·" for h in self.tp_hits)
        return (
            f"ManagedPosition({self.symbol} {self.direction}"
            f" entry={self.entry_price} qty={self.qty_remaining}/{self.qty}"
            f" stop={self.stop_price} tp=[{tps}])"
        )


@dataclass
class JournalEntry:
    """One event in the trade journal."""
    ts:        str
    event:     str            # ORDER_PLACED | ORDER_FILLED | ORDER_REJECTED | …
    symbol:    str
    direction: str  = ""
    qty:       float = 0.0
    price:     float | None = None
    order_id:  str  = ""
    details:   dict = field(default_factory=dict)
