"""Append-only JSONL trade journal."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from execution.models import JournalEntry

log = logging.getLogger(__name__)

# ── Event name constants ──────────────────────────────────────────────────────
ORDER_PLACED    = "ORDER_PLACED"
ORDER_FILLED    = "ORDER_FILLED"
ORDER_PARTIAL   = "ORDER_PARTIAL"
ORDER_REJECTED  = "ORDER_REJECTED"
ORDER_CANCELLED = "ORDER_CANCELLED"
ORDER_FALLBACK  = "ORDER_FALLBACK"     # limit → market retry
ORDER_FAILED    = "ORDER_FAILED"       # all attempts exhausted
STOP_HIT        = "STOP_HIT"
TP_HIT          = "TP_HIT"
BREAKEVEN_SET   = "BREAKEVEN_SET"
POSITION_CLOSED = "POSITION_CLOSED"


class TradeJournal:
    """
    Append-only JSONL log — one JSON object per line, one per event.
    Safe to tail, grep, or load into pandas.

    Example lines:
        {"ts":"2024-01-05T14:23:01Z","event":"ORDER_PLACED","symbol":"AAPL",...}
        {"ts":"2024-01-05T14:23:02Z","event":"ORDER_FILLED","symbol":"AAPL",...}
    """

    def __init__(self, path: str = "logs/trade_journal.jsonl"):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ── Write ─────────────────────────────────────────────────────────────────

    def log(
        self,
        event:     str,
        symbol:    str,
        direction: str   = "",
        qty:       float = 0.0,
        price:     float | None = None,
        order_id:  str   = "",
        **details,
    ) -> JournalEntry:
        entry = JournalEntry(
            ts=datetime.now(timezone.utc).isoformat(),
            event=event,
            symbol=symbol,
            direction=direction,
            qty=float(qty),
            price=price,
            order_id=order_id,
            details=details,
        )
        record = {
            "ts":        entry.ts,
            "event":     entry.event,
            "symbol":    entry.symbol,
            "direction": entry.direction,
            "qty":       entry.qty,
            "price":     entry.price,
            "order_id":  entry.order_id,
            **entry.details,
        }
        with self._path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        log.info(
            "JOURNAL  %-18s  %s %-5s  qty=%-6s  price=%s",
            event, symbol, direction, qty, price,
        )
        return entry

    # ── Read ──────────────────────────────────────────────────────────────────

    def recent(self, n: int = 50) -> list[JournalEntry]:
        if not self._path.exists():
            return []
        lines = self._path.read_text().strip().splitlines()
        entries: list[JournalEntry] = []
        for raw in lines[-n:]:
            try:
                d = json.loads(raw)
                core = {"ts", "event", "symbol", "direction", "qty", "price", "order_id"}
                entries.append(JournalEntry(
                    ts=d.get("ts", ""),
                    event=d.get("event", ""),
                    symbol=d.get("symbol", ""),
                    direction=d.get("direction", ""),
                    qty=float(d.get("qty", 0)),
                    price=d.get("price"),
                    order_id=d.get("order_id", ""),
                    details={k: v for k, v in d.items() if k not in core},
                ))
            except (json.JSONDecodeError, ValueError):
                pass
        return entries

    def for_symbol(self, symbol: str) -> list[JournalEntry]:
        return [e for e in self.recent(100_000) if e.symbol == symbol]

    def __len__(self) -> int:
        if not self._path.exists():
            return 0
        return sum(1 for _ in self._path.open())
