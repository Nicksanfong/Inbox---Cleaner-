"""
JSON-backed disk persistence for pending orders and open positions.

Files are re-written atomically on every mutation so state survives
a process restart.  Load order on startup restores all in-flight state.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from execution.models import ManagedPosition, PendingOrder

log = logging.getLogger(__name__)

_ORDERS_FILE    = "pending_orders.json"
_POSITIONS_FILE = "open_positions.json"


class PositionStore:

    def __init__(self, data_dir: str = "data"):
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._orders_path    = self._dir / _ORDERS_FILE
        self._positions_path = self._dir / _POSITIONS_FILE

        self._orders:    dict[str, PendingOrder]    = {}   # keyed by order_id
        self._positions: dict[str, ManagedPosition] = {}   # keyed by symbol

        self._load()

    # ── Pending orders ────────────────────────────────────────────────────────

    def save_order(self, order: PendingOrder) -> None:
        self._orders[order.order_id] = order
        self._flush_orders()

    def update_order(self, order: PendingOrder) -> None:
        self._orders[order.order_id] = order
        self._flush_orders()

    def remove_order(self, order_id: str) -> None:
        self._orders.pop(order_id, None)
        self._flush_orders()

    def get_order(self, order_id: str) -> PendingOrder | None:
        return self._orders.get(order_id)

    def all_orders(self) -> list[PendingOrder]:
        return list(self._orders.values())

    # ── Open positions ────────────────────────────────────────────────────────

    def save_position(self, pos: ManagedPosition) -> None:
        self._positions[pos.symbol] = pos
        self._flush_positions()

    def update_position(self, pos: ManagedPosition) -> None:
        self._positions[pos.symbol] = pos
        self._flush_positions()

    def remove_position(self, symbol: str) -> None:
        self._positions.pop(symbol, None)
        self._flush_positions()

    def get_position(self, symbol: str) -> ManagedPosition | None:
        return self._positions.get(symbol)

    def all_positions(self) -> list[ManagedPosition]:
        return list(self._positions.values())

    # ── Internal ──────────────────────────────────────────────────────────────

    def _flush_orders(self) -> None:
        data = [asdict(o) for o in self._orders.values()]
        self._write_atomic(self._orders_path, data)

    def _flush_positions(self) -> None:
        data = [asdict(p) for p in self._positions.values()]
        self._write_atomic(self._positions_path, data)

    @staticmethod
    def _write_atomic(path: Path, data: list) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(path)

    def _load(self) -> None:
        if self._orders_path.exists():
            try:
                for d in json.loads(self._orders_path.read_text()):
                    o = PendingOrder(**d)
                    self._orders[o.order_id] = o
                if self._orders:
                    log.info("PositionStore: restored %d pending order(s)", len(self._orders))
            except Exception as exc:
                log.warning("PositionStore: failed to load orders — %s", exc)

        if self._positions_path.exists():
            try:
                for d in json.loads(self._positions_path.read_text()):
                    p = ManagedPosition(**d)
                    self._positions[p.symbol] = p
                if self._positions:
                    log.info("PositionStore: restored %d open position(s)", len(self._positions))
            except Exception as exc:
                log.warning("PositionStore: failed to load positions — %s", exc)
