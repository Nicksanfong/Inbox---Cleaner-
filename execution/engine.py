"""
ExecutionEngine — orchestrates order placement, polling, and position management.

Typical flow
────────────
  engine = ExecutionEngine(broker, risk_manager, journal, store)

  # On each signal:
  pending = engine.submit(signal, size, current_price=ask)

  # On each poll loop (every N seconds):
  events = engine.poll()   # returns list of event dicts
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING

from execution.journal import (
    ORDER_PLACED, ORDER_FILLED, ORDER_PARTIAL, ORDER_REJECTED,
    ORDER_CANCELLED, ORDER_FALLBACK, ORDER_FAILED, TP_HIT,
    BREAKEVEN_SET, POSITION_CLOSED,
)
from execution.models import ManagedPosition, PendingOrder

if TYPE_CHECKING:
    from execution.broker import AlpacaBroker
    from execution.journal import TradeJournal
    from execution.store import PositionStore
    from risk.manager import RiskManager
    from risk.models import PositionSize
    from signals.models import Signal

log = logging.getLogger(__name__)


class ExecutionEngine:
    """
    Submits orders, polls for fills, and manages open positions.

    Design decisions:
    - Alpaca bracket orders carry a single TP leg set to TP3 (3R).  TP1 and
      TP2 are tracked locally so we can scale out without maintaining multiple
      simultaneous OCO orders.
    - On partial fills we wait; on rejection we retry once with a market order.
    - All state is persisted immediately so a process restart can resume work.
    """

    _TERMINAL = {"filled", "cancelled", "expired", "rejected", "done_for_day"}

    def __init__(
        self,
        broker:             "AlpacaBroker",
        risk_manager:       "RiskManager",
        journal:            "TradeJournal",
        store:              "PositionStore",
        limit_timeout_secs: int = 120,
        max_attempts:       int = 2,
    ):
        self._broker   = broker
        self._risk     = risk_manager
        self._journal  = journal
        self._store    = store
        self._timeout  = limit_timeout_secs
        self._max_att  = max_attempts

    # ── Public API ────────────────────────────────────────────────────────────

    def submit(
        self,
        signal:        "Signal",
        size:          "PositionSize",
        current_price: float | None = None,
    ) -> PendingOrder | None:
        """
        Place a new bracket order derived from *signal* and *size*.

        Returns the PendingOrder record or None if entry was blocked
        (position already exists, size invalid, broker error).
        """
        symbol    = signal.symbol
        direction = "long" if signal.direction == "BUY" else "short"

        if not size.valid:
            log.warning("ENGINE  skipping %s — invalid size: %s",
                        symbol, size.rejection_reasons)
            return None

        if self._store.get_position(symbol):
            log.info("ENGINE  skipping %s — position already open", symbol)
            return None

        for o in self._store.all_orders():
            if o.symbol == symbol and o.status not in ("filled", "rejected",
                                                        "cancelled", "failed"):
                log.info("ENGINE  skipping %s — order already pending", symbol)
                return None

        # Determine entry price
        if current_price is None:
            try:
                quote = self._broker.get_latest_quote(symbol)
                current_price = quote["ask"] if direction == "long" else quote["bid"]
            except Exception as exc:
                log.warning("ENGINE  quote fetch failed for %s — %s", symbol, exc)
                current_price = size.entry_price

        from alpaca.trading.enums import OrderSide
        side = OrderSide.BUY if direction == "long" else OrderSide.SELL

        client_oid = str(uuid.uuid4())
        try:
            order = self._broker.place_limit_bracket(
                symbol=symbol,
                side=side,
                qty=size.units,
                limit_price=current_price,
                stop_price=size.stop_price,
                tp_price=size.take_profits[-1],   # TP3 as bracket leg
                client_order_id=client_oid,
            )
        except Exception as exc:
            log.error("ENGINE  order submission failed for %s — %s", symbol, exc)
            return None

        pending = PendingOrder(
            order_id=str(order.id),
            client_order_id=client_oid,
            symbol=symbol,
            side=side.value,
            direction=direction,
            qty=size.units,
            order_type="limit",
            limit_price=current_price,
            stop_price=size.stop_price,
            take_profits=list(size.take_profits),
            tp_qtys=list(size.tp_units),
            signal_score=signal.score,
        )
        self._store.save_order(pending)
        self._journal.log(
            ORDER_PLACED, symbol, direction,
            qty=size.units, price=current_price,
            order_id=pending.order_id,
            stop=size.stop_price,
            tp1=size.take_profits[0] if size.take_profits else None,
            tp2=size.take_profits[1] if len(size.take_profits) > 1 else None,
            tp3=size.take_profits[2] if len(size.take_profits) > 2 else None,
            score=signal.score,
        )
        return pending

    def poll(self) -> list[dict]:
        """
        Check all pending orders and open positions for state changes.

        Returns a list of event dicts (one per state change).
        """
        events: list[dict] = []
        events.extend(self._poll_orders())
        events.extend(self._poll_positions())
        return events

    def close_position(self, symbol: str) -> bool:
        """
        Cancel any open orders for *symbol* and remove the tracked position.
        Does NOT place a closing order on Alpaca — caller handles that if needed.
        """
        closed = False
        for order in list(self._store.all_orders()):
            if order.symbol == symbol:
                self._broker.cancel_order(order.order_id)
                self._store.remove_order(order.order_id)
                self._journal.log(ORDER_CANCELLED, symbol,
                                  order_id=order.order_id)
                closed = True

        pos = self._store.get_position(symbol)
        if pos:
            self._journal.log(
                POSITION_CLOSED, symbol, pos.direction,
                qty=pos.qty_remaining,
                order_id=pos.alpaca_order_id,
            )
            self._store.remove_position(symbol)
            closed = True

        return closed

    # ── Internal — order polling ──────────────────────────────────────────────

    def _poll_orders(self) -> list[dict]:
        events: list[dict] = []
        for pending in list(self._store.all_orders()):
            alpaca = self._broker.get_order(pending.order_id)
            if alpaca is None:
                continue

            status = str(alpaca.status.value).lower()

            if status == "filled":
                events.append(self._on_fill(pending, alpaca))

            elif status == "partially_filled":
                events.append(self._on_partial(pending, alpaca))

            elif status == "rejected":
                events.append(self._on_rejection(pending, alpaca))

            elif status in ("cancelled", "expired", "done_for_day"):
                events.append(self._on_cancelled(pending))

            elif status == "new" and pending.order_type == "limit":
                age = time.time() - _epoch(pending.created_at)
                if age > self._timeout:
                    log.info("ENGINE  limit timeout for %s (%ds old), trying market",
                             pending.symbol, int(age))
                    events.append(self._try_market_fallback(pending))

        return events

    def _on_fill(self, order: PendingOrder, alpaca_order) -> dict:
        fill_price = float(alpaca_order.filled_avg_price or order.limit_price or 0)
        fill_qty   = float(alpaca_order.filled_qty or order.qty)

        pos = ManagedPosition(
            symbol=order.symbol,
            direction=order.direction,
            entry_price=fill_price,
            qty=fill_qty,
            qty_remaining=fill_qty,
            stop_price=order.stop_price,
            take_profits=list(order.take_profits),
            tp_qtys=list(order.tp_qtys),
            alpaca_order_id=order.order_id,
        )
        self._store.save_position(pos)
        self._store.remove_order(order.order_id)
        self._journal.log(
            ORDER_FILLED, order.symbol, order.direction,
            qty=fill_qty, price=fill_price,
            order_id=order.order_id,
        )
        return {"event": ORDER_FILLED, "symbol": order.symbol,
                "price": fill_price, "qty": fill_qty}

    def _on_partial(self, order: PendingOrder, alpaca_order) -> dict:
        filled = float(alpaca_order.filled_qty or 0)
        if filled == order.filled_qty:
            return {}   # no change since last poll
        order.filled_qty = filled
        order.status     = "partially_filled"
        self._store.update_order(order)
        self._journal.log(
            ORDER_PARTIAL, order.symbol, order.direction,
            qty=filled, order_id=order.order_id,
        )
        return {"event": ORDER_PARTIAL, "symbol": order.symbol, "qty": filled}

    def _on_rejection(self, order: PendingOrder, alpaca_order) -> dict:
        reason = getattr(alpaca_order, "failed_at", "") or "rejected by broker"
        self._journal.log(
            ORDER_REJECTED, order.symbol, order.direction,
            order_id=order.order_id, reason=str(reason),
        )
        if order.attempts < self._max_att:
            return self._try_market_fallback(order)

        order.status = "failed"
        self._store.update_order(order)
        self._journal.log(ORDER_FAILED, order.symbol, order.direction,
                          order_id=order.order_id)
        return {"event": ORDER_FAILED, "symbol": order.symbol}

    def _on_cancelled(self, order: PendingOrder) -> dict:
        self._store.remove_order(order.order_id)
        self._journal.log(ORDER_CANCELLED, order.symbol, order.direction,
                          order_id=order.order_id)
        return {"event": ORDER_CANCELLED, "symbol": order.symbol}

    def _try_market_fallback(self, order: PendingOrder) -> dict:
        # Cancel the limit order first
        self._broker.cancel_order(order.order_id)

        from alpaca.trading.enums import OrderSide
        side = OrderSide.BUY if order.direction == "long" else OrderSide.SELL

        tp3 = order.take_profits[-1] if order.take_profits else None
        if tp3 is None:
            log.error("ENGINE  no TP available for market fallback on %s", order.symbol)
            order.status = "failed"
            self._store.update_order(order)
            return {"event": ORDER_FAILED, "symbol": order.symbol}

        client_oid = str(uuid.uuid4())
        try:
            new_order = self._broker.place_market_bracket(
                symbol=order.symbol,
                side=side,
                qty=order.qty,
                stop_price=order.stop_price,
                tp_price=tp3,
                client_order_id=client_oid,
            )
        except Exception as exc:
            log.error("ENGINE  market fallback failed for %s — %s", order.symbol, exc)
            order.status = "failed"
            self._store.update_order(order)
            self._journal.log(ORDER_FAILED, order.symbol, order.direction,
                              order_id=order.order_id)
            return {"event": ORDER_FAILED, "symbol": order.symbol}

        old_id       = order.order_id
        order.order_id      = str(new_order.id)
        order.client_order_id = client_oid
        order.order_type    = "market"
        order.limit_price   = None
        order.attempts     += 1
        order.status        = "new"
        self._store.remove_order(old_id)
        self._store.save_order(order)
        self._journal.log(
            ORDER_FALLBACK, order.symbol, order.direction,
            qty=order.qty, order_id=order.order_id,
        )
        return {"event": ORDER_FALLBACK, "symbol": order.symbol,
                "new_order_id": order.order_id}

    # ── Internal — position polling ───────────────────────────────────────────

    def _poll_positions(self) -> list[dict]:
        events: list[dict] = []
        for pos in list(self._store.all_positions()):
            try:
                quote = self._broker.get_latest_quote(pos.symbol)
                price = quote["mid"]
            except Exception:
                continue

            events.extend(self._check_virtual_tps(pos, price))

        return events

    def _check_virtual_tps(
        self, pos: ManagedPosition, current_price: float,
    ) -> list[dict]:
        """Check TP1 / TP2 virtually (TP3 is handled by Alpaca's bracket leg)."""
        events: list[dict] = []
        is_long = pos.direction == "long"
        changed = False

        for i, (tp, qty) in enumerate(zip(pos.take_profits[:2],
                                          pos.tp_qtys[:2])):
            if pos.tp_hits[i]:
                continue
            hit = (is_long and current_price >= tp) or (
                  not is_long and current_price <= tp)
            if not hit:
                continue

            pos.tp_hits[i]   = True
            pos.qty_remaining = max(0.0, pos.qty_remaining - qty)
            changed = True
            self._journal.log(
                TP_HIT, pos.symbol, pos.direction,
                qty=qty, price=tp,
                order_id=pos.alpaca_order_id,
                tp_level=i + 1,
            )
            events.append({"event": TP_HIT, "symbol": pos.symbol,
                           "tp_level": i + 1, "price": tp})

            # Move stop to breakeven on TP1
            if i == 0 and not pos.at_breakeven:
                pos.stop_price  = pos.entry_price
                pos.at_breakeven = True
                self._journal.log(
                    BREAKEVEN_SET, pos.symbol, pos.direction,
                    price=pos.entry_price,
                    order_id=pos.alpaca_order_id,
                )
                events.append({"event": BREAKEVEN_SET, "symbol": pos.symbol,
                               "stop": pos.entry_price})

        if changed:
            self._store.update_position(pos)

        return events


# ── Helpers ───────────────────────────────────────────────────────────────────

def _epoch(iso: str) -> float:
    """Convert ISO 8601 timestamp string to Unix epoch float."""
    from datetime import datetime, timezone
    try:
        return datetime.fromisoformat(iso).replace(tzinfo=timezone.utc).timestamp()
    except Exception:
        return 0.0
