"""
AlpacaBroker — thin, testable wrapper around alpaca-py.

All Alpaca SDK calls are isolated here so the engine never imports
alpaca-py directly and tests can swap in a mock without patching globals.
"""
from __future__ import annotations

import logging
import uuid

log = logging.getLogger(__name__)


class AlpacaBroker:
    """
    Wraps TradingClient (and optionally StockHistoricalDataClient) for:
      - Account info
      - Latest quotes
      - Bracket order placement (limit entry → stop + TP legs)
      - Order status queries / cancellation
    """

    def __init__(self, trading_client, data_client=None):
        self._trading = trading_client
        self._data    = data_client

    # ── Account ───────────────────────────────────────────────────────────────

    def get_account(self) -> dict:
        acct = self._trading.get_account()
        return {
            "balance":         float(acct.equity),
            "buying_power":    float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
            "currency":        acct.currency,
        }

    # ── Quotes ────────────────────────────────────────────────────────────────

    def get_latest_quote(self, symbol: str) -> dict:
        """Returns {"bid": float, "ask": float, "mid": float}."""
        if self._data is None:
            raise RuntimeError("data_client required for get_latest_quote()")
        from alpaca.data.requests import StockLatestQuoteRequest
        req   = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quote = self._data.get_stock_latest_quote(req)[symbol]
        bid   = float(quote.bid_price)
        ask   = float(quote.ask_price)
        return {"bid": bid, "ask": ask, "mid": round((bid + ask) / 2, 4)}

    # ── Order placement ───────────────────────────────────────────────────────

    def place_limit_bracket(
        self,
        symbol:     str,
        side,                           # OrderSide enum
        qty:        float,
        limit_price: float,
        stop_price:  float,
        tp_price:    float,
        client_order_id: str | None = None,
    ):
        """
        Bracket limit order: entry at limit_price, stop-loss at stop_price,
        take-profit limit at tp_price.  Stop and TP are OCO legs — when one
        fires the other is automatically cancelled by Alpaca.
        """
        from alpaca.trading.enums  import OrderClass, TimeInForce
        from alpaca.trading.requests import (
            LimitOrderRequest, StopLossRequest, TakeProfitRequest,
        )
        req = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            limit_price=round(limit_price, 2),
            stop_loss=StopLossRequest(stop_price=round(stop_price, 2)),
            take_profit=TakeProfitRequest(limit_price=round(tp_price, 2)),
            client_order_id=client_order_id or str(uuid.uuid4()),
        )
        order = self._trading.submit_order(req)
        log.info(
            "BROKER  limit_bracket  %s %s  qty=%s  limit=%.2f  sl=%.2f  tp=%.2f  id=%s",
            side.value, symbol, qty, limit_price, stop_price, tp_price, order.id,
        )
        return order

    def place_market_bracket(
        self,
        symbol:    str,
        side,
        qty:       float,
        stop_price: float,
        tp_price:   float,
        client_order_id: str | None = None,
    ):
        """Market entry with stop + TP legs (bracket class, fills immediately)."""
        from alpaca.trading.enums    import OrderClass, TimeInForce
        from alpaca.trading.requests import (
            MarketOrderRequest, StopLossRequest, TakeProfitRequest,
        )
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            stop_loss=StopLossRequest(stop_price=round(stop_price, 2)),
            take_profit=TakeProfitRequest(limit_price=round(tp_price, 2)),
            client_order_id=client_order_id or str(uuid.uuid4()),
        )
        order = self._trading.submit_order(req)
        log.info(
            "BROKER  market_bracket  %s %s  qty=%s  sl=%.2f  tp=%.2f  id=%s",
            side.value, symbol, qty, stop_price, tp_price, order.id,
        )
        return order

    # ── Order queries ─────────────────────────────────────────────────────────

    def get_order(self, order_id: str):
        try:
            return self._trading.get_order_by_id(order_id)
        except Exception as exc:
            log.warning("BROKER  get_order %s — %s", order_id, exc)
            return None

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._trading.cancel_order_by_id(order_id)
            log.info("BROKER  cancelled  %s", order_id)
            return True
        except Exception as exc:
            log.warning("BROKER  cancel %s failed — %s", order_id, exc)
            return False

    def get_open_orders(self, symbol: str | None = None):
        from alpaca.trading.enums    import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest
        req = GetOrdersRequest(
            status=QueryOrderStatus.OPEN,
            symbols=[symbol] if symbol else None,
        )
        return self._trading.get_orders(req)

    def get_open_position(self, symbol: str):
        try:
            return self._trading.get_open_position(symbol)
        except Exception:
            return None
