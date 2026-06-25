"""
Tests for the execution layer: models, journal, store, broker wrapper, engine.

All Alpaca SDK calls are mocked — no network required.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from execution.engine  import ExecutionEngine
from execution.journal import (
    ORDER_CANCELLED, ORDER_FALLBACK, ORDER_FAILED, ORDER_FILLED,
    ORDER_PARTIAL, ORDER_PLACED, ORDER_REJECTED, BREAKEVEN_SET,
    POSITION_CLOSED, TP_HIT, TradeJournal,
)
from execution.models  import JournalEntry, ManagedPosition, PendingOrder
from execution.store   import PositionStore


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_dir(tmp_path):
    return tmp_path


@pytest.fixture()
def store(tmp_dir):
    return PositionStore(data_dir=str(tmp_dir / "data"))


@pytest.fixture()
def journal(tmp_dir):
    return TradeJournal(path=str(tmp_dir / "logs" / "test.jsonl"))


def _make_signal(symbol="AAPL", direction="BUY", score=82.0):
    from signals.models import Signal
    import pandas as pd
    return Signal(
        symbol=symbol,
        direction=direction,
        score=score,
        timeframe="15m",
        timestamp=pd.Timestamp("2024-01-10 10:00"),
    )


def _make_size(entry=150.0, stop=146.0, symbol="AAPL", direction="long"):
    from risk.models import PositionSize
    stop_dist = abs(entry - stop)
    units     = 10.0
    tp1 = round(entry + stop_dist * 1.0, 2)
    tp2 = round(entry + stop_dist * 2.0, 2)
    tp3 = round(entry + stop_dist * 3.0, 2)
    return PositionSize(
        symbol=symbol,
        direction=direction,
        entry_price=entry,
        stop_price=stop,
        stop_distance=stop_dist,
        units=units,
        risk_amount=150.0,
        rr_ratio=3.0,
        take_profits=[tp1, tp2, tp3],
        tp_units=[4.0, 3.5, 2.5],
        breakeven_trigger=tp1,
        valid=True,
    )


def _make_alpaca_order(
    order_id=None,
    status="new",
    filled_qty="0",
    filled_avg_price=None,
):
    o = SimpleNamespace(
        id=order_id or str(uuid.uuid4()),
        status=SimpleNamespace(value=status),
        filled_qty=filled_qty,
        filled_avg_price=filled_avg_price,
        failed_at=None,
    )
    return o


def _make_broker(
    alpaca_order=None,
    quote=None,
):
    broker = MagicMock()
    ao     = alpaca_order or _make_alpaca_order()
    broker.place_limit_bracket.return_value  = ao
    broker.place_market_bracket.return_value = ao
    broker.get_order.return_value            = ao
    broker.cancel_order.return_value         = True
    broker.get_latest_quote.return_value     = quote or {"bid": 149.9, "ask": 150.1, "mid": 150.0}
    return broker


def _make_engine(broker=None, journal=None, store=None, tmp_dir=None):
    from risk.manager import RiskManager
    b = broker or _make_broker()
    if tmp_dir is None:
        import tempfile
        tmp_dir = Path(tempfile.mkdtemp())
    j = journal or TradeJournal(path=str(tmp_dir / "j.jsonl"))
    s = store   or PositionStore(data_dir=str(tmp_dir / "data"))
    rm = RiskManager()
    return ExecutionEngine(b, rm, j, s), b, j, s


# ══════════════════════════════════════════════════════════════════════════════
# PendingOrder model
# ══════════════════════════════════════════════════════════════════════════════

class TestPendingOrder:
    def test_repr(self):
        o = PendingOrder(
            order_id="abc", client_order_id="xyz", symbol="AAPL",
            side="buy", direction="long", qty=10.0, order_type="limit",
            limit_price=150.0, stop_price=146.0,
        )
        assert "AAPL" in repr(o)
        assert "long" in repr(o)
        assert "limit" in repr(o)

    def test_defaults(self):
        o = PendingOrder(
            order_id="x", client_order_id="y", symbol="TSLA",
            side="sell", direction="short", qty=5.0, order_type="market",
            limit_price=None, stop_price=102.0,
        )
        assert o.status      == "new"
        assert o.filled_qty  == 0.0
        assert o.attempts    == 1
        assert o.take_profits == []
        assert o.tp_qtys     == []


# ══════════════════════════════════════════════════════════════════════════════
# ManagedPosition model
# ══════════════════════════════════════════════════════════════════════════════

class TestManagedPosition:
    def test_repr_shows_checkmarks(self):
        pos = ManagedPosition(
            symbol="AAPL", direction="long", entry_price=150.0,
            qty=10.0, qty_remaining=6.0, stop_price=150.0,
            tp_hits=[True, False, False],
        )
        r = repr(pos)
        assert "✓" in r
        assert "·" in r

    def test_defaults(self):
        pos = ManagedPosition(
            symbol="TSLA", direction="short", entry_price=200.0,
            qty=5.0, qty_remaining=5.0, stop_price=205.0,
        )
        assert pos.at_breakeven    == False
        assert pos.high_water_mark == 0.0
        assert pos.tp_hits         == [False, False, False]


# ══════════════════════════════════════════════════════════════════════════════
# TradeJournal
# ══════════════════════════════════════════════════════════════════════════════

class TestTradeJournal:
    def test_log_writes_jsonl(self, journal, tmp_dir):
        journal.log(ORDER_PLACED, "AAPL", "long", qty=10, price=150.0,
                    order_id="oid-1", stop=146.0)
        path = tmp_dir / "logs" / "test.jsonl"
        assert path.exists()
        line = json.loads(path.read_text().strip())
        assert line["event"]  == ORDER_PLACED
        assert line["symbol"] == "AAPL"
        assert line["price"]  == 150.0
        assert line["stop"]   == 146.0

    def test_multiple_events_appended(self, journal):
        for evt in [ORDER_PLACED, ORDER_FILLED, TP_HIT]:
            journal.log(evt, "AAPL", order_id="x")
        assert len(journal) == 3

    def test_recent_returns_last_n(self, journal):
        for i in range(10):
            journal.log(ORDER_PLACED, "AAPL", price=float(i))
        entries = journal.recent(3)
        assert len(entries) == 3
        assert entries[-1].price == 9.0

    def test_for_symbol_filters(self, journal):
        journal.log(ORDER_PLACED, "AAPL")
        journal.log(ORDER_PLACED, "TSLA")
        journal.log(ORDER_FILLED, "AAPL")
        results = journal.for_symbol("AAPL")
        assert len(results) == 2
        assert all(e.symbol == "AAPL" for e in results)

    def test_len_empty(self, journal):
        assert len(journal) == 0

    def test_recent_empty(self, journal):
        assert journal.recent() == []

    def test_log_returns_journal_entry(self, journal):
        entry = journal.log(ORDER_PLACED, "AAPL", qty=5.0, price=100.0)
        assert isinstance(entry, JournalEntry)
        assert entry.event  == ORDER_PLACED
        assert entry.symbol == "AAPL"

    def test_extra_kwargs_in_details(self, journal):
        journal.log(ORDER_PLACED, "AAPL", stop=145.0, tp1=155.0)
        e = journal.recent(1)[0]
        assert e.details.get("stop") == 145.0
        assert e.details.get("tp1")  == 155.0


# ══════════════════════════════════════════════════════════════════════════════
# PositionStore
# ══════════════════════════════════════════════════════════════════════════════

class TestPositionStore:
    def _order(self, order_id="o1", symbol="AAPL"):
        return PendingOrder(
            order_id=order_id, client_order_id="c1", symbol=symbol,
            side="buy", direction="long", qty=10.0, order_type="limit",
            limit_price=150.0, stop_price=146.0,
        )

    def _position(self, symbol="AAPL"):
        return ManagedPosition(
            symbol=symbol, direction="long", entry_price=150.0,
            qty=10.0, qty_remaining=10.0, stop_price=146.0,
        )

    def test_save_and_get_order(self, store):
        o = self._order()
        store.save_order(o)
        assert store.get_order("o1") is not None
        assert store.get_order("o1").symbol == "AAPL"

    def test_update_order(self, store):
        o = self._order()
        store.save_order(o)
        o.status = "filled"
        store.update_order(o)
        assert store.get_order("o1").status == "filled"

    def test_remove_order(self, store):
        store.save_order(self._order())
        store.remove_order("o1")
        assert store.get_order("o1") is None

    def test_all_orders(self, store):
        store.save_order(self._order("o1"))
        store.save_order(self._order("o2", "TSLA"))
        assert len(store.all_orders()) == 2

    def test_save_and_get_position(self, store):
        store.save_position(self._position())
        assert store.get_position("AAPL") is not None

    def test_update_position(self, store):
        p = self._position()
        store.save_position(p)
        p.at_breakeven = True
        store.update_position(p)
        assert store.get_position("AAPL").at_breakeven == True

    def test_remove_position(self, store):
        store.save_position(self._position())
        store.remove_position("AAPL")
        assert store.get_position("AAPL") is None

    def test_persistence_survives_reload(self, tmp_dir):
        s1 = PositionStore(data_dir=str(tmp_dir / "data"))
        s1.save_order(self._order())
        s1.save_position(self._position())

        s2 = PositionStore(data_dir=str(tmp_dir / "data"))
        assert s2.get_order("o1") is not None
        assert s2.get_position("AAPL") is not None

    def test_atomic_write_no_partial_file(self, store, tmp_dir):
        """tmp file should not exist after a successful write."""
        store.save_order(self._order())
        tmp = Path(str(tmp_dir / "data" / "pending_orders.tmp"))
        assert not tmp.exists()

    def test_corrupted_file_logs_warning(self, tmp_dir):
        d = tmp_dir / "data"
        d.mkdir(parents=True, exist_ok=True)
        (d / "pending_orders.json").write_text("not json")
        # Should not raise
        s = PositionStore(data_dir=str(d))
        assert s.all_orders() == []


# ══════════════════════════════════════════════════════════════════════════════
# ExecutionEngine — submit()
# ══════════════════════════════════════════════════════════════════════════════

class TestEngineSubmit:
    def test_submit_creates_pending_order(self, tmp_dir):
        engine, broker, journal, store = _make_engine(tmp_dir=tmp_dir)
        sig  = _make_signal()
        size = _make_size()
        pending = engine.submit(sig, size, current_price=150.0)

        assert pending is not None
        assert pending.symbol    == "AAPL"
        assert pending.direction == "long"
        assert pending.order_type == "limit"
        assert store.get_order(pending.order_id) is not None

    def test_submit_writes_journal(self, tmp_dir):
        engine, broker, journal, store = _make_engine(tmp_dir=tmp_dir)
        engine.submit(_make_signal(), _make_size(), current_price=150.0)
        entries = journal.recent(5)
        assert any(e.event == ORDER_PLACED for e in entries)

    def test_submit_invalid_size_returns_none(self, tmp_dir):
        engine, *_ = _make_engine(tmp_dir=tmp_dir)
        size = _make_size()
        size.valid = False
        size.rejection_reasons = ["stop distance is zero"]
        result = engine.submit(_make_signal(), size, current_price=150.0)
        assert result is None

    def test_submit_skips_duplicate_position(self, tmp_dir):
        engine, broker, journal, store = _make_engine(tmp_dir=tmp_dir)
        size = _make_size()
        sig  = _make_signal()
        engine.submit(sig, size, current_price=150.0)

        # Insert a fake open position for AAPL
        store.save_position(ManagedPosition(
            symbol="AAPL", direction="long", entry_price=150.0,
            qty=10.0, qty_remaining=10.0, stop_price=146.0,
        ))
        result = engine.submit(sig, size, current_price=150.0)
        assert result is None
        assert broker.place_limit_bracket.call_count == 1  # only first call

    def test_submit_skips_existing_pending(self, tmp_dir):
        engine, broker, journal, store = _make_engine(tmp_dir=tmp_dir)
        sig  = _make_signal()
        size = _make_size()
        engine.submit(sig, size, current_price=150.0)
        engine.submit(sig, size, current_price=150.0)
        assert broker.place_limit_bracket.call_count == 1

    def test_submit_fetches_quote_when_price_none(self, tmp_dir):
        broker = _make_broker(quote={"bid": 149.9, "ask": 150.1, "mid": 150.0})
        engine, _, _, _ = _make_engine(broker=broker, tmp_dir=tmp_dir)
        engine.submit(_make_signal(), _make_size())
        broker.get_latest_quote.assert_called_once_with("AAPL")

    def test_submit_uses_bid_for_short(self, tmp_dir):
        broker = _make_broker(quote={"bid": 149.8, "ask": 150.2, "mid": 150.0})
        engine, _, _, _ = _make_engine(broker=broker, tmp_dir=tmp_dir)
        sig  = _make_signal(direction="SELL")
        size = _make_size(direction="short", entry=150.0, stop=154.0)
        engine.submit(sig, size)
        call_kwargs = broker.place_limit_bracket.call_args
        assert call_kwargs.kwargs["limit_price"] == 149.8

    def test_submit_broker_error_returns_none(self, tmp_dir):
        broker = _make_broker()
        broker.place_limit_bracket.side_effect = RuntimeError("API down")
        engine, _, _, store = _make_engine(broker=broker, tmp_dir=tmp_dir)
        result = engine.submit(_make_signal(), _make_size(), current_price=150.0)
        assert result is None
        assert store.all_orders() == []

    def test_tp3_used_as_bracket_leg(self, tmp_dir):
        engine, broker, _, _ = _make_engine(tmp_dir=tmp_dir)
        size = _make_size(entry=150.0, stop=146.0)
        engine.submit(_make_signal(), size, current_price=150.0)
        call = broker.place_limit_bracket.call_args
        assert call.kwargs["tp_price"] == size.take_profits[2]


# ══════════════════════════════════════════════════════════════════════════════
# ExecutionEngine — poll() / fill handling
# ══════════════════════════════════════════════════════════════════════════════

class TestEngineFill:
    def test_filled_order_creates_position(self, tmp_dir):
        ao = _make_alpaca_order(status="filled",
                                filled_qty="10",
                                filled_avg_price="150.50")
        broker = _make_broker(alpaca_order=ao)
        engine, _, journal, store = _make_engine(broker=broker, tmp_dir=tmp_dir)

        sig  = _make_signal()
        size = _make_size()
        pending = engine.submit(sig, size, current_price=150.0)
        ao.id  = pending.order_id   # keep id consistent

        events = engine.poll()
        filled_events = [e for e in events if e.get("event") == ORDER_FILLED]
        assert len(filled_events) == 1
        assert filled_events[0]["price"] == 150.5

        pos = store.get_position("AAPL")
        assert pos is not None
        assert pos.entry_price == 150.5
        assert store.get_order(pending.order_id) is None  # removed

    def test_fill_writes_journal(self, tmp_dir):
        ao = _make_alpaca_order(status="filled",
                                filled_qty="10",
                                filled_avg_price="151.0")
        broker = _make_broker(alpaca_order=ao)
        engine, _, journal, store = _make_engine(broker=broker, tmp_dir=tmp_dir)
        pending = engine.submit(_make_signal(), _make_size(), current_price=150.0)
        ao.id = pending.order_id
        engine.poll()
        assert any(e.event == ORDER_FILLED for e in journal.recent())

    def test_partial_fill_updates_order(self, tmp_dir):
        ao = _make_alpaca_order(status="partially_filled", filled_qty="4")
        broker = _make_broker(alpaca_order=ao)
        engine, _, journal, store = _make_engine(broker=broker, tmp_dir=tmp_dir)
        pending = engine.submit(_make_signal(), _make_size(), current_price=150.0)
        ao.id = pending.order_id
        engine.poll()

        updated = store.get_order(pending.order_id)
        assert updated is not None
        assert updated.filled_qty == 4.0
        assert any(e.event == ORDER_PARTIAL for e in journal.recent())

    def test_partial_fill_no_duplicate_event(self, tmp_dir):
        """Second poll with same filled_qty should not emit another event."""
        ao = _make_alpaca_order(status="partially_filled", filled_qty="4")
        broker = _make_broker(alpaca_order=ao)
        engine, _, journal, store = _make_engine(broker=broker, tmp_dir=tmp_dir)
        pending = engine.submit(_make_signal(), _make_size(), current_price=150.0)
        ao.id = pending.order_id
        engine.poll()
        engine.poll()
        partial_events = [e for e in journal.recent() if e.event == ORDER_PARTIAL]
        assert len(partial_events) == 1


# ══════════════════════════════════════════════════════════════════════════════
# ExecutionEngine — rejection + market fallback
# ══════════════════════════════════════════════════════════════════════════════

class TestEngineRejection:
    def test_rejection_triggers_market_fallback(self, tmp_dir):
        ao = _make_alpaca_order(status="rejected")
        new_ao = _make_alpaca_order(status="new")  # market order
        broker = _make_broker(alpaca_order=ao)
        broker.place_market_bracket.return_value = new_ao
        engine, _, journal, store = _make_engine(broker=broker, tmp_dir=tmp_dir)

        pending = engine.submit(_make_signal(), _make_size(), current_price=150.0)
        ao.id = pending.order_id
        events = engine.poll()

        fallback = [e for e in events if e.get("event") == ORDER_FALLBACK]
        assert len(fallback) == 1
        broker.place_market_bracket.assert_called_once()

    def test_rejection_writes_journal_fallback(self, tmp_dir):
        ao     = _make_alpaca_order(status="rejected")
        new_ao = _make_alpaca_order(status="new")
        broker = _make_broker(alpaca_order=ao)
        broker.place_market_bracket.return_value = new_ao
        engine, _, journal, store = _make_engine(broker=broker, tmp_dir=tmp_dir)

        pending = engine.submit(_make_signal(), _make_size(), current_price=150.0)
        ao.id = pending.order_id
        engine.poll()
        events_logged = [e.event for e in journal.recent()]
        assert ORDER_REJECTED in events_logged
        assert ORDER_FALLBACK in events_logged

    def test_max_attempts_marks_failed(self, tmp_dir):
        ao = _make_alpaca_order(status="rejected")
        broker = _make_broker(alpaca_order=ao)
        # Make market order also fail on submit
        broker.place_market_bracket.side_effect = RuntimeError("still rejected")
        engine, _, journal, store = _make_engine(broker=broker, tmp_dir=tmp_dir)

        pending = engine.submit(_make_signal(), _make_size(), current_price=150.0)
        ao.id = pending.order_id
        engine.poll()
        updated = store.get_order(pending.order_id)
        assert updated is not None
        assert updated.status == "failed"
        assert any(e.event == ORDER_FAILED for e in journal.recent())


# ══════════════════════════════════════════════════════════════════════════════
# ExecutionEngine — limit timeout fallback
# ══════════════════════════════════════════════════════════════════════════════

class TestEngineLimitTimeout:
    def test_timeout_triggers_market_fallback(self, tmp_dir):
        ao     = _make_alpaca_order(status="new")
        new_ao = _make_alpaca_order(status="new")
        broker = _make_broker(alpaca_order=ao)
        broker.place_market_bracket.return_value = new_ao

        # Engine with 0-second timeout so every poll triggers fallback
        from risk.manager import RiskManager
        j = TradeJournal(path=str(tmp_dir / "j.jsonl"))
        s = PositionStore(data_dir=str(tmp_dir / "data"))
        engine = ExecutionEngine(broker, RiskManager(), j, s, limit_timeout_secs=0)

        pending = engine.submit(_make_signal(), _make_size(), current_price=150.0)
        ao.id = pending.order_id

        # Wait a moment so age > 0
        time.sleep(0.05)
        events = engine.poll()
        fallback = [e for e in events if e.get("event") == ORDER_FALLBACK]
        assert len(fallback) == 1


# ══════════════════════════════════════════════════════════════════════════════
# ExecutionEngine — virtual TP polling
# ══════════════════════════════════════════════════════════════════════════════

class TestEngineVirtualTPs:
    def _open_position(self, store, entry=150.0, stop=146.0):
        stop_dist = entry - stop
        pos = ManagedPosition(
            symbol="AAPL",
            direction="long",
            entry_price=entry,
            qty=10.0,
            qty_remaining=10.0,
            stop_price=stop,
            take_profits=[entry + stop_dist,
                          entry + 2 * stop_dist,
                          entry + 3 * stop_dist],
            tp_qtys=[4.0, 3.5, 2.5],
            alpaca_order_id="parent-id",
        )
        store.save_position(pos)
        return pos

    def test_tp1_hit_sets_breakeven(self, tmp_dir):
        broker = _make_broker(quote={"bid": 155.0, "ask": 155.0, "mid": 155.0})
        engine, _, journal, store = _make_engine(broker=broker, tmp_dir=tmp_dir)
        pos = self._open_position(store, entry=150.0, stop=146.0)
        # TP1 = 154.0, current = 155 → hit

        events = engine.poll()
        tp_events = [e for e in events if e.get("event") == TP_HIT]
        assert any(e["tp_level"] == 1 for e in tp_events)

        be_events = [e for e in events if e.get("event") == BREAKEVEN_SET]
        assert len(be_events) == 1
        assert be_events[0]["stop"] == 150.0

        updated = store.get_position("AAPL")
        assert updated.at_breakeven == True
        assert updated.stop_price   == 150.0
        assert updated.tp_hits[0]   == True

    def test_tp2_hit_reduces_qty(self, tmp_dir):
        broker = _make_broker(quote={"bid": 160.0, "ask": 160.0, "mid": 160.0})
        engine, _, journal, store = _make_engine(broker=broker, tmp_dir=tmp_dir)
        self._open_position(store, entry=150.0, stop=146.0)
        # TP1=154, TP2=158, price=160 → both hit

        engine.poll()
        updated = store.get_position("AAPL")
        assert updated.tp_hits[0] and updated.tp_hits[1]
        assert updated.qty_remaining == pytest.approx(10.0 - 4.0 - 3.5, abs=0.01)

    def test_tp_not_hit_when_below_level(self, tmp_dir):
        broker = _make_broker(quote={"bid": 152.0, "ask": 152.0, "mid": 152.0})
        engine, _, journal, store = _make_engine(broker=broker, tmp_dir=tmp_dir)
        self._open_position(store, entry=150.0, stop=146.0)
        # TP1=154 not reached yet

        engine.poll()
        updated = store.get_position("AAPL")
        assert updated.tp_hits == [False, False, False]

    def test_tp_not_double_counted(self, tmp_dir):
        broker = _make_broker(quote={"bid": 155.0, "ask": 155.0, "mid": 155.0})
        engine, _, journal, store = _make_engine(broker=broker, tmp_dir=tmp_dir)
        self._open_position(store, entry=150.0, stop=146.0)

        engine.poll()
        engine.poll()  # second poll at same price
        tp1_events = [e for e in journal.recent()
                      if e.event == TP_HIT and e.details.get("tp_level") == 1]
        assert len(tp1_events) == 1

    def test_short_tp1_hit(self, tmp_dir):
        broker = _make_broker(quote={"bid": 143.0, "ask": 143.0, "mid": 143.0})
        engine, _, journal, store = _make_engine(broker=broker, tmp_dir=tmp_dir)
        pos = ManagedPosition(
            symbol="AAPL", direction="short", entry_price=150.0,
            qty=10.0, qty_remaining=10.0, stop_price=154.0,
            take_profits=[146.0, 142.0, 138.0],
            tp_qtys=[4.0, 3.5, 2.5],
        )
        store.save_position(pos)
        engine.poll()
        updated = store.get_position("AAPL")
        assert updated.tp_hits[0] == True


# ══════════════════════════════════════════════════════════════════════════════
# ExecutionEngine — close_position()
# ══════════════════════════════════════════════════════════════════════════════

class TestEngineClosePosition:
    def test_close_removes_position_and_logs(self, tmp_dir):
        broker = _make_broker()
        engine, _, journal, store = _make_engine(broker=broker, tmp_dir=tmp_dir)

        store.save_position(ManagedPosition(
            symbol="AAPL", direction="long", entry_price=150.0,
            qty=10.0, qty_remaining=10.0, stop_price=146.0,
        ))
        result = engine.close_position("AAPL")
        assert result is True
        assert store.get_position("AAPL") is None
        assert any(e.event == POSITION_CLOSED for e in journal.recent())

    def test_close_cancels_pending_order(self, tmp_dir):
        broker = _make_broker()
        engine, _, journal, store = _make_engine(broker=broker, tmp_dir=tmp_dir)

        store.save_order(PendingOrder(
            order_id="oid-1", client_order_id="cid-1", symbol="AAPL",
            side="buy", direction="long", qty=10.0, order_type="limit",
            limit_price=150.0, stop_price=146.0,
        ))
        engine.close_position("AAPL")
        broker.cancel_order.assert_called_once_with("oid-1")
        assert store.get_order("oid-1") is None

    def test_close_nonexistent_returns_false(self, tmp_dir):
        engine, *_ = _make_engine(tmp_dir=tmp_dir)
        assert engine.close_position("NOPE") is False


# ══════════════════════════════════════════════════════════════════════════════
# Cancelled / expired order cleanup
# ══════════════════════════════════════════════════════════════════════════════

class TestEngineCancelled:
    def test_cancelled_order_removed_from_store(self, tmp_dir):
        ao = _make_alpaca_order(status="cancelled")
        broker = _make_broker(alpaca_order=ao)
        engine, _, journal, store = _make_engine(broker=broker, tmp_dir=tmp_dir)

        pending = engine.submit(_make_signal(), _make_size(), current_price=150.0)
        ao.id = pending.order_id
        engine.poll()

        assert store.get_order(pending.order_id) is None
        assert any(e.event == ORDER_CANCELLED for e in journal.recent())
