"""
Tests for the data layer.
All Alpaca API calls are mocked so these tests run without real API keys.
"""
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from data.database import fetch_bars, get_bar_count, init_db, upsert_bars


# ── Database tests ────────────────────────────────────────────────────────────

@pytest.fixture()
def db(tmp_path: Path) -> Path:
    path = tmp_path / "test.db"
    init_db(path)
    return path


SAMPLE_ROWS = [
    {
        "symbol": "SPY", "timestamp": "2024-01-02T00:00:00+00:00",
        "open": 470.0, "high": 475.0, "low": 469.0, "close": 473.5,
        "volume": 82_000_000, "timeframe": "1Day",
    },
    {
        "symbol": "SPY", "timestamp": "2024-01-03T00:00:00+00:00",
        "open": 473.5, "high": 478.0, "low": 472.0, "close": 476.0,
        "volume": 75_000_000, "timeframe": "1Day",
    },
]


def test_init_creates_db_file(tmp_path: Path) -> None:
    path = tmp_path / "new.db"
    init_db(path)
    assert path.exists()


def test_upsert_bars_inserts_rows(db: Path) -> None:
    n = upsert_bars(SAMPLE_ROWS, db_path=db)
    assert n == 2


def test_upsert_bars_ignores_duplicates(db: Path) -> None:
    upsert_bars(SAMPLE_ROWS, db_path=db)
    n2 = upsert_bars(SAMPLE_ROWS, db_path=db)  # same rows again
    assert n2 == 0


def test_fetch_bars_returns_rows(db: Path) -> None:
    upsert_bars(SAMPLE_ROWS, db_path=db)
    rows = fetch_bars("SPY", limit=10, db_path=db)
    assert len(rows) == 2
    assert rows[0]["symbol"] == "SPY"


def test_fetch_bars_limit(db: Path) -> None:
    upsert_bars(SAMPLE_ROWS, db_path=db)
    rows = fetch_bars("SPY", limit=1, db_path=db)
    assert len(rows) == 1


def test_fetch_bars_newest_first(db: Path) -> None:
    upsert_bars(SAMPLE_ROWS, db_path=db)
    rows = fetch_bars("SPY", db_path=db)
    # rows[0] should be the later date
    assert rows[0]["timestamp"] > rows[1]["timestamp"]


def test_get_bar_count(db: Path) -> None:
    upsert_bars(SAMPLE_ROWS, db_path=db)
    assert get_bar_count("SPY", db_path=db) == 2
    assert get_bar_count("AAPL", db_path=db) == 0


def test_upsert_empty_list_is_noop(db: Path) -> None:
    assert upsert_bars([], db_path=db) == 0


# ── Historical fetch tests (mocked Alpaca client) ─────────────────────────────

def _make_mock_bar(ts: str, o=100.0, h=105.0, l=99.0, c=103.0, v=1_000_000):
    bar = MagicMock()
    bar.timestamp = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
    bar.open, bar.high, bar.low, bar.close, bar.volume = o, h, l, c, v
    return bar


def test_fetch_and_store_saves_bars(tmp_path: Path) -> None:
    db = tmp_path / "hist.db"
    init_db(db)

    mock_bar = _make_mock_bar("2024-06-01T00:00:00")
    mock_bar_set = MagicMock()
    mock_bar_set.data = {"AAPL": [mock_bar]}

    mock_client = MagicMock()
    mock_client.get_stock_bars.return_value = mock_bar_set

    with patch("data.historical.get_historical_client", return_value=mock_client):
        from data.historical import fetch_and_store
        results = fetch_and_store(["AAPL"], days=30, db_path=db)

    assert results["AAPL"] == 1
    assert get_bar_count("AAPL", db_path=db) == 1


def test_fetch_and_store_handles_empty_response(tmp_path: Path) -> None:
    db = tmp_path / "empty.db"
    init_db(db)

    mock_bar_set = MagicMock()
    mock_bar_set.data = {}  # Alpaca returned no data for symbol

    mock_client = MagicMock()
    mock_client.get_stock_bars.return_value = mock_bar_set

    with patch("data.historical.get_historical_client", return_value=mock_client):
        from data.historical import fetch_and_store
        results = fetch_and_store(["XYZ"], days=30, db_path=db)

    assert results.get("XYZ", 0) == 0


# ── WebSocket manager tests ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_websocket_dispatches_to_callbacks() -> None:
    from data.websocket_manager import WebSocketManager

    received = []

    async def my_callback(quote):
        received.append(quote)

    manager = WebSocketManager("key", "secret", ["SPY"])
    manager.subscribe(my_callback)

    fake_quote = MagicMock()
    fake_quote.symbol = "SPY"
    fake_quote.bid_price = 450.0
    fake_quote.ask_price = 450.01

    await manager._dispatch(fake_quote)

    assert len(received) == 1
    assert received[0].symbol == "SPY"


@pytest.mark.asyncio
async def test_websocket_dispatches_sync_callbacks() -> None:
    from data.websocket_manager import WebSocketManager

    received = []

    def sync_callback(quote):   # plain function, not async
        received.append(quote)

    manager = WebSocketManager("key", "secret", ["AAPL"])
    manager.subscribe(sync_callback)

    fake_quote = MagicMock()
    await manager._dispatch(fake_quote)
    assert len(received) == 1


# ── OANDA client tests ────────────────────────────────────────────────────────

def test_fetch_forex_candles_stores_bars(tmp_path: Path) -> None:
    db = tmp_path / "forex.db"
    init_db(db)

    fake_response = {
        "candles": [
            {
                "complete": True,
                "volume": 12345,
                "time": "2024-06-01T00:00:00.000000000Z",
                "mid": {"o": "1.08500", "h": "1.09000", "l": "1.08200", "c": "1.08750"},
            },
            {
                "complete": True,
                "volume": 9876,
                "time": "2024-06-02T00:00:00.000000000Z",
                "mid": {"o": "1.08750", "h": "1.09200", "l": "1.08500", "c": "1.09100"},
            },
            {   # incomplete candle — should be skipped
                "complete": False,
                "volume": 100,
                "time": "2024-06-03T00:00:00.000000000Z",
                "mid": {"o": "1.09100", "h": "1.09100", "l": "1.09100", "c": "1.09100"},
            },
        ]
    }

    mock_api = MagicMock()
    mock_request_instance = MagicMock()
    mock_request_instance.response = fake_response

    with (
        patch("data.oanda_client.get_oanda_api", return_value=(mock_api, "acct-123")),
        patch("data.oanda_client.instruments.InstrumentsCandles", return_value=mock_request_instance),
    ):
        from data.oanda_client import fetch_forex_candles
        stored = fetch_forex_candles("EUR_USD", count=10, db_path=db)

    assert stored == 2   # only complete candles stored
    assert get_bar_count("EUR_USD", db_path=db) == 2


def test_fetch_forex_candles_skips_duplicates(tmp_path: Path) -> None:
    db = tmp_path / "forex_dup.db"
    init_db(db)

    fake_response = {
        "candles": [
            {
                "complete": True, "volume": 1000,
                "time": "2024-06-01T00:00:00.000000000Z",
                "mid": {"o": "1.08", "h": "1.09", "l": "1.07", "c": "1.085"},
            }
        ]
    }

    mock_api = MagicMock()
    mock_request_instance = MagicMock()
    mock_request_instance.response = fake_response

    with (
        patch("data.oanda_client.get_oanda_api", return_value=(mock_api, "acct-123")),
        patch("data.oanda_client.instruments.InstrumentsCandles", return_value=mock_request_instance),
    ):
        from data.oanda_client import fetch_forex_candles
        first  = fetch_forex_candles("EUR_USD", count=10, db_path=db)
        second = fetch_forex_candles("EUR_USD", count=10, db_path=db)

    assert first  == 1
    assert second == 0   # duplicate ignored


# ── OANDA stream manager tests ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_oanda_stream_dispatches_price_tick() -> None:
    from data.oanda_stream import OandaStreamManager

    received = []

    async def on_tick(tick):
        received.append(tick)

    manager = OandaStreamManager("token", "acct-123", ["EUR_USD"])
    manager.subscribe(on_tick)

    tick = {
        "type": "PRICE",
        "instrument": "EUR_USD",
        "bids": [{"price": "1.08500", "liquidity": 10_000_000}],
        "asks": [{"price": "1.08502", "liquidity": 10_000_000}],
        "tradeable": True,
        "time": "2024-06-01T10:30:00.000000000Z",
    }
    await manager._dispatch(tick)

    assert len(received) == 1
    assert received[0]["instrument"] == "EUR_USD"


@pytest.mark.asyncio
async def test_oanda_stream_dispatches_sync_callback() -> None:
    from data.oanda_stream import OandaStreamManager

    received = []

    def sync_cb(tick):
        received.append(tick)

    manager = OandaStreamManager("token", "acct-123", ["GBP_USD"])
    manager.subscribe(sync_cb)

    tick = {"type": "PRICE", "instrument": "GBP_USD",
            "bids": [{"price": "1.27000"}], "asks": [{"price": "1.27002"}]}
    await manager._dispatch(tick)
    assert len(received) == 1
