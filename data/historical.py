import logging
from datetime import datetime, timedelta, timezone

from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from data.alpaca_client import get_historical_client
from data.database import upsert_bars

log = logging.getLogger(__name__)


def fetch_forex_bars(
    instruments: list[str] | None = None,
    days: int = 365,
    granularity: str = "D",
    db_path=None,
) -> dict[str, int]:
    """
    Pull historical candles from OANDA for forex instruments and store in SQLite.
    Returns {instrument: rows_stored}.

    Instruments use OANDA format: EUR_USD, GBP_USD, USD_JPY.
    If instruments is None, reads the list from config.yaml → forex.instruments.
    """
    from data.oanda_client import fetch_forex_candles
    from data.database import DB_PATH
    if db_path is None:
        db_path = DB_PATH

    if instruments is None:
        from config.loader import get_config
        instruments = get_config().get("forex", {}).get("instruments", ["EUR_USD"])

    log.info("Fetching %d days of forex candles for %s ...", days, instruments)
    results: dict[str, int] = {}
    for inst in instruments:
        stored = fetch_forex_candles(inst, count=min(days, 500), granularity=granularity, db_path=db_path)
        results[inst] = stored
    return results


def fetch_and_store(
    symbols: list[str],
    days: int = 365,
    timeframe: TimeFrame = TimeFrame.Day,
    db_path=None,
) -> dict[str, int]:
    """
    Pull daily OHLCV bars from Alpaca for each symbol and store in SQLite.
    Returns {symbol: rows_stored}.

    Note: uses the IEX free data feed. Upgrade to "sip" feed on paid plans
    for consolidated data from all US exchanges.
    """
    from data.database import DB_PATH
    if db_path is None:
        db_path = DB_PATH

    client = get_historical_client()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=timeframe,
        start=start,
        end=end,
        adjustment="all",  # adjust for splits and dividends
        feed="iex",
    )

    log.info("Fetching %d days of bars for %s ...", days, symbols)
    bar_set = client.get_stock_bars(request)

    results: dict[str, int] = {}
    for symbol in symbols:
        bars = bar_set.data.get(symbol, [])
        rows = [
            {
                "symbol": symbol,
                "timestamp": bar.timestamp.isoformat(),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
                "timeframe": "1Day",
            }
            for bar in bars
        ]
        stored = upsert_bars(rows, db_path=db_path)
        results[symbol] = stored
        log.info("  %s: %d bars fetched, %d new rows stored", symbol, len(rows), stored)

    return results
