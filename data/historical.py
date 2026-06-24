import logging
from datetime import datetime, timedelta, timezone

from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from data.alpaca_client import get_historical_client
from data.database import upsert_bars

log = logging.getLogger(__name__)


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
