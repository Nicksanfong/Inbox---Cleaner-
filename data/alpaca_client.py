import logging
import os

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.trading.client import TradingClient

log = logging.getLogger(__name__)


def _require_keys() -> tuple[str, str]:
    key = os.getenv("ALPACA_API_KEY", "")
    secret = os.getenv("ALPACA_API_SECRET", "")
    if not key or not secret:
        raise EnvironmentError(
            "ALPACA_API_KEY and ALPACA_API_SECRET must be set in your .env file.\n"
            "Copy .env.example to .env and add your Alpaca paper trading keys."
        )
    return key, secret


def get_historical_client() -> StockHistoricalDataClient:
    key, secret = _require_keys()
    return StockHistoricalDataClient(key, secret)


def get_trading_client(paper: bool = True) -> TradingClient:
    key, secret = _require_keys()
    return TradingClient(key, secret, paper=paper)
