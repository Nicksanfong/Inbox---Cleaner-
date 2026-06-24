import logging
import os

import oandapyV20
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.instruments as instruments

log = logging.getLogger(__name__)


def _require_oanda_creds() -> tuple[str, str, str]:
    token   = os.getenv("OANDA_API_TOKEN", "")
    acct_id = os.getenv("OANDA_ACCOUNT_ID", "")
    env     = os.getenv("OANDA_ENVIRONMENT", "practice")
    if not token or not acct_id:
        raise EnvironmentError(
            "OANDA_API_TOKEN and OANDA_ACCOUNT_ID must be set in your .env file.\n"
            "Create a free demo account at oanda.com → My Account → API Access."
        )
    return token, acct_id, env


def get_oanda_api() -> tuple[oandapyV20.API, str]:
    """Return (api_client, account_id). Environment is 'practice' by default."""
    token, acct_id, env = _require_oanda_creds()
    api = oandapyV20.API(access_token=token, environment=env)
    return api, acct_id


def fetch_forex_candles(
    instrument: str,
    count: int = 365,
    granularity: str = "D",
    db_path=None,
) -> int:
    """
    Fetch up to `count` daily candles for an OANDA instrument (e.g. "EUR_USD")
    and store them in SQLite. Returns the number of new rows stored.

    OANDA instruments use underscores: EUR_USD, GBP_USD, USD_JPY, etc.
    Maximum count per request is 500 for granularity D.
    """
    from data.database import DB_PATH, upsert_bars
    if db_path is None:
        db_path = DB_PATH

    api, _ = get_oanda_api()

    params = {
        "count": min(count, 500),
        "granularity": granularity,
        "price": "M",        # midpoint (average of bid and ask)
    }
    r = instruments.InstrumentsCandles(instrument=instrument, params=params)
    api.request(r)

    candles = r.response.get("candles", [])
    rows = []
    for c in candles:
        if not c.get("complete", False):
            continue   # skip the current (incomplete) candle
        mid = c.get("mid", {})
        rows.append({
            "symbol":    instrument,
            "timestamp": c["time"],
            "open":      float(mid["o"]),
            "high":      float(mid["h"]),
            "low":       float(mid["l"]),
            "close":     float(mid["c"]),
            "volume":    float(c.get("volume", 0)),
            "timeframe": f"1{granularity}",
        })

    stored = upsert_bars(rows, db_path=db_path)
    log.info("  %s: %d candles fetched, %d new rows stored", instrument, len(rows), stored)
    return stored
