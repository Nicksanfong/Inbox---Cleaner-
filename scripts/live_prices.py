#!/usr/bin/env python3
"""
Live price monitor — prints real-time bid/ask prices for:
  • Stocks (SPY, AAPL, MSFT) via Alpaca WebSocket
  • Forex (EUR_USD, GBP_USD, USD_JPY) via OANDA HTTP stream

Both streams run concurrently in the same event loop.

Market hours for quotes:
  • Stocks: 9:30am–4pm ET, Mon–Fri
  • Forex:  ~24/5 (closed Fri 5pm – Sun 5pm ET)

Usage:
    python scripts/live_prices.py
"""
import asyncio
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from config.loader import load_config
from config.logger import setup_logging

load_dotenv()
load_config()
setup_logging()

log = logging.getLogger(__name__)

STOCK_SYMBOLS   = ["SPY", "AAPL", "MSFT"]
FOREX_INSTRUMENTS = ["EUR_USD", "GBP_USD", "USD_JPY"]

HEADER = "\033[1m"   # bold
RESET  = "\033[0m"
GREEN  = "\033[32m"
CYAN   = "\033[36m"


# ── Formatters ─────────────────────────────────────────────────────────────────

def _fmt_stock_quote(quote) -> str:
    now    = datetime.now().strftime("%H:%M:%S")
    bid    = quote.bid_price or 0.0
    ask    = quote.ask_price or 0.0
    spread = ask - bid
    return (
        f"[{now}]  {GREEN}STOCK{RESET}  {quote.symbol:<6}  "
        f"bid=${bid:<9.3f}  ask=${ask:<9.3f}  spread=${spread:.4f}"
    )


def _fmt_forex_tick(tick: dict) -> str:
    now        = datetime.now().strftime("%H:%M:%S")
    instrument = tick.get("instrument", "???")
    bids       = tick.get("bids", [{}])
    asks       = tick.get("asks", [{}])
    bid        = float(bids[0].get("price", 0)) if bids else 0.0
    ask        = float(asks[0].get("price", 0)) if asks else 0.0
    spread     = ask - bid
    return (
        f"[{now}]  {CYAN}FOREX{RESET}  {instrument:<8}  "
        f"bid={bid:<10.5f}  ask={ask:<10.5f}  spread={spread:.5f}"
    )


# ── Callbacks ──────────────────────────────────────────────────────────────────

async def _on_stock_quote(quote) -> None:
    print(_fmt_stock_quote(quote), flush=True)


async def _on_forex_tick(tick: dict) -> None:
    print(_fmt_forex_tick(tick), flush=True)


# ── Main ───────────────────────────────────────────────────────────────────────

async def _main() -> None:
    alpaca_key    = os.getenv("ALPACA_API_KEY", "")
    alpaca_secret = os.getenv("ALPACA_API_SECRET", "")
    oanda_token   = os.getenv("OANDA_API_TOKEN", "")
    oanda_acct    = os.getenv("OANDA_ACCOUNT_ID", "")

    tasks = []
    active_feeds: list[str] = []

    if alpaca_key and alpaca_secret:
        from data.websocket_manager import WebSocketManager
        stock_mgr = WebSocketManager(alpaca_key, alpaca_secret, STOCK_SYMBOLS)
        stock_mgr.subscribe(_on_stock_quote)
        tasks.append(asyncio.create_task(stock_mgr.run(), name="alpaca-stocks"))
        active_feeds.append(f"Stocks {STOCK_SYMBOLS}  (Alpaca)")
    else:
        print("  [STOCKS ] Skipped — set ALPACA_API_KEY + ALPACA_API_SECRET in .env")

    if oanda_token and oanda_acct:
        from data.oanda_stream import OandaStreamManager
        forex_mgr = OandaStreamManager(oanda_token, oanda_acct, FOREX_INSTRUMENTS)
        forex_mgr.subscribe(_on_forex_tick)
        tasks.append(asyncio.create_task(forex_mgr.run(), name="oanda-forex"))
        active_feeds.append(f"Forex  {FOREX_INSTRUMENTS}  (OANDA)")
    else:
        print("  [FOREX  ] Skipped — set OANDA_API_TOKEN + OANDA_ACCOUNT_ID in .env")

    if not tasks:
        print(
            "\nNo API keys found. Copy .env.example to .env and add your keys.\n"
            "  Alpaca (stocks): https://app.alpaca.markets\n"
            "  OANDA   (forex): https://www.oanda.com/register/#demo\n"
        )
        sys.exit(1)

    print(f"\n{'='*65}")
    print(f"  Live Price Monitor")
    for feed in active_feeds:
        print(f"  •  {feed}")
    print(f"  Stocks: 9:30am–4pm ET, Mon–Fri")
    print(f"  Forex:  ~24/5 (closed Fri 5pm – Sun 5pm ET)")
    print(f"  Press Ctrl+C to stop.")
    print(f"{'='*65}\n")

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, loop.stop)

    await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == "__main__":
    asyncio.run(_main())
