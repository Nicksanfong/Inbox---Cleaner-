#!/usr/bin/env python3
"""
Live price monitor — prints real-time bid/ask quotes for configured symbols.

Quotes only flow during US market hours (9:30 am – 4:00 pm ET, Mon–Fri).
Outside those hours the script will connect successfully but show the
heartbeat warning after 30 seconds — that is normal.

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

# Make sure the project root is on the Python path when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from config.loader import load_config
from config.logger import setup_logging
from data.websocket_manager import WebSocketManager

load_dotenv()
load_config()
setup_logging()

log = logging.getLogger(__name__)

# EUR/USD is not available on Alpaca (forex only); use stock symbols instead.
# For crypto, add e.g. "BTC/USD" via CryptoDataStream (separate setup).
SYMBOLS = ["SPY", "AAPL", "MSFT"]


def _fmt_quote(quote) -> str:
    now    = datetime.now().strftime("%H:%M:%S")
    bid    = quote.bid_price or 0.0
    ask    = quote.ask_price or 0.0
    spread = ask - bid
    return (
        f"[{now}]  {quote.symbol:<6}  "
        f"bid=${bid:<9.3f}  "
        f"ask=${ask:<9.3f}  "
        f"spread=${spread:.4f}"
    )


async def _on_quote(quote) -> None:
    print(_fmt_quote(quote), flush=True)


async def _main() -> None:
    api_key    = os.getenv("ALPACA_API_KEY", "")
    api_secret = os.getenv("ALPACA_API_SECRET", "")

    if not api_key or not api_secret:
        print(
            "\nERROR: Missing API keys.\n"
            "Open the file  .env  and set:\n"
            "    ALPACA_API_KEY=<your key>\n"
            "    ALPACA_API_SECRET=<your secret>\n"
            "Get free paper-trading keys at https://app.alpaca.markets\n"
        )
        sys.exit(1)

    print(f"\n{'='*58}")
    print(f"  Live Price Monitor  —  {SYMBOLS}")
    print(f"  Quotes flow during market hours: 9:30am–4pm ET, Mon–Fri")
    print(f"  Press Ctrl+C to stop.")
    print(f"{'='*58}\n")

    manager = WebSocketManager(api_key, api_secret, SYMBOLS)
    manager.subscribe(_on_quote)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, loop.stop)

    await manager.run()


if __name__ == "__main__":
    asyncio.run(_main())
