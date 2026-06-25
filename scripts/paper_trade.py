"""
End-to-end paper trade walkthrough.

Usage:
    python scripts/paper_trade.py

Requires ALPACA_API_KEY and ALPACA_SECRET_KEY in environment (or .env file).
Uses paper trading endpoint — no real money.

Walkthrough steps:
  1. Connect to Alpaca paper account and show balance
  2. Generate a synthetic AAPL signal (score ~82)
  3. Run the full risk approval pipeline
  4. Place a real bracket limit order on Alpaca paper
  5. Poll once to show order status
  6. Print the trade journal tail
  7. Show the persisted pending_orders.json
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

# ── load .env ─────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("paper_trade")

ALPACA_KEY    = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY")

if not ALPACA_KEY or not ALPACA_SECRET:
    sys.exit(
        "ERROR: Set ALPACA_API_KEY and ALPACA_SECRET_KEY in your environment "
        "or .env file before running this script."
    )

# ── imports ───────────────────────────────────────────────────────────────────
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient

from execution.broker  import AlpacaBroker
from execution.engine  import ExecutionEngine
from execution.journal import TradeJournal
from execution.store   import PositionStore
from risk.manager      import RiskManager
from risk.models       import RiskConfig
from signals.models    import Signal


# ── helpers ───────────────────────────────────────────────────────────────────

def banner(title: str) -> None:
    print("\n" + "─" * 60)
    print(f"  {title}")
    print("─" * 60)


def pprint(label: str, value) -> None:
    if isinstance(value, dict):
        print(f"  {label}:")
        for k, v in value.items():
            print(f"    {k}: {v}")
    else:
        print(f"  {label}: {value}")


# ── main walkthrough ──────────────────────────────────────────────────────────

def main() -> None:

    # ── Step 1: Connect ───────────────────────────────────────────────────────
    banner("Step 1 — Connect to Alpaca paper account")
    trading_client = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=True)
    data_client    = StockHistoricalDataClient(ALPACA_KEY, ALPACA_SECRET)
    broker         = AlpacaBroker(trading_client, data_client)

    account = broker.get_account()
    pprint("Account", account)

    # ── Step 2: Get a real-time quote ─────────────────────────────────────────
    banner("Step 2 — Fetch latest AAPL quote")
    symbol = "AAPL"
    try:
        quote = broker.get_latest_quote(symbol)
        pprint("Quote", quote)
        ask_price = quote["ask"]
    except Exception as exc:
        log.warning("Quote fetch failed (%s) — using fallback price 175.00", exc)
        ask_price = 175.00
        quote = {"bid": ask_price - 0.05, "ask": ask_price, "mid": ask_price}
        pprint("Quote (fallback)", quote)

    # ── Step 3: Build a synthetic signal ─────────────────────────────────────
    banner("Step 3 — Generate synthetic BUY signal for AAPL")
    import pandas as pd
    sig = Signal(
        symbol=symbol,
        direction="BUY",
        score=82.4,
        timeframe="15m",
        timestamp=pd.Timestamp.now(),
        notes="Synthetic demo signal — 7-TF alignment, EMA stack + MACD bullish",
    )
    print(f"  {sig}")

    # ── Step 4: Risk approval ─────────────────────────────────────────────────
    banner("Step 4 — Run risk management approval")
    cfg = RiskConfig(
        risk_pct=0.01,          # 1% of account per trade
        min_rr=2.0,
        allow_fractional_units=True,
    )
    rm = RiskManager(cfg)

    entry_price = ask_price
    stop_price  = round(entry_price * 0.97, 2)    # 3% stop
    balance     = account["balance"]

    size = rm.approve_trade(
        entry_price=entry_price,
        stop_price=stop_price,
        account_balance=balance,
        day_start_balance=balance,
        direction="long",
        symbol=symbol,
    )

    if not size.valid:
        print(f"  REJECTED: {size.rejection_reasons}")
        print("  (Continuing demo anyway — placing a minimum-size order)")
        size.valid = True
        size.units = 1.0

    pprint("Size", {
        "entry":       size.entry_price,
        "stop":        size.stop_price,
        "units":       round(size.units, 4),
        "risk_amount": f"${size.risk_amount:.2f}",
        "r:r ratio":   f"1:{size.rr_ratio}",
        "TP1":         size.take_profits[0],
        "TP2":         size.take_profits[1],
        "TP3":         size.take_profits[2],
    })

    # ── Step 5: Wire up execution layer ──────────────────────────────────────
    banner("Step 5 — Initialise execution engine")
    journal = TradeJournal("logs/trade_journal.jsonl")
    store   = PositionStore("data")
    engine  = ExecutionEngine(
        broker=broker,
        risk_manager=rm,
        journal=journal,
        store=store,
        limit_timeout_secs=60,
        max_attempts=2,
    )
    print("  Engine ready.")

    # ── Step 6: Submit order ──────────────────────────────────────────────────
    banner("Step 6 — Submit bracket limit order to Alpaca paper")

    # Use 1 share minimum for demo to avoid over-buying
    size.units = max(1.0, min(size.units, 5.0))

    pending = engine.submit(sig, size, current_price=ask_price)
    if pending is None:
        print("  Order not submitted (check logs above).")
        return

    print(f"\n  Pending order created:")
    pprint("  order_id",    pending.order_id)
    pprint("  client_id",   pending.client_order_id)
    pprint("  symbol",      pending.symbol)
    pprint("  direction",   pending.direction)
    pprint("  qty",         pending.qty)
    pprint("  limit_price", pending.limit_price)
    pprint("  stop_price",  pending.stop_price)
    pprint("  take_profits",pending.take_profits)

    # ── Step 7: Poll for status ───────────────────────────────────────────────
    banner("Step 7 — Poll order status (1 check)")
    time.sleep(1)   # give Alpaca a moment to register the order
    events = engine.poll()
    if events:
        for ev in events:
            print(f"  Event: {ev}")
    else:
        alpaca_order = broker.get_order(pending.order_id)
        status = alpaca_order.status.value if alpaca_order else "unknown"
        print(f"  Order status on Alpaca: {status}")
        print("  (Limit order is sitting open — this is expected.)")

    # ── Step 8: Show persisted state ─────────────────────────────────────────
    banner("Step 8 — Persisted state (data/pending_orders.json)")
    orders_path = Path("data/pending_orders.json")
    if orders_path.exists():
        data = json.loads(orders_path.read_text())
        print(json.dumps(data, indent=2))
    else:
        print("  (file not written yet)")

    # ── Step 9: Journal tail ──────────────────────────────────────────────────
    banner("Step 9 — Trade journal tail (last 5 entries)")
    for entry in journal.recent(5):
        print(f"  [{entry.ts[:19]}]  {entry.event:<20}  {entry.symbol}  "
              f"qty={entry.qty}  price={entry.price}  id={entry.order_id[:8]}...")

    # ── Summary ───────────────────────────────────────────────────────────────
    banner("Done")
    print(f"""
  What just happened:
    1. Connected to Alpaca paper account (${balance:,.2f} equity).
    2. Fetched live {symbol} quote → ask = ${ask_price:.2f}.
    3. Built a synthetic BUY signal (score = {sig.score}).
    4. Risk manager sized the trade:
         {size.units} shares @ ${entry_price:.2f}
         Stop: ${stop_price:.2f}  ({round(abs(entry_price-stop_price)/entry_price*100,1)}% away)
         TP1 / TP2 / TP3: ${size.take_profits[0]:.2f} / ${size.take_profits[1]:.2f} / ${size.take_profits[2]:.2f}
    5. ExecutionEngine placed a BRACKET LIMIT order on Alpaca.
         - Stop-loss leg: attached to bracket.
         - Take-profit leg: set at TP3 ({size.take_profits[2]:.2f}).
         - TP1 and TP2 are tracked locally and fire when price is polled.
    6. Order ID {pending.order_id} is in data/pending_orders.json.
    7. Full event log in logs/trade_journal.jsonl.

  To cancel the order:
    python -c "
from alpaca.trading.client import TradingClient
import os
c = TradingClient(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_SECRET_KEY'), paper=True)
c.cancel_order_by_id('{pending.order_id}')
print('cancelled')
"
""")


if __name__ == "__main__":
    main()
