"""
Data loading layer for the dashboard.

Detects whether real Alpaca credentials are present:
  - DEMO MODE  — no keys; returns synthetic realistic data for every panel
  - LIVE MODE  — keys present; reads from broker, store, and journal

All public functions return plain Python dicts/lists/DataFrames so the
Streamlit layer has no direct coupling to Alpaca SDK types.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── Project root on path ──────────────────────────────────────────────────────
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from execution.journal import (
    ORDER_FILLED, POSITION_CLOSED, STOP_HIT, TP_HIT, TradeJournal,
)
from execution.store import PositionStore


# ── Helpers ───────────────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def is_demo() -> bool:
    key = os.getenv("ALPACA_API_KEY", "").strip()
    sec = os.getenv("ALPACA_SECRET_KEY", "").strip()
    return not (key and sec)


def get_trading_mode() -> str:
    """Returns 'PAPER' or 'LIVE'."""
    try:
        from config.loader import get_config
        return get_config().get("trading", {}).get("mode", "paper").upper()
    except Exception:
        return "PAPER"


# ── Broker connection ─────────────────────────────────────────────────────────

def make_broker():
    """Return an AlpacaBroker or None if in demo mode."""
    if is_demo():
        return None
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.data.historical import StockHistoricalDataClient
        from execution.broker import AlpacaBroker

        paper = get_trading_mode() == "PAPER"
        tc = TradingClient(
            os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"), paper=paper
        )
        dc = StockHistoricalDataClient(
            os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
        )
        return AlpacaBroker(tc, dc)
    except Exception:
        return None


# ── Account ───────────────────────────────────────────────────────────────────

def get_account(broker) -> dict:
    if broker is None:
        return _demo_account()
    try:
        return broker.get_account()
    except Exception:
        return _demo_account()


def _demo_account() -> dict:
    return {
        "balance":         10_245.32,
        "buying_power":    7_123.45,
        "portfolio_value": 10_245.32,
        "currency":        "USD",
    }


# ── Open positions ────────────────────────────────────────────────────────────

def get_open_positions(store: PositionStore, broker) -> list[dict]:
    """
    Merge local store with live Alpaca quotes to add current_price / unrealized_pnl.
    Falls back to demo positions when in demo mode.
    """
    if is_demo():
        return _demo_positions()

    out = []
    for pos in store.all_positions():
        d: dict[str, Any] = {
            "symbol":       pos.symbol,
            "direction":    pos.direction,
            "entry_price":  pos.entry_price,
            "qty":          pos.qty,
            "qty_remaining": pos.qty_remaining,
            "stop_price":   pos.stop_price,
            "take_profits": list(pos.take_profits),
            "tp_hits":      list(pos.tp_hits),
            "at_breakeven": pos.at_breakeven,
            "current_price": pos.entry_price,   # updated below
            "unrealized_pnl": 0.0,
            "unrealized_pct": 0.0,
        }
        if broker:
            try:
                q = broker.get_latest_quote(pos.symbol)
                d["current_price"] = q["mid"]
            except Exception:
                pass

        sign = 1 if pos.direction == "long" else -1
        d["unrealized_pnl"] = (
            pos.qty_remaining * (d["current_price"] - pos.entry_price) * sign
        )
        if pos.entry_price > 0:
            d["unrealized_pct"] = (
                (d["current_price"] - pos.entry_price) / pos.entry_price * sign * 100
            )
        out.append(d)
    return out


def _demo_positions() -> list[dict]:
    return [
        {
            "symbol":        "AAPL",
            "direction":     "long",
            "entry_price":   178.25,
            "qty":           5.6,
            "qty_remaining": 5.6,
            "stop_price":    172.80,
            "take_profits":  [183.70, 189.15, 194.60],
            "tp_hits":       [False, False, False],
            "at_breakeven":  False,
            "current_price": 180.45,
            "unrealized_pnl": 12.32,
            "unrealized_pct": 1.23,
        },
        {
            "symbol":        "MSFT",
            "direction":     "long",
            "entry_price":   408.15,
            "qty":           2.1,
            "qty_remaining": 1.26,
            "stop_price":    408.15,   # breakeven
            "take_profits":  [420.10, 432.05, 444.00],
            "tp_hits":       [True, False, False],
            "at_breakeven":  True,
            "current_price": 424.80,
            "unrealized_pnl": 21.00,
            "unrealized_pct": 4.08,
        },
    ]


# ── P&L summary ───────────────────────────────────────────────────────────────

def get_pnl_summary(journal: TradeJournal, open_positions: list[dict]) -> dict:
    """
    Compute daily / weekly / monthly / all-time realised P&L from journal,
    plus total unrealized from open positions.
    """
    if is_demo():
        return _demo_pnl()

    entries = journal.recent(10_000)
    now = _utcnow()
    periods = {
        "daily":   now - timedelta(days=1),
        "weekly":  now - timedelta(weeks=1),
        "monthly": now - timedelta(days=30),
        "alltime": datetime(2000, 1, 1, tzinfo=timezone.utc),
    }
    pnl: dict[str, float] = {k: 0.0 for k in periods}

    for entry in entries:
        if entry.event not in (TP_HIT, STOP_HIT, POSITION_CLOSED):
            continue
        try:
            ts = datetime.fromisoformat(entry.ts).replace(tzinfo=timezone.utc)
        except Exception:
            continue
        # TP_HIT entries have qty and price in details; compute partial P&L
        event_pnl = float(entry.details.get("pnl", 0.0))
        for period, cutoff in periods.items():
            if ts >= cutoff:
                pnl[period] += event_pnl

    unrealized = sum(p.get("unrealized_pnl", 0.0) for p in open_positions)
    pnl["unrealized"] = unrealized
    pnl["today_total"] = pnl["daily"] + unrealized
    return pnl


def _demo_pnl() -> dict:
    return {
        "daily":       +124.50,
        "weekly":      +387.20,
        "monthly":     +892.15,
        "alltime":    +1_245.32,
        "unrealized":   +33.32,
        "today_total": +157.82,
    }


# ── Equity curve ──────────────────────────────────────────────────────────────

def get_equity_curve(journal: TradeJournal, initial_capital: float = 10_000.0) -> pd.DataFrame:
    """Build a time-series equity curve from journal events."""
    if is_demo():
        return _demo_equity_curve()

    entries = journal.recent(100_000)
    rows = []
    equity = initial_capital
    for entry in entries:
        if entry.event not in (TP_HIT, STOP_HIT, POSITION_CLOSED):
            continue
        pnl = float(entry.details.get("pnl", 0.0))
        if pnl == 0.0:
            continue
        equity += pnl
        rows.append({"ts": entry.ts, "equity": equity, "event": entry.event})

    if not rows:
        return _demo_equity_curve()
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"])
    return df.sort_values("ts").reset_index(drop=True)


def _demo_equity_curve() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=252)
    equity = 10_000.0
    rows = []
    for d in dates:
        equity *= 1 + rng.normal(0.0005, 0.008)
        rows.append({"ts": d, "equity": round(equity, 2), "event": ""})
    return pd.DataFrame(rows)


# ── Signals ───────────────────────────────────────────────────────────────────

def get_recent_signals(symbols: list[str], broker=None) -> list[dict]:
    """
    Run signal engine on recent price data for each symbol.
    Falls back to synthetic data if live data unavailable.
    """
    if is_demo():
        return _demo_signals()

    signals = []
    for sym in symbols:
        try:
            df = _fetch_recent_bars(sym, broker, n=300)
            if df is None or df.empty:
                continue
            from signals.engine import SignalEngine
            engine = SignalEngine(threshold=65.0, min_confirmations=2)
            found = engine.scan({"D": df}, symbol=sym)
            for sig in found:
                signals.append({
                    "symbol":     sig.symbol,
                    "direction":  sig.direction,
                    "score":      sig.score,
                    "timeframe":  sig.timeframe,
                    "timestamp":  str(sig.timestamp)[:16],
                    "breakdown":  sig.breakdown,
                    "notes":      sig.notes,
                    "confs":      len(sig.confirmations),
                })
        except Exception:
            pass

    return signals or _demo_signals()


def _fetch_recent_bars(symbol: str, broker, n: int = 300) -> pd.DataFrame | None:
    """Attempt to fetch recent OHLCV bars from Alpaca data client."""
    if broker is None:
        return None
    try:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=n + 50)
        req = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
        )
        bars = broker._data.get_stock_bars(req)[symbol]
        records = []
        for b in bars:
            records.append({
                "open": float(b.open), "high": float(b.high),
                "low": float(b.low), "close": float(b.close),
                "volume": float(b.volume),
            })
            idx = [b.timestamp for b in bars]
        df = pd.DataFrame(records, index=pd.DatetimeIndex(idx))
        return df.sort_index()
    except Exception:
        return None


def _demo_signals() -> list[dict]:
    return [
        {
            "symbol":    "SPY",
            "direction": "BUY",
            "score":     79.3,
            "timeframe": "D",
            "timestamp": "2024-01-15 16:00",
            "breakdown": {
                "Trend":      {"bullish": 42.5, "bearish": 10.0},
                "Momentum":   {"bullish": 15.0, "bearish":  4.0},
                "Volatility": {"bullish": 12.5, "bearish":  2.5},
                "Patterns":   {"bullish":  8.0, "bearish":  0.0},
                "Multi-TF":   {"bullish":  6.0},
            },
            "notes": "EMA stack aligned, MACD crossover, RSI recovering from oversold",
            "confs": 7,
        },
        {
            "symbol":    "NVDA",
            "direction": "BUY",
            "score":     76.8,
            "timeframe": "4H",
            "timestamp": "2024-01-15 12:00",
            "breakdown": {
                "Trend":      {"bullish": 38.0, "bearish":  8.0},
                "Momentum":   {"bullish": 18.0, "bearish":  3.0},
                "Volatility": {"bullish": 14.0, "bearish":  2.0},
                "Patterns":   {"bullish":  6.0, "bearish":  0.0},
                "Multi-TF":   {"bullish":  3.0},
            },
            "notes": "Breakout above ascending triangle",
            "confs": 6,
        },
    ]


# ── Price data with indicators ────────────────────────────────────────────────

def get_price_with_indicators(
    symbol: str,
    broker=None,
    n_bars: int = 120,
) -> pd.DataFrame:
    """
    Return recent OHLCV + computed indicators for a symbol.
    Falls back to synthetic data in demo mode.
    """
    df = None
    if broker is not None:
        df = _fetch_recent_bars(symbol, broker, n=n_bars + 250)

    if df is None or df.empty:
        df = _synthetic_bars(symbol, n_bars + 250)

    try:
        from indicators.combined import run_all
        return run_all(df).tail(n_bars)
    except Exception:
        return df.tail(n_bars)


def _synthetic_bars(symbol: str, n: int) -> pd.DataFrame:
    seed = sum(ord(c) for c in symbol)
    rng = np.random.default_rng(seed)
    price = 150.0 + rng.integers(0, 200)
    closes, opens, highs, lows, vols = [], [], [], [], []
    for _ in range(n):
        price = price * (1 + 0.0003 + rng.normal(0, 0.010))
        o = price * (1 + rng.normal(0, 0.002))
        h = max(o, price) * (1 + abs(rng.normal(0, 0.004)))
        l = min(o, price) * (1 - abs(rng.normal(0, 0.004)))
        closes.append(price); opens.append(o)
        highs.append(h); lows.append(l)
        vols.append(int(abs(rng.normal(5_000_000, 1_000_000))))
    idx = pd.bdate_range(end=pd.Timestamp.today(), periods=n)
    return pd.DataFrame({"open": opens, "high": highs, "low": lows,
                          "close": closes, "volume": vols}, index=idx)


# ── Trade history ─────────────────────────────────────────────────────────────

def get_trade_history(journal: TradeJournal) -> list[dict]:
    if is_demo():
        return _demo_trade_history()

    entries = journal.recent(500)
    rows = []
    for e in entries:
        if e.event not in (ORDER_FILLED, TP_HIT, STOP_HIT, POSITION_CLOSED):
            continue
        rows.append({
            "time":      e.ts[:19],
            "event":     e.event,
            "symbol":    e.symbol,
            "direction": e.direction,
            "qty":       e.qty,
            "price":     e.price,
            "order_id":  e.order_id[:8] + "…" if e.order_id else "",
        })
    return rows or _demo_trade_history()


def _demo_trade_history() -> list[dict]:
    rng = np.random.default_rng(7)
    symbols = ["AAPL", "MSFT", "SPY", "NVDA", "TSLA"]
    events = [ORDER_FILLED, TP_HIT, TP_HIT, STOP_HIT, POSITION_CLOSED]
    rows = []
    t = datetime.now(timezone.utc) - timedelta(days=30)
    for _ in range(40):
        t += timedelta(hours=int(rng.integers(4, 48)))
        sym = symbols[int(rng.integers(0, len(symbols)))]
        ev  = events[int(rng.integers(0, len(events)))]
        rows.append({
            "time":      t.strftime("%Y-%m-%d %H:%M"),
            "event":     ev,
            "symbol":    sym,
            "direction": "long" if rng.random() > 0.35 else "short",
            "qty":       round(float(rng.uniform(1, 10)), 2),
            "price":     round(float(rng.uniform(100, 500)), 2),
            "order_id":  "demo1234…",
        })
    return rows


# ── Risk metrics ──────────────────────────────────────────────────────────────

def get_risk_metrics(equity_curve_df: pd.DataFrame, initial_capital: float = 10_000.0) -> dict:
    """Compute Sharpe, Sortino, max DD, etc. from equity curve."""
    if equity_curve_df.empty:
        return {}
    vals = equity_curve_df["equity"].values
    if len(vals) < 2:
        return {}
    rets = np.diff(vals) / vals[:-1]
    mean_r = float(np.mean(rets))
    std_r  = float(np.std(rets, ddof=1)) if len(rets) > 1 else 1e-9
    down   = rets[rets < 0]
    down_std = float(np.std(down, ddof=1)) if len(down) > 1 else abs(float(np.mean(down))) if len(down) else 1e-9
    ann = np.sqrt(252)
    sharpe  = (mean_r / std_r * ann) if std_r > 0 else 0.0
    sortino = (mean_r / down_std * ann) if down_std > 0 else 0.0
    # Max drawdown
    peak = vals[0]
    max_dd = 0.0
    for v in vals:
        peak = max(peak, v)
        dd = (peak - v) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)
    total_return = (vals[-1] / initial_capital - 1) * 100
    return {
        "Sharpe Ratio":   round(sharpe, 3),
        "Sortino Ratio":  round(sortino, 3),
        "Max Drawdown":   f"{max_dd*100:.1f}%",
        "Total Return":   f"{total_return:+.1f}%",
        "Final Equity":   f"${vals[-1]:,.2f}",
        "Avg Daily Ret":  f"{mean_r*100:.3f}%",
        "Daily Vol":      f"{std_r*100:.3f}%",
        "Ann. Vol":       f"{std_r*ann*100:.1f}%",
    }


# ── Close position ────────────────────────────────────────────────────────────

def close_position(symbol: str, broker, store: PositionStore) -> bool:
    """Cancel open orders and remove local position record."""
    if is_demo():
        return True   # pretend it worked
    try:
        from execution.engine import ExecutionEngine
        from execution.journal import TradeJournal
        from risk.manager import RiskManager
        j = TradeJournal("logs/trade_journal.jsonl")
        engine = ExecutionEngine(broker, RiskManager(), j, store)
        return engine.close_position(symbol)
    except Exception:
        return False
