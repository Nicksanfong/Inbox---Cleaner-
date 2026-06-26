"""
Backtest runner — runs one strategy over a chosen symbol and date range.

Usage
─────
    python scripts/run_backtest.py                          # AAPL daily, 3 years
    python scripts/run_backtest.py --symbol MSFT            # change symbol
    python scripts/run_backtest.py --tf D --start 2021-01-01 --end 2024-01-01
    python scripts/run_backtest.py --symbol SPY --threshold 60 --atr-mult 1.0
    python scripts/run_backtest.py --symbol TSLA --multi-tf  # add weekly TF

Options
───────
    --symbol    STR    Ticker (default AAPL)
    --tf        STR    Entry timeframe: D | W | (default D)
    --start     DATE   Start date YYYY-MM-DD (default 3 years ago)
    --end       DATE   End date YYYY-MM-DD (default today)
    --capital   FLOAT  Initial capital (default 10000)
    --threshold FLOAT  Confluence score threshold (default 65)
    --atr-mult  FLOAT  ATR multiplier for stop distance (default 1.5)
    --risk-pct  FLOAT  Fraction of capital risked per trade (default 0.01)
    --multi-tf         Include weekly timeframe for MTF scoring
    --no-chart         Skip chart generation
    --synthetic        Use generated synthetic data (no network required)
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Ensure project root is on sys.path when the script is run directly
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# ── Argument parsing ──────────────────────────────────────────────────────────

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a strategy backtest over historical data."
    )
    p.add_argument("--symbol",    default="AAPL",  type=str)
    p.add_argument("--tf",        default="D",     type=str)
    p.add_argument("--start",     default=None,    type=str)
    p.add_argument("--end",       default=None,    type=str)
    p.add_argument("--capital",   default=10_000,  type=float)
    p.add_argument("--threshold", default=65.0,    type=float)
    p.add_argument("--atr-mult",  default=1.5,     type=float, dest="atr_mult")
    p.add_argument("--risk-pct",  default=0.01,    type=float, dest="risk_pct")
    p.add_argument("--multi-tf",  action="store_true", dest="multi_tf")
    p.add_argument("--no-chart",  action="store_true", dest="no_chart")
    p.add_argument("--synthetic", action="store_true",
                   help="Use synthetic OHLCV data (no network required)")
    return p.parse_args()


# ── Data fetching ─────────────────────────────────────────────────────────────

def _fetch(symbol: str, start: str, end: str, interval: str = "1d") -> "pd.DataFrame":
    """Download OHLCV from Yahoo Finance and normalise column names."""
    import yfinance as yf
    import pandas as pd

    df = yf.download(symbol, start=start, end=end, interval=interval,
                     auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {symbol} [{start} → {end}]")

    # yfinance returns MultiIndex columns when auto_adjust=True
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]

    # Keep only standard OHLCV
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df = df.dropna()
    return df


def _synthetic(symbol: str, start: str, end: str, seed: int = 42) -> "pd.DataFrame":
    """Generate synthetic daily OHLCV data for offline demo."""
    import numpy as np
    import pandas as pd

    dates  = pd.bdate_range(start=start, end=end)
    n      = len(dates)
    rng    = np.random.default_rng(seed)
    price  = 150.0
    closes = []
    for _ in range(n):
        price = price * (1 + 0.0004 + rng.normal(0, 0.012))
        closes.append(price)

    opens  = [c * (1 + rng.normal(0, 0.002)) for c in closes]
    highs  = [max(o, c) * (1 + abs(rng.normal(0, 0.004))) for o, c in zip(opens, closes)]
    lows   = [min(o, c) * (1 - abs(rng.normal(0, 0.004))) for o, c in zip(opens, closes)]
    vols   = [int(abs(rng.normal(5_000_000, 1_000_000))) for _ in closes]
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols},
        index=dates,
    )


def _resample_weekly(daily_df: "pd.DataFrame") -> "pd.DataFrame":
    """Resample daily OHLCV into weekly candles (Monday-open, Friday-close)."""
    agg = {
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }
    return daily_df.resample("W-FRI").agg(agg).dropna()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse()

    # Default date range: last 3 years
    today = datetime.today()
    start = args.start or (today - timedelta(days=3 * 365)).strftime("%Y-%m-%d")
    end   = args.end   or today.strftime("%Y-%m-%d")

    print(f"\n{'═'*60}")
    print(f"  Backtest  {args.symbol}  [{args.tf}]  {start} → {end}")
    print(f"  Capital ${args.capital:,.0f}  |  Threshold {args.threshold}"
          f"  |  ATR×{args.atr_mult}  |  Risk {args.risk_pct*100:.1f}%")
    print(f"{'═'*60}")

    # ── 1. Fetch data ────────────────────────────────────────────────────────
    if args.synthetic:
        print(f"\n[1/4] Generating synthetic daily data for {args.symbol}…")
        daily_df = _synthetic(args.symbol, start, end)
        print(f"      {len(daily_df)} synthetic bars "
              f"({daily_df.index[0].date()} → {daily_df.index[-1].date()})")
    else:
        print(f"\n[1/4] Downloading {args.symbol} daily data from Yahoo Finance…")
        try:
            daily_df = _fetch(args.symbol, start, end, interval="1d")
            print(f"      {len(daily_df)} daily bars loaded "
                  f"({daily_df.index[0].date()} → {daily_df.index[-1].date()})")
        except Exception as exc:
            print(f"      Yahoo Finance unavailable: {exc}")
            print(f"      Falling back to synthetic data…")
            daily_df = _synthetic(args.symbol, start, end)
            print(f"      {len(daily_df)} synthetic bars generated")

    # Build frames dict
    frames = {args.tf: daily_df}
    if args.multi_tf:
        weekly_df = _resample_weekly(daily_df)
        frames["W"] = weekly_df
        print(f"      {len(weekly_df)} weekly bars (resampled from daily)")

    # ── 2. Configure engine ──────────────────────────────────────────────────
    print(f"\n[2/4] Configuring BacktestEngine…")
    from backtest.engine  import BacktestEngine
    from risk.models      import RiskConfig

    risk_cfg = RiskConfig(
        risk_pct=args.risk_pct,
        min_rr=2.0,
        tp_r_levels=(1.0, 2.0, 3.0),
        tp_scale_pcts=(0.40, 0.35, 0.25),
        allow_fractional_units=True,
    )
    engine = BacktestEngine(
        threshold=args.threshold,
        min_confirmations=2,
        risk_config=risk_cfg,
        atr_mult=args.atr_mult,
        entry_at="next_open",
        slippage_pct=0.0005,
        warmup_bars=220,    # EMA-200 needs at least 200 bars
        max_open_bars=60,   # force-close after ~3 months
    )

    # ── 3. Run backtest ──────────────────────────────────────────────────────
    print(f"[3/4] Running backtest…")
    import time
    t0 = time.perf_counter()
    result = engine.run(
        symbol=args.symbol,
        frames=frames,
        entry_tf=args.tf,
        initial_capital=args.capital,
        start_date=start,
        end_date=end,
    )
    elapsed = time.perf_counter() - t0
    print(f"      Done in {elapsed:.2f}s — {result.metrics['n_trades']} trade(s) found")

    # ── 4. Results ───────────────────────────────────────────────────────────
    print(f"\n[4/4] Results\n")
    print(result.summary())

    if result.trades:
        print(f"\n  Last 10 trades:")
        print(f"  {'#':>3}  {'Dir':>5}  {'Entry':>8}  {'Exit':>8}  "
              f"{'Exit@':>8}  {'P&L $':>9}  {'P&L R':>7}  {'Reason':<10}")
        print(f"  {'─'*70}")
        for i, t in enumerate(result.trades[-10:]):
            print(
                f"  {i+1:>3}  {t.direction:>5}  "
                f"{str(t.entry_time)[:10]:>10}  "
                f"{str(t.exit_time)[:10]:>10}  "
                f"{t.exit_price:>8.2f}  "
                f"{t.pnl:>+9.2f}  "
                f"{t.pnl_r:>+7.3f}  "
                f"{t.exit_reason:<10}"
            )

    # ── Chart ────────────────────────────────────────────────────────────────
    if not args.no_chart:
        from backtest.chart import plot_equity_curve
        chart_path = f"logs/{args.symbol}_{args.tf}_backtest.png"
        saved = plot_equity_curve(result, save_path=chart_path)
        if saved:
            print(f"\n  Equity curve saved → {saved}")
        else:
            print("\n  (no chart — no trades to plot)")
    else:
        print("\n  (chart skipped)")

    # ── Per-category metrics ─────────────────────────────────────────────────
    m = result.metrics
    print(f"""
  Detailed metrics:
    Gross profit   ${m['gross_profit']:>10,.2f}
    Gross loss     ${m['gross_loss']:>10,.2f}
    Avg win (R)    {m['avg_win_r']:>10.3f}
    Avg loss (R)   {m['avg_loss_r']:>10.3f}
    Largest win    {m['largest_win']:>10.3f}R
    Largest loss   {m['largest_loss']:>10.3f}R
""")


if __name__ == "__main__":
    main()
