"""
Compute strategy performance metrics from a completed backtest.

All ratio metrics (Sharpe, Sortino) are trade-frequency-annualised:
  annualisation_factor = sqrt(trades_per_year)
  trades_per_year      = n_trades / (total_days / 365.25)
"""
from __future__ import annotations

import math
import statistics


def compute_metrics(
    trades: list,          # list[BacktestTrade]
    equity_curve: list,    # float per bar close
    risk_free_r: float = 0.0,
) -> dict:
    """
    Returns a dict of performance metrics.

    Keys
    ----
    n_trades, n_wins, n_losses, win_rate,
    profit_factor, sharpe_ratio, sortino_ratio,
    max_drawdown_pct, expectancy_r, avg_rr,
    total_return_pct, gross_profit, gross_loss,
    avg_win_r, avg_loss_r, largest_win, largest_loss,
    avg_trade_r
    """
    empty = {
        "n_trades": 0, "n_wins": 0, "n_losses": 0, "win_rate": 0.0,
        "profit_factor": 0.0, "sharpe_ratio": 0.0, "sortino_ratio": 0.0,
        "max_drawdown_pct": 0.0, "expectancy_r": 0.0, "avg_rr": 0.0,
        "total_return_pct": 0.0, "gross_profit": 0.0, "gross_loss": 0.0,
        "avg_win_r": 0.0, "avg_loss_r": 0.0,
        "largest_win": 0.0, "largest_loss": 0.0, "avg_trade_r": 0.0,
    }
    if not trades:
        return empty

    pnls   = [t.pnl for t in trades]
    rs     = [t.pnl_r for t in trades]
    wins   = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    n      = len(trades)

    # ── Basic stats ───────────────────────────────────────────────────────────

    win_rate      = len(wins) / n
    gross_profit  = sum(t.pnl for t in wins)
    gross_loss    = abs(sum(t.pnl for t in losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    avg_win_r  = statistics.mean(t.pnl_r for t in wins)  if wins   else 0.0
    avg_loss_r = statistics.mean(t.pnl_r for t in losses) if losses else 0.0
    largest_win  = max(rs) if rs else 0.0
    largest_loss = min(rs) if rs else 0.0

    # ── Expectancy (in R) ─────────────────────────────────────────────────────

    mean_r     = statistics.mean(rs)
    expectancy = mean_r     # per-trade expected R = win_rate * avg_win_r + loss_rate * avg_loss_r

    # ── Avg realised R:R ──────────────────────────────────────────────────────
    # Ratio of average win size to average loss size (both in R)

    avg_rr = (avg_win_r / abs(avg_loss_r)) if avg_loss_r != 0 else float("inf")

    # ── Annualisation factor ──────────────────────────────────────────────────

    if n >= 2 and trades[0].entry_time and trades[-1].exit_time:
        try:
            span = trades[-1].exit_time - trades[0].entry_time
            days = max(span.days, 1)
        except Exception:
            days = 365
        trades_per_year = n / (days / 365.25)
    else:
        trades_per_year = 52.0   # assume ~weekly

    ann_factor = math.sqrt(max(trades_per_year, 1.0))

    # ── Sharpe ────────────────────────────────────────────────────────────────

    std_r  = statistics.stdev(rs) if n > 1 else 0.0
    sharpe = ((mean_r - risk_free_r) / std_r * ann_factor) if std_r > 0 else 0.0

    # ── Sortino ───────────────────────────────────────────────────────────────
    # Semi-deviation: sqrt(mean(max(0, rf - r)^2)) over ALL trades.
    # This is well-defined even when all downside returns are identical.

    semi_sq  = [(max(0.0, risk_free_r - r)) ** 2 for r in rs]
    down_std = math.sqrt(sum(semi_sq) / n) if n > 0 else 0.0
    sortino  = ((mean_r - risk_free_r) / down_std * ann_factor) if down_std > 0 else 0.0

    # ── Max drawdown from equity curve ────────────────────────────────────────

    max_dd = 0.0
    peak   = equity_curve[0] if equity_curve else 1.0
    for v in equity_curve:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak
            if dd > max_dd:
                max_dd = dd

    # ── Total return ──────────────────────────────────────────────────────────

    total_return = 0.0
    if equity_curve and equity_curve[0] > 0:
        total_return = (equity_curve[-1] / equity_curve[0]) - 1.0

    return {
        "n_trades":         n,
        "n_wins":           len(wins),
        "n_losses":         len(losses),
        "win_rate":         round(win_rate, 4),
        "profit_factor":    round(profit_factor, 4),
        "sharpe_ratio":     round(sharpe, 4),
        "sortino_ratio":    round(sortino, 4),
        "max_drawdown_pct": round(max_dd, 4),
        "expectancy_r":     round(expectancy, 4),
        "avg_rr":           round(avg_rr, 4),
        "total_return_pct": round(total_return, 4),
        "gross_profit":     round(gross_profit, 2),
        "gross_loss":       round(gross_loss, 2),
        "avg_win_r":        round(avg_win_r, 4),
        "avg_loss_r":       round(avg_loss_r, 4),
        "largest_win":      round(largest_win, 4),
        "largest_loss":     round(largest_loss, 4),
        "avg_trade_r":      round(mean_r, 4),
    }
