"""
Equity curve chart for backtest results.

Layout (3 panels, stacked):
  Top    (55 %): equity curve line + per-trade entry/exit markers
  Middle (25 %): per-trade P&L bar chart (green = win, red = loss)
  Bottom (20 %): drawdown area fill
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backtest.models import BacktestResult


def plot_equity_curve(
    result:    "BacktestResult",
    save_path: str | None = None,
    show:      bool = False,
) -> str | None:
    """
    Render the equity curve chart.

    Parameters
    ----------
    result    : completed BacktestResult
    save_path : file path to save PNG; if None, auto-generates next to journal
    show      : call plt.show() after saving (useful in notebooks)

    Returns
    -------
    Absolute path of the saved file, or None if no trades to plot.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend; no display required
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        raise ImportError("matplotlib required: pip install matplotlib")

    if not result.equity_curve:
        return None

    # ── Data prep ─────────────────────────────────────────────────────────────

    timestamps = result.equity_timestamps
    equity     = result.equity_curve
    trades     = result.trades

    # Drawdown series
    peak  = equity[0]
    dd    = []
    for v in equity:
        if v > peak:
            peak = v
        dd.append(-(peak - v) / peak * 100 if peak > 0 else 0.0)

    # Trade markers: entry and exit times + prices
    win_entries  = []   # (ts, price) for wins
    loss_entries = []
    win_exits    = []
    loss_exits   = []

    for t in trades:
        is_win = t.pnl > 0
        if t.entry_time is not None:
            (win_entries if is_win else loss_entries).append(
                (t.entry_time, t.entry_price)
            )
        if t.exit_time is not None:
            (win_exits if is_win else loss_exits).append(
                (t.exit_time, t.exit_price)
            )

    # Per-trade P&L bars
    trade_times  = [t.entry_time for t in trades if t.entry_time]
    trade_pnls_r = [t.pnl_r      for t in trades if t.entry_time]

    # ── Figure ────────────────────────────────────────────────────────────────

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1,
        figsize=(14, 9),
        gridspec_kw={"height_ratios": [5, 2.5, 1.5]},
        sharex=False,
    )
    fig.patch.set_facecolor("#0d0d0d")
    for ax in (ax1, ax2, ax3):
        ax.set_facecolor("#111111")
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        ax.spines["bottom"].set_color("#333333")
        ax.spines["top"].set_color("#333333")
        ax.spines["left"].set_color("#333333")
        ax.spines["right"].set_color("#333333")
        ax.yaxis.label.set_color("#aaaaaa")
        ax.xaxis.label.set_color("#aaaaaa")

    m   = result.metrics
    rtn = m.get("total_return_pct", 0) * 100
    title = (
        f"{result.symbol}  [{result.entry_tf}]  "
        f"{result.start_date} → {result.end_date}  |  "
        f"Return {rtn:+.1f}%  |  "
        f"Sharpe {m.get('sharpe_ratio', 0):.2f}  |  "
        f"Win {m.get('win_rate', 0)*100:.0f}%  |  "
        f"DD {m.get('max_drawdown_pct', 0)*100:.1f}%  |  "
        f"{m.get('n_trades', 0)} trades"
    )
    fig.suptitle(title, color="#dddddd", fontsize=9.5, y=0.98)

    # ── Panel 1: Equity curve ─────────────────────────────────────────────────

    ax1.plot(timestamps, equity, color="#4fc3f7", linewidth=1.2, zorder=2)
    ax1.fill_between(timestamps, equity, equity[0],
                     where=[e >= equity[0] for e in equity],
                     alpha=0.15, color="#4fc3f7", zorder=1)
    ax1.fill_between(timestamps, equity, equity[0],
                     where=[e < equity[0] for e in equity],
                     alpha=0.15, color="#ef5350", zorder=1)

    # Entry / exit markers
    def _scatter(pts, marker, color, size):
        if pts:
            xs, ys = zip(*pts)
            ax1.scatter(xs, ys, marker=marker, color=color, s=size, zorder=5, linewidths=0)

    _scatter(win_entries,  "^", "#00e676", 30)
    _scatter(loss_entries, "^", "#ef5350", 30)
    _scatter(win_exits,    "v", "#00e676", 30)
    _scatter(loss_exits,   "v", "#ef5350", 30)

    init = result.initial_capital
    ax1.axhline(init, color="#555555", linewidth=0.8, linestyle="--", zorder=1)
    ax1.set_ylabel("Portfolio Value ($)", color="#aaaaaa", fontsize=8)
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    ax1.grid(axis="y", color="#222222", linewidth=0.5)

    # ── Panel 2: Per-trade P&L ────────────────────────────────────────────────

    if trade_times and trade_pnls_r:
        colors = ["#00e676" if r > 0 else "#ef5350" for r in trade_pnls_r]
        ax2.bar(range(len(trade_times)), trade_pnls_r, color=colors,
                width=0.6, zorder=2)
        ax2.axhline(0, color="#555555", linewidth=0.8)
        ax2.set_ylabel("P&L (R)", color="#aaaaaa", fontsize=8)
        ax2.set_xlabel("Trade #", color="#aaaaaa", fontsize=8)
        ax2.grid(axis="y", color="#222222", linewidth=0.5)
        ax2.set_xlim(-0.5, len(trade_times) - 0.5)

    # ── Panel 3: Drawdown ─────────────────────────────────────────────────────

    ax3.fill_between(range(len(dd)), dd, 0, color="#ef5350", alpha=0.6, zorder=2)
    ax3.plot(range(len(dd)), dd, color="#ef5350", linewidth=0.7, zorder=3)
    ax3.axhline(0, color="#555555", linewidth=0.8)
    ax3.set_ylabel("Drawdown %", color="#aaaaaa", fontsize=8)
    ax3.set_xlabel("Bar", color="#aaaaaa", fontsize=8)
    ax3.grid(axis="y", color="#222222", linewidth=0.5)

    # ── Metrics annotation box ────────────────────────────────────────────────

    ann = (
        f"PF {m.get('profit_factor', 0):.2f}  |  "
        f"Sortino {m.get('sortino_ratio', 0):.2f}  |  "
        f"Exp {m.get('expectancy_r', 0):+.3f}R  |  "
        f"Avg R:R {m.get('avg_rr', 0):.2f}  |  "
        f"Max DD {m.get('max_drawdown_pct', 0)*100:.1f}%"
    )
    ax1.text(0.01, 0.02, ann, transform=ax1.transAxes,
             color="#aaaaaa", fontsize=7.5,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a1a",
                       edgecolor="#333333", alpha=0.85))

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # ── Save ──────────────────────────────────────────────────────────────────

    if save_path is None:
        save_path = f"logs/{result.symbol}_{result.entry_tf}_backtest.png"

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    if show:
        plt.show()
    plt.close(fig)
    return str(Path(save_path).resolve())
