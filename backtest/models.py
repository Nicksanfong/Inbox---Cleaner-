"""Backtest data structures — trade record and result container."""
from __future__ import annotations

from dataclasses import dataclass, field


class BacktestTrade:
    """
    One simulated round-trip trade.

    Mutable during replay (_stop may move to breakeven, qty scales down at TPs).
    Finalised when _closed is set — thereafter treat as read-only.
    """

    __slots__ = (
        "symbol", "direction", "entry_time", "entry_price", "qty",
        "stop_price", "take_profits", "tp_fracs",
        "signal_score", "risk_amount",
        "_stop", "_qty_remaining", "_tp_hits",
        "_realized_pnl", "_closed",
        "exit_time", "exit_price", "exit_reason",
        "pnl", "pnl_r", "partial_exits",
    )

    def __init__(
        self,
        symbol:        str,
        direction:     str,    # "long" | "short"
        entry_time,            # pd.Timestamp
        entry_price:   float,
        qty:           float,
        stop_price:    float,
        take_profits:  list,   # [tp1, tp2, tp3] prices
        tp_fracs:      list,   # [0.40, 0.35, 0.25] fractions of qty
        signal_score:  float = 0.0,
        risk_amount:   float = 0.0,
    ):
        self.symbol       = symbol
        self.direction    = direction
        self.entry_time   = entry_time
        self.entry_price  = entry_price
        self.qty          = qty
        self.stop_price   = stop_price
        self.take_profits = list(take_profits)
        self.tp_fracs     = list(tp_fracs)
        self.signal_score = signal_score
        self.risk_amount  = risk_amount
        # Mutable simulation state
        self._stop          = stop_price
        self._qty_remaining = qty
        self._tp_hits       = [False] * len(take_profits)
        self._realized_pnl  = 0.0
        self._closed        = False
        # Finalised at close
        self.exit_time   = None
        self.exit_price  = 0.0
        self.exit_reason = ""    # "tp1" | "tp2" | "tp3" | "stop" | "eod"
        self.pnl         = 0.0
        self.pnl_r       = 0.0
        self.partial_exits: list = []   # [(ts, price, qty, reason)]

    def unrealized_pnl(self, price: float) -> float:
        sign = 1.0 if self.direction == "long" else -1.0
        return self._qty_remaining * (price - self.entry_price) * sign

    def total_pnl(self, price: float) -> float:
        return self._realized_pnl + self.unrealized_pnl(price)

    def __repr__(self) -> str:
        state = "CLOSED" if self._closed else "OPEN"
        return (
            f"BacktestTrade({state} {self.direction} {self.symbol}"
            f" entry={self.entry_price:.2f} stop={self.stop_price:.2f}"
            f" score={self.signal_score:.1f}"
            f" pnl={self.pnl:+.2f} [{self.pnl_r:+.2f}R]"
            f" reason={self.exit_reason!r})"
        )


@dataclass
class BacktestResult:
    """Immutable snapshot of a completed backtest run."""

    symbol:            str
    entry_tf:          str
    start_date:        str
    end_date:          str
    initial_capital:   float
    trades:            list           # list[BacktestTrade]
    equity_curve:      list           # float per bar close
    equity_timestamps: list           # pd.Timestamp per bar close
    metrics:           dict

    def summary(self) -> str:
        m = self.metrics
        final = self.equity_curve[-1] if self.equity_curve else self.initial_capital
        lines = [
            f"{'─'*56}",
            f"  Backtest  {self.symbol}  [{self.entry_tf}]"
            f"  {self.start_date} → {self.end_date}",
            f"  Capital   ${self.initial_capital:>10,.2f}  →  ${final:>10,.2f}",
            f"{'─'*56}",
            f"  Trades          {m.get('n_trades', 0):>6}",
            f"  Wins / Losses   {m.get('n_wins', 0):>3} / {m.get('n_losses', 0):<3}",
            f"  Win rate        {m.get('win_rate', 0)*100:>6.1f} %",
            f"  Profit factor   {m.get('profit_factor', 0):>6.2f}",
            f"  Sharpe ratio    {m.get('sharpe_ratio', 0):>6.2f}",
            f"  Sortino ratio   {m.get('sortino_ratio', 0):>6.2f}",
            f"  Max drawdown    {m.get('max_drawdown_pct', 0)*100:>6.1f} %",
            f"  Expectancy      {m.get('expectancy_r', 0):>+6.3f} R",
            f"  Avg R:R         {m.get('avg_rr', 0):>6.2f}",
            f"  Total return    {m.get('total_return_pct', 0)*100:>+6.1f} %",
            f"{'─'*56}",
        ]
        return "\n".join(lines)
