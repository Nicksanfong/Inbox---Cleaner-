"""
BacktestEngine — replays historical OHLCV data through the full
signal → risk → execution pipeline without placing real orders.

Quick start
───────────
    from backtest.engine import BacktestEngine
    import yfinance as yf

    df = yf.download("AAPL", start="2022-01-01", end="2024-01-01")
    df.columns = [c.lower() for c in df.columns]

    engine = BacktestEngine(threshold=70.0, atr_mult=1.5)
    result = engine.run(
        symbol="AAPL",
        frames={"D": df},
        entry_tf="D",
        initial_capital=10_000,
    )
    print(result.summary())

Notes
─────
- Indicators are pre-computed once per DataFrame (not per bar).  This is
  safe because all indicators (EMA, RSI, MACD, ATR, …) are causal — each
  value depends only on past bars.
- Entry timing: entry_at="next_open" fills at the NEXT bar's open + slippage.
  entry_at="close" fills at the signal bar's close (slightly optimistic).
- Stop / TP resolution: TPs are checked first on each bar (optimistic for TPs).
  If stop AND TP3 both trigger on the same bar and the open price has not
  gapped through TP3, the stop fires first (conservative).
- Multi-TP scaling: 40 % at TP1, 35 % at TP2, 25 % at TP3.  Stop moves to
  breakeven after TP1 fills.
- Equity curve is mark-to-market: realized_capital + unrealized_pnl_at_close.
  This reflects real drawdowns caused by open losing positions.
"""
from __future__ import annotations

import logging

import pandas as pd

from backtest.metrics import compute_metrics
from backtest.models import BacktestResult, BacktestTrade
from indicators.combined import run_all
from patterns.detector import scan_all
from risk.manager import RiskManager
from risk.models import RiskConfig
from signals.engine import _TF_WEIGHT, _mtf_bonus
from signals.scoring import score_bar

log = logging.getLogger(__name__)

_DEFAULT_TP_FRACS = (0.40, 0.35, 0.25)
_DEFAULT_TP_R     = (1.0,  2.0,  3.0)


class BacktestEngine:
    """
    Replay historical data through the full pipeline.

    Parameters
    ----------
    threshold         : minimum confluence score to fire a signal (default 75)
    min_confirmations : minimum distinct confirmation sources required
    risk_config       : RiskConfig (default: 1 % risk, 1:2 min R:R)
    atr_mult          : stop distance = ATR-14 × atr_mult
    tp_r_levels       : R-multiples for TP1 / TP2 / TP3
    tp_fracs          : fraction of qty to exit at each TP
    entry_at          : "next_open" (realistic) | "close" (signal bar)
    slippage_pct      : fractional slippage applied to fill price
    warmup_bars       : entry TF bars to skip at start for indicator convergence
    max_open_bars     : force-close any trade still open after N bars
    """

    def __init__(
        self,
        threshold:         float = 75.0,
        min_confirmations: int   = 3,
        risk_config:       RiskConfig | None = None,
        atr_mult:          float = 1.5,
        tp_r_levels:       tuple = _DEFAULT_TP_R,
        tp_fracs:          tuple = _DEFAULT_TP_FRACS,
        entry_at:          str   = "next_open",
        slippage_pct:      float = 0.0005,
        warmup_bars:       int   = 50,
        max_open_bars:     int   = 100,
    ):
        self.threshold         = threshold
        self.min_confirmations = min_confirmations
        self.risk_config       = risk_config or RiskConfig(
            risk_pct=0.01, min_rr=2.0,
            tp_r_levels=tp_r_levels, tp_scale_pcts=tp_fracs,
            allow_fractional_units=True,
        )
        self.atr_mult      = atr_mult
        self.tp_r_levels   = tp_r_levels
        self.tp_fracs      = tp_fracs
        self.entry_at      = entry_at
        self.slippage      = slippage_pct
        self.warmup_bars   = warmup_bars
        self.max_open_bars = max_open_bars
        self._rm           = RiskManager(self.risk_config)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        symbol:          str,
        frames:          dict[str, pd.DataFrame],
        entry_tf:        str   = "D",
        initial_capital: float = 10_000.0,
        start_date:      str | None = None,
        end_date:        str | None = None,
    ) -> BacktestResult:
        """
        Run the backtest and return a BacktestResult.

        Parameters
        ----------
        symbol          : ticker label (display only)
        frames          : {tf_label: raw_OHLCV_DataFrame} — any subset of
                          ["W","D","4H","1H","30m","15m","5m"]
        entry_tf        : timeframe to iterate bar-by-bar for signal detection
        initial_capital : starting portfolio value in USD
        start_date      : ISO date string "YYYY-MM-DD"; skip bars before this
        end_date        : ISO date string "YYYY-MM-DD"; stop after this date
        """
        if entry_tf not in frames:
            raise ValueError(f"entry_tf '{entry_tf}' not found in provided frames")

        # 1. Pre-compute indicators + patterns once per TF
        precomputed = self._precompute(frames)
        if entry_tf not in precomputed:
            raise ValueError(f"entry_tf '{entry_tf}' failed indicator computation")

        entry_ind, entry_pats = precomputed[entry_tf]

        # 2. Determine bar index range [start_i, end_i)
        start_i, end_i = self._date_range(
            entry_ind, start_date, end_date, self.warmup_bars
        )
        if start_i >= end_i:
            log.warning("BacktestEngine: no bars in the requested date range")
            return self._empty_result(symbol, entry_tf, start_date, end_date,
                                      initial_capital)

        # 3. Replay
        capital:    float        = initial_capital
        equity_curve: list[float] = [initial_capital]
        equity_ts:    list        = [entry_ind.index[start_i]]
        trades:       list        = []
        open_trade:   BacktestTrade | None = None
        open_since:   int         = 0

        for i in range(start_i, end_i):
            row = entry_ind.iloc[i]
            ts  = entry_ind.index[i]

            # ── a. Update open trade ──────────────────────────────────────────
            if open_trade is not None:
                bars_open = i - open_since
                if bars_open >= self.max_open_bars:
                    capital += self._force_close(open_trade, row, ts)
                else:
                    capital += self._update_trade(open_trade, row, ts)

                if open_trade._closed:
                    trades.append(open_trade)
                    open_trade = None

            # ── b. Check for new signal ───────────────────────────────────────
            if open_trade is None:
                sig = self._try_signal(entry_tf, i, ts, precomputed)
                if sig is not None:
                    direction, score = sig
                    trade = self._enter_trade(
                        symbol, direction, score, i, entry_ind, precomputed,
                        entry_tf, capital
                    )
                    if trade is not None:
                        open_trade = trade
                        open_since = i

            # ── c. Equity mark-to-market ──────────────────────────────────────
            close_price = float(row["close"])
            mtm = (
                capital + open_trade.total_pnl(close_price)
                if open_trade else capital
            )
            equity_curve.append(mtm)
            equity_ts.append(ts)

        # 4. Close any trade still open at end-of-data
        if open_trade is not None:
            last_row = entry_ind.iloc[end_i - 1]
            last_ts  = entry_ind.index[end_i - 1]
            capital += self._force_close(open_trade, last_row, last_ts)
            trades.append(open_trade)
            equity_curve[-1] = capital

        metrics = compute_metrics(trades, equity_curve)

        actual_start = str(entry_ind.index[start_i])[:10]
        actual_end   = str(entry_ind.index[end_i - 1])[:10]

        return BacktestResult(
            symbol=symbol,
            entry_tf=entry_tf,
            start_date=actual_start,
            end_date=actual_end,
            initial_capital=initial_capital,
            trades=trades,
            equity_curve=equity_curve,
            equity_timestamps=equity_ts,
            metrics=metrics,
        )

    # ── Pre-computation ───────────────────────────────────────────────────────

    def _precompute(self, frames: dict) -> dict:
        """Run run_all() and scan_all() once per TF. Returns {tf: (ind_df, bar_pats)}."""
        out = {}
        for tf, raw_df in frames.items():
            if raw_df is None or raw_df.empty:
                continue
            try:
                ind_df = run_all(raw_df.copy())
                pats   = scan_all(ind_df)
                bar_pats: dict[int, list] = {}
                for p in pats:
                    bar_pats.setdefault(p.bar_index, []).append(p)
                out[tf] = (ind_df, bar_pats)
                log.info("BacktestEngine: precomputed %s  %d bars", tf, len(ind_df))
            except Exception as exc:
                log.warning("BacktestEngine: precompute failed for %s — %s", tf, exc)
        return out

    # ── Signal detection ──────────────────────────────────────────────────────

    def _try_signal(
        self,
        entry_tf:     str,
        entry_bar_i:  int,
        entry_ts,             # pd.Timestamp
        precomputed:  dict,
    ) -> tuple[str, float] | None:
        """
        Score the entry TF bar and all higher TFs visible at entry_ts.
        Returns (direction, score) or None.
        """
        tf_scored: dict = {}
        for tf, (ind_df, bar_pats) in precomputed.items():
            if tf == entry_tf:
                i = entry_bar_i
            else:
                # Most recent bar with timestamp ≤ entry_ts (no lookahead)
                idx = ind_df.index.searchsorted(entry_ts, side="right") - 1
                if idx < 0:
                    continue
                i = int(idx)
            if i < 0 or i >= len(ind_df):
                continue
            row  = ind_df.iloc[i]
            pats = bar_pats.get(i, [])
            tf_scored[tf] = score_bar(row, pats)

        if not tf_scored:
            return None

        best = None
        for direction in ("bullish", "bearish"):
            total_w   = sum(_TF_WEIGHT.get(tf, 0.25) for tf in tf_scored)
            weighted  = (
                sum(_TF_WEIGHT.get(tf, 0.25) * s[direction] for tf, s in tf_scored.items())
                / total_w
            ) if total_w > 0 else 0.0

            agreements = sum(
                1 for s in tf_scored.values()
                if (s["bullish"] >= s["bearish"]) == (direction == "bullish")
            )
            bonus = _mtf_bonus(agreements, len(tf_scored))
            final = min(round(weighted + bonus, 1), 100.0)

            if final < self.threshold:
                continue

            conf_key = "bull_confirmations" if direction == "bullish" else "bear_confirmations"
            confs    = tf_scored.get(entry_tf, {}).get(conf_key, [])
            if len(confs) < self.min_confirmations:
                continue

            if best is None or final > best[1]:
                best = ("BUY" if direction == "bullish" else "SELL", final)

        return best

    # ── Trade entry ───────────────────────────────────────────────────────────

    def _enter_trade(
        self,
        symbol:       str,
        direction:    str,    # "BUY" | "SELL"
        score:        float,
        bar_i:        int,
        entry_ind,            # indicator DataFrame for entry TF
        precomputed:  dict,   # unused here but available for extension
        entry_tf:     str,
        capital:      float,
    ) -> BacktestTrade | None:
        """
        Compute entry price, ATR-based stop, TP levels and size the trade.
        Returns None if risk checks reject the trade.
        """
        row       = entry_ind.iloc[bar_i]
        direction_str = "long" if direction == "BUY" else "short"
        is_long   = direction_str == "long"

        # Entry price
        if self.entry_at == "next_open" and bar_i + 1 < len(entry_ind):
            next_row    = entry_ind.iloc[bar_i + 1]
            entry_price = float(next_row["open"])
            entry_ts    = entry_ind.index[bar_i + 1]
        else:
            entry_price = float(row["close"])
            entry_ts    = entry_ind.index[bar_i]

        # Apply slippage
        entry_price = entry_price * (1 + self.slippage if is_long else 1 - self.slippage)
        entry_price = round(entry_price, 4)

        # ATR-based stop
        atr_val = float(row.get("atr_14", entry_price * 0.02))
        stop_dist = max(atr_val * self.atr_mult, entry_price * 0.005)
        stop_price = round(
            entry_price - stop_dist if is_long else entry_price + stop_dist, 4
        )

        # TP levels
        sign = 1 if is_long else -1
        take_profits = [
            round(entry_price + sign * r * stop_dist, 4)
            for r in self.tp_r_levels
        ]

        # Risk sizing
        size = self._rm.approve_trade(
            entry_price=entry_price,
            stop_price=stop_price,
            account_balance=capital,
            day_start_balance=capital,
            direction=direction_str,
            symbol=symbol,
        )
        if not size.valid:
            log.debug("BacktestEngine: trade rejected at bar %d — %s",
                      bar_i, size.rejection_reasons)
            return None

        qty = size.units
        if qty <= 0:
            return None

        trade = BacktestTrade(
            symbol=symbol,
            direction=direction_str,
            entry_time=entry_ts,
            entry_price=entry_price,
            qty=qty,
            stop_price=stop_price,
            take_profits=take_profits,
            tp_fracs=list(self.tp_fracs),
            signal_score=score,
            risk_amount=size.risk_amount,
        )
        log.info(
            "BACKTEST  ENTER  %s %s  bar=%d  entry=%.4f  stop=%.4f  "
            "tp=[%.2f,%.2f,%.2f]  qty=%.4f  score=%.1f",
            direction_str, symbol, bar_i, entry_price, stop_price,
            take_profits[0], take_profits[1], take_profits[2], qty, score,
        )
        return trade

    # ── Trade update (bar by bar) ─────────────────────────────────────────────

    def _update_trade(
        self,
        trade: BacktestTrade,
        bar,          # pd.Series row from indicator df
        ts,           # pd.Timestamp
    ) -> float:
        """
        Process one bar for an open trade.
        Returns incremental realised P&L for this bar.
        """
        is_long = trade.direction == "long"
        o = float(bar.get("open", bar["close"]))
        h = float(bar["high"])
        l = float(bar["low"])
        incremental = 0.0

        # ── TP1 ──────────────────────────────────────────────────────────────
        if not trade._tp_hits[0]:
            tp1 = trade.take_profits[0]
            if (is_long and h >= tp1) or (not is_long and l <= tp1):
                fill = (max(o, tp1) if is_long else min(o, tp1))
                qty  = round(trade.qty * trade.tp_fracs[0], 8)
                qty  = min(qty, trade._qty_remaining)
                incremental += self._partial_exit(trade, ts, fill, qty, "tp1", is_long)
                trade._stop = trade.entry_price   # move stop to breakeven

        # ── TP2 ──────────────────────────────────────────────────────────────
        if not trade._closed and not trade._tp_hits[1]:
            tp2 = trade.take_profits[1]
            if (is_long and h >= tp2) or (not is_long and l <= tp2):
                fill = (max(o, tp2) if is_long else min(o, tp2))
                qty  = round(trade.qty * trade.tp_fracs[1], 8)
                qty  = min(qty, trade._qty_remaining)
                incremental += self._partial_exit(trade, ts, fill, qty, "tp2", is_long)

        # ── TP3 ──────────────────────────────────────────────────────────────
        if not trade._closed and not trade._tp_hits[2]:
            tp3 = trade.take_profits[2]
            tp3_hit = (is_long and h >= tp3) or (not is_long and l <= tp3)
            if tp3_hit:
                stop_also = (is_long and l <= trade._stop) or (not is_long and h >= trade._stop)
                # If stop also fires on same bar, TP3 wins only if open gapped through it
                if stop_also:
                    tp3_hit = (is_long and o >= tp3) or (not is_long and o <= tp3)
                if tp3_hit:
                    fill = (max(o, tp3) if is_long else min(o, tp3))
                    qty  = trade._qty_remaining
                    incremental += self._partial_exit(trade, ts, fill, qty, "tp3", is_long)
                    self._close_trade(trade, ts, "tp3")

        # ── Stop ─────────────────────────────────────────────────────────────
        if not trade._closed and trade._qty_remaining > 0:
            stop_hit = (is_long and l <= trade._stop) or (not is_long and h >= trade._stop)
            if stop_hit:
                fill = (min(o, trade._stop) if is_long else max(o, trade._stop))
                qty  = trade._qty_remaining
                incremental += self._partial_exit(trade, ts, fill, qty, "stop", is_long)
                self._close_trade(trade, ts, "stop")

        # ── Fully exited via TPs ──────────────────────────────────────────────
        if not trade._closed and trade._qty_remaining <= 1e-8:
            reason = trade.partial_exits[-1][3] if trade.partial_exits else "tp3"
            self._close_trade(trade, ts, reason)

        return incremental

    def _force_close(
        self, trade: BacktestTrade, bar, ts
    ) -> float:
        """Close trade at current bar's close (end-of-data or max_open_bars exceeded)."""
        if trade._closed:
            return 0.0
        close = float(bar["close"])
        qty   = trade._qty_remaining
        is_long = trade.direction == "long"
        incremental = self._partial_exit(trade, ts, close, qty, "eod", is_long)
        self._close_trade(trade, ts, "eod")
        return incremental

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _partial_exit(
        trade:   BacktestTrade,
        ts,
        price:   float,
        qty:     float,
        reason:  str,
        is_long: bool,
    ) -> float:
        """Execute one partial exit; returns realised P&L delta."""
        qty  = max(qty, 0.0)
        sign = 1.0 if is_long else -1.0
        pnl  = qty * (price - trade.entry_price) * sign
        trade._realized_pnl += pnl
        trade._qty_remaining  = max(trade._qty_remaining - qty, 0.0)
        tp_map = {"tp1": 0, "tp2": 1, "tp3": 2}
        if reason in tp_map:
            trade._tp_hits[tp_map[reason]] = True
        trade.partial_exits.append((ts, price, qty, reason))
        return pnl

    @staticmethod
    def _close_trade(trade: BacktestTrade, ts, reason: str) -> None:
        """Finalise a closed trade."""
        trade._closed    = True
        trade.exit_time  = ts
        trade.exit_reason = reason
        trade.pnl        = trade._realized_pnl
        if trade.partial_exits:
            total_qty        = sum(e[2] for e in trade.partial_exits)
            trade.exit_price = (
                sum(e[1] * e[2] for e in trade.partial_exits) / total_qty
                if total_qty > 0 else trade.entry_price
            )
        risk = trade.risk_amount
        trade.pnl_r = (trade.pnl / risk) if risk > 0 else 0.0
        log.info(
            "BACKTEST  EXIT   %s %s  %s  pnl=%+.2f  pnl_r=%+.3f R  exit=%.4f",
            trade.direction, trade.symbol, reason,
            trade.pnl, trade.pnl_r, trade.exit_price,
        )

    @staticmethod
    def _date_range(
        ind_df, start_date, end_date, warmup: int
    ) -> tuple[int, int]:
        """Return (start_i, end_i) bar index range within ind_df."""
        n = len(ind_df)
        start_i = warmup
        end_i   = n

        if start_date:
            ts = pd.Timestamp(start_date)
            pos = ind_df.index.searchsorted(ts, side="left")
            start_i = max(pos, warmup)

        if end_date:
            ts = pd.Timestamp(end_date)
            pos = ind_df.index.searchsorted(ts, side="right")
            end_i = min(pos, n)

        return start_i, end_i

    @staticmethod
    def _empty_result(
        symbol, entry_tf, start_date, end_date, initial_capital
    ) -> BacktestResult:
        return BacktestResult(
            symbol=symbol, entry_tf=entry_tf,
            start_date=start_date or "", end_date=end_date or "",
            initial_capital=initial_capital,
            trades=[], equity_curve=[initial_capital],
            equity_timestamps=[], metrics=compute_metrics([], [initial_capital]),
        )
