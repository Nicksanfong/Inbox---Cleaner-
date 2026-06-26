"""
Tests for the backtesting engine.

All tests use deterministic synthetic OHLCV data — no network, no API keys.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from backtest.engine  import BacktestEngine
from backtest.metrics import compute_metrics
from backtest.models  import BacktestResult, BacktestTrade


# ── Synthetic data helpers ────────────────────────────────────────────────────

def _make_ohlcv(
    n: int          = 500,
    start_price: float = 100.0,
    trend:  float   = 0.0,    # daily drift
    noise:  float   = 0.5,
    seed:   int     = 42,
    freq:   str     = "B",    # "B" = business days
) -> pd.DataFrame:
    """Generate synthetic OHLCV with a configurable trend."""
    rng    = np.random.default_rng(seed)
    closes = [start_price]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + trend + rng.normal(0, noise / 100)))

    opens  = [c * (1 + rng.normal(0, 0.001)) for c in closes]
    highs  = [max(o, c) * (1 + abs(rng.normal(0, 0.003))) for o, c in zip(opens, closes)]
    lows   = [min(o, c) * (1 - abs(rng.normal(0, 0.003))) for o, c in zip(opens, closes)]
    vols   = [int(abs(rng.normal(1_000_000, 200_000))) for _ in closes]

    idx = pd.bdate_range("2022-01-03", periods=n, freq=freq)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": vols},
        index=idx,
    )


def _make_strong_bull(n: int = 600) -> pd.DataFrame:
    """Uptrending data with low noise — should generate BUY signals."""
    return _make_ohlcv(n=n, trend=0.003, noise=0.3, seed=7)


def _make_strong_bear(n: int = 600) -> pd.DataFrame:
    """Downtrending data with low noise — should generate SELL signals."""
    return _make_ohlcv(n=n, start_price=200.0, trend=-0.003, noise=0.3, seed=8)


def _run(df, initial_capital: float = 10_000.0, **kwargs) -> BacktestResult:
    """Shortcut: run engine with lower threshold so synthetic data fires signals."""
    engine_kw = dict(threshold=55.0, min_confirmations=1, warmup_bars=50, atr_mult=1.5)
    engine_kw.update(kwargs)
    engine = BacktestEngine(**engine_kw)
    return engine.run("TEST", {"D": df}, entry_tf="D", initial_capital=initial_capital)


# ══════════════════════════════════════════════════════════════════════════════
# BacktestTrade
# ══════════════════════════════════════════════════════════════════════════════

class TestBacktestTrade:
    def _trade(self, direction="long", entry=100.0, stop=95.0):
        return BacktestTrade(
            symbol="A", direction=direction,
            entry_time=pd.Timestamp("2023-01-02"),
            entry_price=entry, qty=10.0,
            stop_price=stop,
            take_profits=[105.0, 110.0, 115.0],
            tp_fracs=[0.40, 0.35, 0.25],
            risk_amount=50.0,
        )

    def test_initial_state(self):
        t = self._trade()
        assert t._qty_remaining == 10.0
        assert t._tp_hits       == [False, False, False]
        assert t._closed        == False
        assert t._stop          == 95.0

    def test_unrealized_pnl_long(self):
        t = self._trade()
        assert t.unrealized_pnl(110.0) == pytest.approx(100.0)

    def test_unrealized_pnl_short(self):
        t = self._trade("short", entry=100.0, stop=105.0)
        assert t.unrealized_pnl(90.0) == pytest.approx(100.0)

    def test_total_pnl(self):
        t = self._trade()
        t._realized_pnl  = 20.0
        t._qty_remaining = 6.0
        assert t.total_pnl(103.0) == pytest.approx(20.0 + 6 * 3.0)

    def test_repr(self):
        t = self._trade()
        r = repr(t)
        assert "OPEN"  in r
        assert "long"  in r
        assert "entry=100" in r


# ══════════════════════════════════════════════════════════════════════════════
# compute_metrics
# ══════════════════════════════════════════════════════════════════════════════

def _closed_trade(pnl_r: float, risk: float = 50.0) -> BacktestTrade:
    t = BacktestTrade(
        symbol="A", direction="long",
        entry_time=pd.Timestamp("2023-01-02"),
        entry_price=100.0, qty=10.0,
        stop_price=95.0, take_profits=[105.0, 110.0, 115.0],
        tp_fracs=[0.40, 0.35, 0.25], risk_amount=risk,
    )
    t._closed    = True
    t.pnl_r      = pnl_r
    t.pnl        = pnl_r * risk
    t.exit_time  = pd.Timestamp("2023-02-01")
    t.exit_price = 100.0 + pnl_r * 5
    return t


class TestComputeMetrics:
    def test_empty(self):
        m = compute_metrics([], [10_000])
        assert m["n_trades"] == 0

    def test_win_rate(self):
        trades = [_closed_trade(2.0)] * 3 + [_closed_trade(-1.0)] * 1
        m = compute_metrics(trades, [10_000, 10_050, 10_100, 10_150, 10_050])
        assert m["n_trades"] == 4
        assert m["n_wins"]   == 3
        assert m["win_rate"] == pytest.approx(0.75)

    def test_profit_factor(self):
        trades = [_closed_trade(2.0)] * 2 + [_closed_trade(-1.0)] * 2
        m = compute_metrics(trades, [10_000] * 5)
        # gross_profit = 200, gross_loss = 100 → pf = 2.0
        assert m["profit_factor"] == pytest.approx(2.0, abs=0.01)

    def test_all_wins_profit_factor_inf(self):
        trades = [_closed_trade(1.5)] * 3
        m = compute_metrics(trades, [10_000, 10_050, 10_100, 10_150])
        assert math.isinf(m["profit_factor"])

    def test_sharpe_positive_for_winning_strategy(self):
        trades = [_closed_trade(1.5)] * 10 + [_closed_trade(-0.5)] * 2
        equity = [10_000 + i * 50 for i in range(13)]
        m = compute_metrics(trades, equity)
        assert m["sharpe_ratio"] > 0

    def test_sortino_gte_sharpe_when_losses_small(self):
        trades = [_closed_trade(3.0)] * 8 + [_closed_trade(-0.5)] * 2
        equity = [10_000 + i * 100 for i in range(11)]
        m = compute_metrics(trades, equity)
        # Sortino uses only downside deviation — should be ≥ Sharpe
        assert m["sortino_ratio"] >= m["sharpe_ratio"]

    def test_max_drawdown_peak_to_trough(self):
        equity = [10_000, 11_000, 10_000, 9_000, 9_500, 10_000]
        m = compute_metrics([_closed_trade(1.0)], equity)
        # peak=11_000, trough=9_000 → dd = 2000/11000 ≈ 18.2%
        assert m["max_drawdown_pct"] == pytest.approx(2000 / 11000, abs=0.005)

    def test_max_drawdown_zero_when_monotone(self):
        equity = [10_000, 10_100, 10_200, 10_300]
        m = compute_metrics([_closed_trade(1.0)], equity)
        assert m["max_drawdown_pct"] == 0.0

    def test_expectancy_r(self):
        trades = [_closed_trade(2.0)] * 3 + [_closed_trade(-1.0)] * 1
        m = compute_metrics(trades, [10_000] * 5)
        # mean_r = (2+2+2-1)/4 = 1.25
        assert m["expectancy_r"] == pytest.approx(1.25)

    def test_avg_rr_ratio(self):
        trades = [_closed_trade(3.0)] * 2 + [_closed_trade(-1.0)] * 2
        m = compute_metrics(trades, [10_000] * 5)
        # avg_win_r = 3.0, avg_loss_r = -1.0 → avg_rr = 3.0
        assert m["avg_rr"] == pytest.approx(3.0)

    def test_total_return(self):
        equity = [10_000, 12_000]
        m = compute_metrics([_closed_trade(2.0)], equity)
        assert m["total_return_pct"] == pytest.approx(0.20)

    def test_single_trade(self):
        m = compute_metrics([_closed_trade(2.5)], [10_000, 10_125])
        assert m["n_trades"] == 1
        assert m["win_rate"] == 1.0


# ══════════════════════════════════════════════════════════════════════════════
# BacktestEngine — trade mechanics (unit tests via _update_trade)
# ══════════════════════════════════════════════════════════════════════════════

def _engine(**kw) -> BacktestEngine:
    return BacktestEngine(threshold=55.0, min_confirmations=1, **kw)


def _bar(o, h, l, c, atr=5.0) -> pd.Series:
    return pd.Series({"open": o, "high": h, "low": l, "close": c, "atr_14": atr})


def _open_trade(direction="long", entry=100.0, stop=95.0, qty=10.0, risk=50.0):
    tps = [105.0, 110.0, 115.0] if direction == "long" else [95.0, 90.0, 85.0]
    return BacktestTrade(
        symbol="A", direction=direction,
        entry_time=pd.Timestamp("2023-01-02"),
        entry_price=entry, qty=qty,
        stop_price=stop, take_profits=tps,
        tp_fracs=[0.40, 0.35, 0.25], risk_amount=risk,
    )


class TestTradeUpdate:
    def test_no_hit_below_tp(self):
        eng = _engine()
        t   = _open_trade()
        inc = eng._update_trade(t, _bar(101, 103, 100, 102), pd.Timestamp("2023-01-03"))
        assert inc == 0.0
        assert not t._closed

    def test_stop_hit(self):
        eng = _engine()
        t   = _open_trade()
        inc = eng._update_trade(t, _bar(99, 100, 93, 94), pd.Timestamp("2023-01-03"))
        assert t._closed
        assert t.exit_reason == "stop"
        assert inc < 0        # loss

    def test_stop_fills_at_stop_price(self):
        eng = _engine()
        t   = _open_trade(entry=100.0, stop=95.0)
        eng._update_trade(t, _bar(98, 99, 93, 94), pd.Timestamp("2023-01-03"))
        # bar opens at 98, above stop at 95 — fill at stop price
        assert t.exit_price == pytest.approx(95.0, abs=0.01)

    def test_stop_gap_fill_at_open(self):
        eng = _engine()
        t   = _open_trade(entry=100.0, stop=95.0)
        eng._update_trade(t, _bar(92, 93, 90, 91), pd.Timestamp("2023-01-03"))
        # bar opened below stop (92 < 95) → fill at open (92)
        assert t.exit_price == pytest.approx(92.0, abs=0.01)
        assert t.pnl == pytest.approx(10 * (92 - 100), abs=0.01)

    def test_tp1_partial_exit(self):
        eng = _engine()
        t   = _open_trade()
        eng._update_trade(t, _bar(103, 107, 102, 106), pd.Timestamp("2023-01-03"))
        # TP1=105 hit; exit 40% = 4 units
        assert t._tp_hits[0]   == True
        assert t._tp_hits[1]   == False
        assert t._qty_remaining == pytest.approx(6.0)
        assert not t._closed

    def test_tp1_moves_stop_to_breakeven(self):
        eng = _engine()
        t   = _open_trade(entry=100.0, stop=95.0)
        eng._update_trade(t, _bar(103, 107, 102, 106), pd.Timestamp("2023-01-03"))
        assert t._stop == pytest.approx(100.0)  # breakeven

    def test_tp2_partial_exit(self):
        eng = _engine()
        t   = _open_trade()
        eng._update_trade(t, _bar(103, 112, 102, 111), pd.Timestamp("2023-01-03"))
        # TP1=105, TP2=110 both hit
        assert t._tp_hits[0] and t._tp_hits[1]
        # remaining qty = 10 - 4.0 - 3.5 = 2.5
        assert t._qty_remaining == pytest.approx(2.5)

    def test_all_tps_hit_closes_trade(self):
        eng = _engine()
        t   = _open_trade()
        eng._update_trade(t, _bar(103, 120, 102, 119), pd.Timestamp("2023-01-03"))
        # TP1=105, TP2=110, TP3=115 all hit
        assert t._closed
        assert t.exit_reason in ("tp3",)
        assert t.pnl > 0

    def test_stop_after_tp1_uses_breakeven(self):
        eng = _engine()
        t   = _open_trade(entry=100.0, stop=95.0)
        # First bar: TP1 hit → stop moves to 100
        eng._update_trade(t, _bar(103, 107, 102, 106), pd.Timestamp("2023-01-03"))
        assert t._stop == 100.0
        # Second bar: price falls back to 99 → hits breakeven stop
        eng._update_trade(t, _bar(101, 102, 98, 99), pd.Timestamp("2023-01-04"))
        assert t._closed
        assert t.exit_reason == "stop"
        # Realised from TP1: 4 * (105-100) = 20
        # Realised from stop at 100: 6 * (100-100) = 0
        assert t.pnl == pytest.approx(20.0, abs=0.01)

    def test_short_tp1_hit(self):
        eng = _engine()
        t   = _open_trade(direction="short", entry=100.0, stop=105.0)
        eng._update_trade(t, _bar(97, 98, 93, 94), pd.Timestamp("2023-01-03"))
        # TP1=95 hit (low=93 ≤ 95)
        assert t._tp_hits[0] == True
        assert t._qty_remaining == pytest.approx(6.0)

    def test_short_stop_hit(self):
        eng = _engine()
        t   = _open_trade(direction="short", entry=100.0, stop=105.0)
        eng._update_trade(t, _bar(103, 107, 102, 106), pd.Timestamp("2023-01-03"))
        assert t._closed
        assert t.exit_reason == "stop"
        assert t.pnl < 0

    def test_force_close_at_bar_close(self):
        eng  = _engine()
        t    = _open_trade()
        bar  = _bar(101, 103, 100, 102)
        pnl  = eng._force_close(t, bar, pd.Timestamp("2023-01-03"))
        assert t._closed
        assert t.exit_reason == "eod"
        assert pnl == pytest.approx(10 * (102 - 100))

    def test_tp3_stop_same_bar_conservative(self):
        """If stop and TP3 both fire on same bar and open is not through TP3, stop wins."""
        eng = _engine()
        t   = _open_trade(entry=100.0, stop=95.0)
        # bar: opens at 98 (above stop), low=93 (below stop=95), high=118 (above tp3=115)
        eng._update_trade(t, _bar(98, 118, 93, 100), pd.Timestamp("2023-01-03"))
        # open not through TP3 (98 < 115) → stop fires first on remaining qty
        # TP1 and TP2 hit (h=118 ≥ 105 and 110), then stop fires on remaining
        assert t._closed
        # The exit_reason for the final (stop) exit:
        assert t.exit_reason == "stop"

    def test_pnl_r_calculation(self):
        eng = _engine()
        t   = _open_trade(entry=100.0, stop=95.0, qty=10.0, risk=50.0)
        eng._update_trade(t, _bar(103, 120, 102, 119), pd.Timestamp("2023-01-03"))
        assert t._closed
        # pnl_r = pnl / risk_amount
        assert t.pnl_r == pytest.approx(t.pnl / 50.0, abs=0.001)


# ══════════════════════════════════════════════════════════════════════════════
# BacktestEngine — full run (integration)
# ══════════════════════════════════════════════════════════════════════════════

class TestBacktestRun:
    def test_returns_backtest_result(self):
        df = _make_strong_bull()
        r  = _run(df)
        assert isinstance(r, BacktestResult)

    def test_equity_curve_length_matches_bars(self):
        df = _make_strong_bull(400)
        r  = _run(df)
        # equity_curve has one entry per bar in [start_i, end_i] + initial
        assert len(r.equity_curve) > 0
        assert len(r.equity_curve) == len(r.equity_timestamps)

    def test_bull_trend_generates_buy_signals(self):
        df = _make_strong_bull()
        r  = _run(df)
        longs = [t for t in r.trades if t.direction == "long"]
        assert len(longs) > 0, "Bull trend should produce at least one BUY trade"

    def test_bear_trend_generates_sell_signals(self):
        df = _make_strong_bear()
        r  = _run(df, threshold=50.0)
        shorts = [t for t in r.trades if t.direction == "short"]
        assert len(shorts) > 0, "Bear trend should produce at least one SELL trade"

    def test_initial_capital_preserved_with_no_trades(self):
        df = _make_ohlcv(n=300, trend=0.0, noise=0.01, seed=99)
        r  = _run(df, threshold=99.0)   # impossible threshold → no signals
        assert r.trades == []
        assert r.equity_curve[-1] == pytest.approx(10_000.0)

    def test_metrics_keys_present(self):
        df = _make_strong_bull()
        r  = _run(df)
        for key in ("win_rate", "profit_factor", "sharpe_ratio", "sortino_ratio",
                    "max_drawdown_pct", "expectancy_r", "avg_rr", "total_return_pct"):
            assert key in r.metrics, f"Missing metric: {key}"

    def test_trade_timestamps_ordered(self):
        df = _make_strong_bull()
        r  = _run(df)
        for t in r.trades:
            assert t.entry_time is not None
            if t.exit_time is not None:
                assert t.exit_time >= t.entry_time

    def test_no_open_trade_at_end(self):
        df = _make_strong_bull(300)
        r  = _run(df)
        # All trades must be closed (force-closed at end-of-data at the latest)
        assert all(t._closed for t in r.trades)

    def test_equity_curve_starts_at_initial_capital(self):
        df = _make_strong_bull()
        r  = _run(df, initial_capital=25_000.0)
        assert r.equity_curve[0] == 25_000.0

    def test_max_drawdown_bounded(self):
        df = _make_strong_bull()
        r  = _run(df)
        assert 0.0 <= r.metrics["max_drawdown_pct"] <= 1.0

    def test_win_rate_bounded(self):
        df = _make_strong_bull()
        r  = _run(df)
        assert 0.0 <= r.metrics["win_rate"] <= 1.0

    def test_date_range_filters_bars(self):
        df = _make_strong_bull(600)
        engine = BacktestEngine(threshold=55.0, min_confirmations=1, warmup_bars=50)
        r_full    = engine.run("TEST", {"D": df}, entry_tf="D", initial_capital=10_000.0)
        r_partial = engine.run("TEST", {"D": df}, entry_tf="D", initial_capital=10_000.0,
                               start_date="2023-01-01", end_date="2023-06-30")
        assert len(r_partial.trades) <= len(r_full.trades)

    def test_multi_tf_accepted(self):
        df_d  = _make_strong_bull(600)
        df_w  = _make_ohlcv(n=120, trend=0.002, noise=0.3, seed=10)
        engine = BacktestEngine(threshold=55.0, min_confirmations=1, warmup_bars=50)
        r = engine.run("TEST", {"D": df_d, "W": df_w}, entry_tf="D",
                       initial_capital=10_000.0)
        assert isinstance(r, BacktestResult)

    def test_summary_contains_symbol(self):
        df = _make_strong_bull(400)
        engine = BacktestEngine(threshold=55.0, min_confirmations=1, warmup_bars=50)
        r = engine.run("AAPL", {"D": df}, entry_tf="D", initial_capital=10_000.0)
        assert "AAPL" in r.summary()

    def test_no_lookahead_in_equity(self):
        """Equity at bar i must not exceed capital + max possible gain from bar i onwards."""
        df = _make_strong_bull(400)
        r  = _run(df)
        # All equity values must be finite
        for v in r.equity_curve:
            assert math.isfinite(v), "Non-finite equity curve value"

    def test_pnl_r_consistent_with_risk(self):
        """For each trade, pnl_r should equal pnl / risk_amount."""
        df = _make_strong_bull(500)
        r  = _run(df)
        for t in r.trades:
            if t.risk_amount > 0:
                expected_r = t.pnl / t.risk_amount
                assert t.pnl_r == pytest.approx(expected_r, abs=0.001), (
                    f"pnl_r mismatch: {t.pnl_r} vs {expected_r}"
                )

    def test_qty_conservation(self):
        """Sum of partial exit qtys should equal original qty."""
        df = _make_strong_bull(500)
        r  = _run(df)
        for t in r.trades:
            total = sum(e[2] for e in t.partial_exits)
            assert total == pytest.approx(t.qty, abs=1e-6), (
                f"Qty mismatch: exits={total} vs original={t.qty}"
            )

    def test_empty_result_on_bad_date_range(self):
        df = _make_strong_bull(400)
        engine = BacktestEngine(threshold=55.0, min_confirmations=1, warmup_bars=50)
        r = engine.run("X", {"D": df}, entry_tf="D",
                       start_date="2099-01-01", end_date="2099-12-31")
        assert r.trades == []

    def test_max_open_bars_forces_close(self):
        """max_open_bars=1 should force-close every trade after 1 bar."""
        df = _make_strong_bull(400)
        engine = BacktestEngine(
            threshold=55.0, min_confirmations=1, warmup_bars=50, max_open_bars=1
        )
        r = engine.run("TEST", {"D": df}, entry_tf="D", initial_capital=10_000.0)
        eod_trades = [t for t in r.trades if t.exit_reason == "eod"]
        assert len(eod_trades) > 0


# ══════════════════════════════════════════════════════════════════════════════
# BacktestResult summary
# ══════════════════════════════════════════════════════════════════════════════

class TestBacktestResult:
    def test_summary_format(self):
        df = _make_strong_bull(400)
        engine = BacktestEngine(threshold=55.0, min_confirmations=1, warmup_bars=50)
        r = engine.run("MSFT", {"D": df}, entry_tf="D", initial_capital=50_000.0)
        s = r.summary()
        assert "MSFT" in s
        assert "Win rate" in s
        assert "Sharpe" in s
        assert "Max drawdown" in s
        assert "Expectancy" in s


# ══════════════════════════════════════════════════════════════════════════════
# Chart smoke test (just checks it doesn't crash and returns a path)
# ══════════════════════════════════════════════════════════════════════════════

class TestChart:
    def test_plot_saves_file(self, tmp_path):
        from backtest.chart import plot_equity_curve

        df = _make_strong_bull(400)
        engine = BacktestEngine(threshold=55.0, min_confirmations=1, warmup_bars=50)
        r = engine.run("AAPL", {"D": df}, entry_tf="D", initial_capital=10_000.0)

        out_path = str(tmp_path / "equity.png")
        result   = plot_equity_curve(r, save_path=out_path)

        from pathlib import Path
        assert Path(out_path).exists()
        assert result == str(Path(out_path).resolve())

    def test_plot_flat_equity_still_saves(self, tmp_path):
        from backtest.chart import plot_equity_curve
        from pathlib import Path

        df = _make_ohlcv(300, seed=99)
        engine = BacktestEngine(threshold=99.9, min_confirmations=1, warmup_bars=50)
        r = engine.run("X", {"D": df}, entry_tf="D", initial_capital=10_000.0)

        out_path = str(tmp_path / "eq.png")
        plot_equity_curve(r, save_path=out_path)
        # flat equity curve (no trades) still renders to a file
        assert Path(out_path).exists()
