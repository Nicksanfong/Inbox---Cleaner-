"""
Trading dashboard — Streamlit app.

Run:
    streamlit run dashboard/app.py

Environment variables (optional — falls back to demo mode if missing):
    ALPACA_API_KEY     paper-trading key
    ALPACA_SECRET_KEY  paper-trading secret

Demo mode generates all data from synthetic OHLCV so no broker connection
is needed.
"""
from __future__ import annotations

import time
from datetime import datetime

import pandas as pd
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────

st.set_page_config(
    page_title="Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Project root on path ──────────────────────────────────────────────────────

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# ── Imports (lazy — give useful error if missing) ─────────────────────────────

try:
    from dashboard.loader import (
        is_demo,
        get_trading_mode,
        make_broker,
        get_account,
        get_open_positions,
        get_pnl_summary,
        get_equity_curve,
        get_recent_signals,
        get_price_with_indicators,
        get_trade_history,
        get_risk_metrics,
        close_position,
    )
    from dashboard.charts import (
        candlestick_chart,
        equity_curve_chart,
        signal_breakdown_chart,
        pnl_bar_chart,
        monthly_returns_heatmap,
    )
    from execution.journal import TradeJournal
    from execution.store   import PositionStore
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# ── Auto-refresh ──────────────────────────────────────────────────────────────

try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
    _autorefresh_available = True
except ImportError:
    _autorefresh_available = False

# ── Shared singletons ─────────────────────────────────────────────────────────

@st.cache_resource
def _get_store() -> PositionStore:
    return PositionStore()


@st.cache_resource
def _get_journal() -> TradeJournal:
    return TradeJournal()


@st.cache_resource
def _get_broker():
    return make_broker()


# ── Dark-mode CSS injection ───────────────────────────────────────────────────

st.markdown("""
<style>
    .stApp { background-color: #0d0d0d; }
    section[data-testid="stSidebar"] { background-color: #111111; }
    .kpi-card {
        background: #1a1a1a;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 16px 20px;
        text-align: center;
    }
    .kpi-label { color: #888; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; }
    .kpi-value { color: #eee; font-size: 1.5rem; font-weight: 700; margin: 4px 0 0; }
    .kpi-value.positive { color: #00e676; }
    .kpi-value.negative { color: #ef5350; }
    .mode-badge-paper {
        display: inline-block;
        background: #1565c0;
        color: #90caf9;
        border: 1px solid #1976d2;
        border-radius: 4px;
        padding: 3px 10px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.06em;
    }
    .mode-badge-live {
        display: inline-block;
        background: #b71c1c;
        color: #ef9a9a;
        border: 1px solid #c62828;
        border-radius: 4px;
        padding: 3px 10px;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.06em;
    }
    .demo-banner {
        background: #1a1200;
        border: 1px solid #ffa000;
        border-radius: 6px;
        color: #ffca28;
        padding: 8px 16px;
        font-size: 0.85rem;
        margin-bottom: 10px;
    }
    div[data-testid="stHorizontalBlock"] { gap: 12px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pnl_class(v: float) -> str:
    return "positive" if v >= 0 else "negative"


def _fmt_pnl(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}${v:,.2f}"


def _kpi(label: str, value: str, css_class: str = "") -> None:
    st.markdown(
        f"""<div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value {css_class}">{value}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _mode_badge(mode: str) -> str:
    cls = "mode-badge-paper" if mode == "PAPER" else "mode-badge-live"
    return f'<span class="{cls}">{mode}</span>'


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _sidebar(broker, store, journal, demo: bool, mode: str) -> dict:
    """Render sidebar, return user settings dict."""
    with st.sidebar:
        st.markdown("### Trading Dashboard")
        st.markdown(_mode_badge(mode), unsafe_allow_html=True)

        if demo:
            st.markdown(
                '<div class="demo-banner">DEMO MODE — synthetic data only</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        # Refresh controls
        st.markdown("**Auto-refresh**")
        refresh_interval = st.selectbox(
            "Interval", ["Off", "30s", "1m", "5m"], index=1, label_visibility="collapsed"
        )
        interval_ms = {"Off": 0, "30s": 30_000, "1m": 60_000, "5m": 300_000}[refresh_interval]
        if interval_ms and _autorefresh_available:
            st_autorefresh(interval=interval_ms, key="auto_refresh")

        if st.button("Refresh now", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.divider()

        # Watchlist for charts
        st.markdown("**Watchlist**")
        default_symbols = "AAPL, MSFT, SPY, TSLA, NVDA"
        raw = st.text_input("Symbols (comma-separated)", value=default_symbols,
                            label_visibility="collapsed")
        symbols = [s.strip().upper() for s in raw.split(",") if s.strip()]

        # Chart bars
        n_bars = st.slider("Chart bars", min_value=50, max_value=500, value=120, step=10)

        st.divider()
        st.markdown(
            f"<span style='color:#555;font-size:0.75rem;'>"
            f"Updated {datetime.now().strftime('%H:%M:%S')}</span>",
            unsafe_allow_html=True,
        )

    return {"symbols": symbols, "n_bars": n_bars}


# ── Tab: Overview ─────────────────────────────────────────────────────────────

def _tab_overview(broker, store, journal, demo: bool) -> None:
    account   = get_account(broker)
    positions = get_open_positions(store, broker)
    pnl       = get_pnl_summary(journal, positions)

    # Account KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        _kpi("Portfolio Value", f"${account.get('portfolio_value', 0):,.2f}")
    with c2:
        d = pnl.get("daily", 0)
        _kpi("Today's P&L", _fmt_pnl(d), _pnl_class(d))
    with c3:
        w = pnl.get("weekly", 0)
        _kpi("This Week", _fmt_pnl(w), _pnl_class(w))
    with c4:
        mo = pnl.get("monthly", 0)
        _kpi("This Month", _fmt_pnl(mo), _pnl_class(mo))
    with c5:
        at = pnl.get("alltime", 0)
        _kpi("All-Time", _fmt_pnl(at), _pnl_class(at))

    st.divider()

    c_left, c_right = st.columns([2, 1])

    with c_left:
        st.markdown("##### Equity Curve")
        eq_df = get_equity_curve(journal, account.get("portfolio_value", 10_000))
        st.plotly_chart(equity_curve_chart(eq_df), use_container_width=True)

    with c_right:
        st.markdown("##### Risk Metrics")
        eq_df = get_equity_curve(journal, account.get("portfolio_value", 10_000))
        metrics = get_risk_metrics(eq_df, account.get("portfolio_value", 10_000))

        def _metric_row(label: str, value) -> None:
            if isinstance(value, float):
                formatted = f"{value:.3f}" if abs(value) < 100 else f"{value:.1f}"
            else:
                formatted = str(value)
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.markdown(f"<span style='color:#888;font-size:0.85rem;'>{label}</span>",
                            unsafe_allow_html=True)
            with col_b:
                st.markdown(f"<span style='color:#eee;font-size:0.85rem;'>{formatted}</span>",
                            unsafe_allow_html=True)

        _metric_row("Sharpe Ratio",   metrics.get("sharpe_ratio", 0.0))
        _metric_row("Sortino Ratio",  metrics.get("sortino_ratio", 0.0))
        _metric_row("Max Drawdown",   f"{metrics.get('max_drawdown_pct', 0.0)*100:.2f}%")
        _metric_row("Profit Factor",  metrics.get("profit_factor", 0.0))
        _metric_row("Win Rate",       f"{metrics.get('win_rate', 0.0)*100:.1f}%")
        _metric_row("Expectancy (R)", metrics.get("expectancy_r", 0.0))
        _metric_row("Avg R:R",        metrics.get("avg_rr", 0.0))
        _metric_row("Total Return",   f"{metrics.get('total_return_pct', 0.0)*100:.2f}%")

        st.divider()

        st.markdown("##### Account")
        _metric_row("Buying Power", f"${account.get('buying_power', 0):,.2f}")
        _metric_row("Currency",     account.get("currency", "USD"))
        _metric_row("Mode",         account.get("mode", "PAPER"))


# ── Tab: Positions ────────────────────────────────────────────────────────────

def _tab_positions(broker, store, journal, demo: bool, symbols: list, n_bars: int) -> None:
    positions = get_open_positions(store, broker)

    if not positions:
        st.info("No open positions.")
    else:
        st.markdown(f"##### {len(positions)} Open Position(s)")

        # Build display dataframe
        rows = []
        for p in positions:
            unr = p.get("unrealized_pnl", 0.0)
            rows.append({
                "Symbol":        p.get("symbol", ""),
                "Dir":           p.get("direction", ""),
                "Entry":         f"${p.get('entry_price', 0):.2f}",
                "Current":       f"${p.get('current_price', 0):.2f}",
                "Qty":           p.get("qty_remaining", p.get("qty", 0)),
                "Stop":          f"${p.get('stop_price', 0):.2f}",
                "TP1":           f"${p.get('take_profits', [0])[0]:.2f}" if p.get("take_profits") else "-",
                "TP2":           f"${p.get('take_profits', [0,0])[1]:.2f}" if len(p.get("take_profits", [])) > 1 else "-",
                "TP3":           f"${p.get('take_profits', [0,0,0])[2]:.2f}" if len(p.get("take_profits", [])) > 2 else "-",
                "Unr. P&L":     f"{'+' if unr >= 0 else ''}{unr:,.2f}",
                "B/E":           "Yes" if p.get("at_breakeven") else "No",
            })

        pos_df = pd.DataFrame(rows)
        st.dataframe(pos_df, use_container_width=True, hide_index=True)

        st.divider()

        # Close controls
        st.markdown("##### Manual Close")
        sym_options = [p.get("symbol", "") for p in positions]
        col_sel, col_btn = st.columns([2, 1])
        with col_sel:
            to_close = st.selectbox("Select position to close", sym_options,
                                    label_visibility="collapsed")
        with col_btn:
            if st.button("Close position", type="primary", use_container_width=True):
                if not demo:
                    ok = close_position(to_close, broker, store)
                    if ok:
                        st.success(f"{to_close} closed.")
                        st.cache_data.clear()
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error(f"Failed to close {to_close}.")
                else:
                    st.warning("Demo mode — no real orders placed.")

        st.divider()

    # Price chart for selected symbol
    st.markdown("##### Price Chart")
    all_syms = list({p.get("symbol", "") for p in positions} | set(symbols))
    all_syms = [s for s in all_syms if s]

    if all_syms:
        chart_sym = st.selectbox("Symbol", all_syms)
        pos_for_chart = next((p for p in positions if p.get("symbol") == chart_sym), None)

        with st.spinner(f"Loading {chart_sym} data…"):
            df = get_price_with_indicators(chart_sym, broker, n_bars)

        if not df.empty:
            fig = candlestick_chart(df, chart_sym, pos_for_chart)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No price data for {chart_sym}.")
    else:
        st.info("Add symbols in the sidebar to view charts.")


# ── Tab: Signals ──────────────────────────────────────────────────────────────

def _tab_signals(broker, demo: bool, symbols: list) -> None:
    with st.spinner("Scanning signals…"):
        signals = get_recent_signals(symbols, broker)

    if not signals:
        st.info("No signals above threshold.")
        return

    st.markdown(f"##### {len(signals)} Active Signal(s)")

    # Sort by score descending
    signals = sorted(signals, key=lambda s: s.get("score", 0), reverse=True)

    for sig in signals:
        direction = sig.get("direction", "")
        symbol    = sig.get("symbol", "")
        score     = sig.get("score", 0)
        tf        = sig.get("timeframe", "")
        ts        = sig.get("timestamp", "")
        confs     = sig.get("confs", 0)
        notes     = sig.get("notes", "")

        dir_color  = "#00e676" if direction == "BUY" else "#ef5350"
        score_pct  = min(score / 100, 1.0)

        with st.expander(
            f"{symbol}  [{tf}]  {direction}  —  {score:.1f} / 100  —  {confs} confirmations",
            expanded=(score >= 75),
        ):
            col_chart, col_info = st.columns([1, 1])

            with col_chart:
                fig = signal_breakdown_chart(sig)
                st.plotly_chart(fig, use_container_width=True)

            with col_info:
                st.markdown(
                    f"**Symbol**: `{symbol}`  \n"
                    f"**Direction**: <span style='color:{dir_color};font-weight:700'>{direction}</span>  \n"
                    f"**Score**: {score:.1f}  \n"
                    f"**Timeframe**: {tf}  \n"
                    f"**Timestamp**: {ts}  \n"
                    f"**Confirmations**: {confs}",
                    unsafe_allow_html=True,
                )
                if notes:
                    st.markdown(f"<span style='color:#888;font-size:0.82rem;'>{notes}</span>",
                                unsafe_allow_html=True)

                # Score bar
                st.markdown(
                    f"""<div style="margin-top:12px;">
                      <div style="font-size:0.75rem;color:#888;margin-bottom:4px;">Score</div>
                      <div style="background:#1a1a1a;border-radius:4px;height:10px;width:100%;">
                        <div style="background:{dir_color};border-radius:4px;
                                    height:10px;width:{score_pct*100:.1f}%;"></div>
                      </div>
                    </div>""",
                    unsafe_allow_html=True,
                )


# ── Tab: Performance ──────────────────────────────────────────────────────────

def _tab_performance(broker, store, journal, demo: bool) -> None:
    account  = get_account(broker)
    eq_df    = get_equity_curve(journal, account.get("portfolio_value", 10_000))
    metrics  = get_risk_metrics(eq_df, account.get("portfolio_value", 10_000))
    trades   = get_trade_history(journal)

    col_eq, col_pnl = st.columns([3, 2])

    with col_eq:
        st.markdown("##### Equity Curve")
        st.plotly_chart(equity_curve_chart(eq_df), use_container_width=True)

    with col_pnl:
        st.markdown("##### Per-Trade P&L (R)")
        trade_rows = [
            {"trade_num": i + 1, "pnl_r": t.get("pnl_r", 0), "symbol": t.get("symbol", "")}
            for i, t in enumerate(trades)
        ]
        st.plotly_chart(pnl_bar_chart(trade_rows), use_container_width=True)

    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _kpi("Sharpe", f"{metrics.get('sharpe_ratio', 0):.2f}")
    with c2:
        _kpi("Sortino", f"{metrics.get('sortino_ratio', 0):.2f}")
    with c3:
        dd = metrics.get("max_drawdown_pct", 0) * 100
        _kpi("Max Drawdown", f"{dd:.2f}%", "negative" if dd > 0 else "")
    with c4:
        pf = metrics.get("profit_factor", 0)
        _kpi("Profit Factor", f"{pf:.2f}", "positive" if pf >= 1 else "negative")

    st.divider()

    st.markdown("##### Monthly Returns")
    st.plotly_chart(monthly_returns_heatmap(trades), use_container_width=True)


# ── Tab: History ──────────────────────────────────────────────────────────────

def _tab_history(journal) -> None:
    trades = get_trade_history(journal)

    if not trades:
        st.info("No closed trades yet.")
        return

    st.markdown(f"##### {len(trades)} Closed Trade(s)")

    # Filters
    cf1, cf2, cf3 = st.columns([2, 2, 2])
    with cf1:
        syms = sorted({t.get("symbol", "") for t in trades if t.get("symbol")})
        sym_filter = st.multiselect("Symbol", syms, label_visibility="visible")
    with cf2:
        dir_filter = st.multiselect("Direction", ["BUY", "SELL"], label_visibility="visible")
    with cf3:
        reason_opts = sorted({t.get("exit_reason", "") for t in trades if t.get("exit_reason")})
        reason_filter = st.multiselect("Exit Reason", reason_opts, label_visibility="visible")

    filtered = trades
    if sym_filter:
        filtered = [t for t in filtered if t.get("symbol") in sym_filter]
    if dir_filter:
        filtered = [t for t in filtered if t.get("direction") in dir_filter]
    if reason_filter:
        filtered = [t for t in filtered if t.get("exit_reason") in reason_filter]

    rows = []
    for i, t in enumerate(filtered):
        pnl   = t.get("pnl", 0)
        pnl_r = t.get("pnl_r", 0)
        rows.append({
            "#":         i + 1,
            "Symbol":    t.get("symbol", ""),
            "Dir":       t.get("direction", ""),
            "Entry Time":  str(t.get("entry_time", ""))[:16],
            "Exit Time":   str(t.get("exit_time", ""))[:16],
            "Entry $":   t.get("entry_price", 0),
            "Exit $":    t.get("exit_price", 0),
            "Qty":       t.get("qty", 0),
            "P&L $":     round(pnl, 2),
            "P&L R":     round(pnl_r, 3),
            "Exit Reason": t.get("exit_reason", ""),
        })

    df = pd.DataFrame(rows)

    # Style P&L columns
    def _style_pnl(val):
        if isinstance(val, (int, float)):
            color = "#00e676" if val >= 0 else "#ef5350"
            return f"color: {color}"
        return ""

    styled = df.style.applymap(_style_pnl, subset=["P&L $", "P&L R"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Summary row
    total_pnl = sum(t.get("pnl", 0) for t in filtered)
    wins       = [t for t in filtered if t.get("pnl", 0) > 0]
    win_rate   = len(wins) / len(filtered) if filtered else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        _kpi("Filtered Trades", str(len(filtered)))
    with c2:
        _kpi("Win Rate", f"{win_rate*100:.1f}%", "positive" if win_rate >= 0.5 else "negative")
    with c3:
        _kpi("Total P&L", _fmt_pnl(total_pnl), _pnl_class(total_pnl))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    demo   = is_demo()
    mode   = get_trading_mode()
    broker = _get_broker()
    store  = _get_store()
    journal = _get_journal()

    settings = _sidebar(broker, store, journal, demo, mode)
    symbols  = settings["symbols"]
    n_bars   = settings["n_bars"]

    # Top bar
    top_left, top_right = st.columns([4, 1])
    with top_left:
        st.markdown(
            f"## Trading Dashboard  {_mode_badge(mode)}",
            unsafe_allow_html=True,
        )
    with top_right:
        account = get_account(broker)
        pv = account.get("portfolio_value", 0)
        st.markdown(
            f"<div style='text-align:right;padding-top:16px;'>"
            f"<span style='color:#888;font-size:0.8rem;'>Portfolio</span><br>"
            f"<span style='color:#eee;font-size:1.1rem;font-weight:700;'>${pv:,.2f}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    if demo:
        st.markdown(
            '<div class="demo-banner">'
            'Running in DEMO MODE — all data is synthetic. '
            'Set ALPACA_API_KEY and ALPACA_SECRET_KEY to connect to a live broker.'
            '</div>',
            unsafe_allow_html=True,
        )

    # Tabs
    tab_overview, tab_positions, tab_signals, tab_perf, tab_history = st.tabs([
        "Overview",
        "Positions",
        "Signals",
        "Performance",
        "History",
    ])

    with tab_overview:
        _tab_overview(broker, store, journal, demo)

    with tab_positions:
        _tab_positions(broker, store, journal, demo, symbols, n_bars)

    with tab_signals:
        _tab_signals(broker, demo, symbols)

    with tab_perf:
        _tab_performance(broker, store, journal, demo)

    with tab_history:
        _tab_history(journal)


if __name__ == "__main__":
    main()
