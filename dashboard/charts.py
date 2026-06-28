"""
Plotly chart factories for the trading dashboard.

All functions return a plotly.graph_objects.Figure ready for st.plotly_chart().
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    pass

# ── Shared theme ──────────────────────────────────────────────────────────────

_BG      = "#0d0d0d"
_PAPER   = "#111111"
_GRID    = "#222222"
_TEXT    = "#cccccc"
_GREEN   = "#00e676"
_RED     = "#ef5350"
_BLUE    = "#4fc3f7"
_ORANGE  = "#ffa726"
_PURPLE  = "#ce93d8"
_YELLOW  = "#fff176"

_LAYOUT_BASE = dict(
    paper_bgcolor=_BG,
    plot_bgcolor=_PAPER,
    font=dict(color=_TEXT, size=11),
    margin=dict(l=60, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#333333", borderwidth=1),
    xaxis=dict(gridcolor=_GRID, showgrid=True, zeroline=False),
    yaxis=dict(gridcolor=_GRID, showgrid=True, zeroline=False),
)


def _apply_base(fig: go.Figure, **kwargs) -> go.Figure:
    layout = {**_LAYOUT_BASE, **kwargs}
    fig.update_layout(**layout)
    return fig


# ── Candlestick + indicators ──────────────────────────────────────────────────

def candlestick_chart(
    df: pd.DataFrame,
    symbol: str,
    position: dict | None = None,
) -> go.Figure:
    """
    Candlestick chart with EMA overlays, volume, and RSI subplots.

    Parameters
    ----------
    df       : DataFrame with columns open, high, low, close, volume,
               and optionally ema_8, ema_21, ema_50, ema_200, rsi_14, atr_14
    symbol   : ticker for title
    position : open position dict (adds stop/TP horizontal lines)
    """
    has_rsi    = "rsi_14" in df.columns
    has_volume = "volume" in df.columns

    rows    = 1 + (1 if has_volume else 0) + (1 if has_rsi else 0)
    heights = [0.6, *([0.15] if has_volume else []), *([0.25] if has_rsi else [])]

    fig = make_subplots(
        rows=rows, cols=1,
        shared_xaxes=True,
        row_heights=heights,
        vertical_spacing=0.02,
    )

    # ── Candlestick ───────────────────────────────────────────────────────────

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        increasing_line_color=_GREEN, decreasing_line_color=_RED,
        name="Price",
    ), row=1, col=1)

    # ── EMA overlays ─────────────────────────────────────────────────────────

    ema_specs = [
        ("ema_8",   "#ff8a65", "EMA 8"),
        ("ema_21",  "#ffd54f", "EMA 21"),
        ("ema_50",  _BLUE,     "EMA 50"),
        ("ema_200", _PURPLE,   "EMA 200"),
    ]
    for col, color, name in ema_specs:
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df[col],
                mode="lines", line=dict(color=color, width=1),
                name=name, showlegend=True,
            ), row=1, col=1)

    # ── Stop / TP lines from open position ───────────────────────────────────

    if position:
        x_range = [df.index[0], df.index[-1]]

        stop = position.get("stop_price")
        if stop:
            fig.add_trace(go.Scatter(
                x=x_range, y=[stop, stop],
                mode="lines", line=dict(color=_RED, width=1, dash="dash"),
                name=f"Stop {stop:.2f}", showlegend=True,
            ), row=1, col=1)

        tps = position.get("take_profits", [])
        tp_hits = position.get("tp_hits", 0)
        for i, tp in enumerate(tps):
            hit   = i < tp_hits
            color = _GREEN if not hit else "#555555"
            label = f"TP{i+1} {tp:.2f}"
            if hit:
                label += " ✓"
            fig.add_trace(go.Scatter(
                x=x_range, y=[tp, tp],
                mode="lines", line=dict(color=color, width=1, dash="dot"),
                name=label, showlegend=True,
            ), row=1, col=1)

        # Entry line
        ep = position.get("entry_price")
        if ep:
            fig.add_trace(go.Scatter(
                x=x_range, y=[ep, ep],
                mode="lines", line=dict(color=_ORANGE, width=1, dash="longdash"),
                name=f"Entry {ep:.2f}", showlegend=True,
            ), row=1, col=1)

    # ── Volume bars ───────────────────────────────────────────────────────────

    vol_row = 2
    if has_volume:
        colors = [
            _GREEN if c >= o else _RED
            for o, c in zip(df["open"], df["close"])
        ]
        fig.add_trace(go.Bar(
            x=df.index, y=df["volume"],
            marker_color=colors, opacity=0.5,
            name="Volume", showlegend=False,
        ), row=vol_row, col=1)
        vol_row += 1

    # ── RSI ───────────────────────────────────────────────────────────────────

    if has_rsi:
        rsi_row = vol_row
        fig.add_trace(go.Scatter(
            x=df.index, y=df["rsi_14"],
            mode="lines", line=dict(color=_ORANGE, width=1.2),
            name="RSI 14", showlegend=False,
        ), row=rsi_row, col=1)
        # Overbought / oversold bands
        for level, color in [(70, _RED), (30, _GREEN)]:
            fig.add_trace(go.Scatter(
                x=[df.index[0], df.index[-1]], y=[level, level],
                mode="lines", line=dict(color=color, width=0.5, dash="dot"),
                showlegend=False,
            ), row=rsi_row, col=1)
        fig.update_yaxes(
            range=[0, 100],
            title_text="RSI",
            row=rsi_row, col=1,
            gridcolor=_GRID,
        )

    # ── Layout ────────────────────────────────────────────────────────────────

    fig.update_layout(
        title=dict(text=symbol, font=dict(color=_TEXT, size=14)),
        paper_bgcolor=_BG,
        plot_bgcolor=_PAPER,
        font=dict(color=_TEXT, size=11),
        margin=dict(l=60, r=20, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#333333", borderwidth=1,
                    orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        height=600,
    )
    for i in range(1, rows + 1):
        fig.update_xaxes(gridcolor=_GRID, row=i, col=1)
        fig.update_yaxes(gridcolor=_GRID, row=i, col=1)

    return fig


# ── Equity curve ──────────────────────────────────────────────────────────────

def equity_curve_chart(equity_df: pd.DataFrame) -> go.Figure:
    """
    Line chart of portfolio equity over time with fill.

    Parameters
    ----------
    equity_df : DataFrame with columns 'ts' (datetime) and 'equity' (float)
    """
    if equity_df.empty:
        fig = go.Figure()
        _apply_base(fig, title="No equity data yet")
        return fig

    ts     = equity_df["ts"]
    equity = equity_df["equity"]
    init   = equity.iloc[0]

    fig = go.Figure()

    # Fill above initial (profit zone)
    fig.add_trace(go.Scatter(
        x=ts, y=equity,
        mode="lines",
        line=dict(color=_BLUE, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(79,195,247,0.05)",
        name="Equity",
    ))

    # Zero-basis reference line
    fig.add_hline(
        y=init, line_color="#555555",
        line_dash="dash", line_width=0.8,
        annotation_text=f"Start ${init:,.0f}",
        annotation_font_color="#888888",
    )

    # Drawdown shading: find where below initial
    below = equity < init
    if below.any():
        fig.add_trace(go.Scatter(
            x=ts, y=equity.where(below, init),
            mode="lines", line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(239,83,80,0.12)",
            showlegend=False,
        ))

    _apply_base(
        fig,
        title="Portfolio Equity",
        height=350,
        xaxis=dict(gridcolor=_GRID),
        yaxis=dict(
            gridcolor=_GRID,
            tickprefix="$",
            tickformat=",.0f",
        ),
    )
    return fig


# ── Signal score breakdown ────────────────────────────────────────────────────

def signal_breakdown_chart(signal: dict) -> go.Figure:
    """
    Horizontal stacked bar chart showing bullish/bearish score per category.

    Parameters
    ----------
    signal : dict with 'breakdown' key mapping category → {'bullish': float, 'bearish': float}
    """
    breakdown = signal.get("breakdown", {})
    if not breakdown:
        fig = go.Figure()
        _apply_base(fig, title="No breakdown available")
        return fig

    categories = list(breakdown.keys())
    bullish    = [breakdown[c].get("bullish", 0) for c in categories]
    bearish    = [breakdown[c].get("bearish", 0) for c in categories]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=categories, x=bullish,
        orientation="h",
        name="Bullish",
        marker_color=_GREEN,
        opacity=0.85,
    ))
    fig.add_trace(go.Bar(
        y=categories, x=[-b for b in bearish],
        orientation="h",
        name="Bearish",
        marker_color=_RED,
        opacity=0.85,
    ))

    score = signal.get("score", 0)
    direction = signal.get("direction", "")

    _apply_base(
        fig,
        title=f"{direction}  Score: {score:.1f}",
        barmode="relative",
        height=280,
        xaxis=dict(
            gridcolor=_GRID,
            title="Score contribution",
            tickformat=".0f",
        ),
        yaxis=dict(gridcolor=_GRID),
        showlegend=True,
    )
    # Zero line
    fig.add_vline(x=0, line_color="#555555", line_width=0.8)
    return fig


# ── Per-trade P&L bars ────────────────────────────────────────────────────────

def pnl_bar_chart(trades: list[dict]) -> go.Figure:
    """
    Green/red vertical bars for each trade's P&L in R-multiples.

    Parameters
    ----------
    trades : list of dicts with keys 'trade_num' (or index), 'pnl_r', 'symbol'
    """
    if not trades:
        fig = go.Figure()
        _apply_base(fig, title="No trades yet")
        return fig

    labels = [t.get("trade_num", i + 1) for i, t in enumerate(trades)]
    pnls_r = [t.get("pnl_r", 0.0) for t in trades]
    colors  = [_GREEN if r > 0 else _RED for r in pnls_r]
    texts   = [f"{r:+.2f}R" for r in pnls_r]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=pnls_r,
        marker_color=colors,
        text=texts,
        textposition="outside",
        textfont=dict(size=9, color=_TEXT),
        name="P&L (R)",
    ))
    fig.add_hline(y=0, line_color="#555555", line_width=0.8)

    _apply_base(
        fig,
        title="Per-Trade P&L (R-multiples)",
        height=280,
        xaxis=dict(gridcolor=_GRID, title="Trade #"),
        yaxis=dict(gridcolor=_GRID, title="P&L (R)", zeroline=False),
        showlegend=False,
    )
    return fig


# ── Monthly returns heatmap ───────────────────────────────────────────────────

def monthly_returns_heatmap(trades: list[dict]) -> go.Figure:
    """
    Heatmap of monthly P&L aggregated from trade history.

    Parameters
    ----------
    trades : list of dicts with 'exit_time' (str or datetime) and 'pnl' (float)
    """
    if not trades:
        fig = go.Figure()
        _apply_base(fig, title="No trade data for heatmap")
        return fig

    rows = []
    for t in trades:
        et = t.get("exit_time")
        pnl = t.get("pnl", 0.0)
        if not et:
            continue
        try:
            dt = pd.to_datetime(et)
            rows.append({"year": dt.year, "month": dt.month, "pnl": pnl})
        except Exception:
            continue

    if not rows:
        fig = go.Figure()
        _apply_base(fig, title="No dated trades for heatmap")
        return fig

    df    = pd.DataFrame(rows)
    pivot = df.groupby(["year", "month"])["pnl"].sum().unstack(fill_value=0)

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    x_labels = [month_names[m - 1] for m in pivot.columns]
    y_labels  = [str(y) for y in pivot.index]

    z = pivot.values.tolist()

    abs_max = max(abs(v) for row in z for v in row) or 1

    fig = go.Figure(go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        colorscale=[
            [0.0,   _RED],
            [0.5,   "#1a1a1a"],
            [1.0,   _GREEN],
        ],
        zmid=0,
        zmin=-abs_max,
        zmax=abs_max,
        text=[[f"${v:,.0f}" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorbar=dict(tickprefix="$", title="P&L"),
    ))

    _apply_base(
        fig,
        title="Monthly Returns ($)",
        height=max(200, len(y_labels) * 45 + 80),
        xaxis=dict(gridcolor=_GRID),
        yaxis=dict(gridcolor=_GRID),
    )
    return fig
