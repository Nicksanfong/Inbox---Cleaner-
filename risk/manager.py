"""
RiskManager — central risk layer that governs every trade.

Typical flow
────────────
  rm   = RiskManager()                     # 1% risk, 5% max drawdown
  size = rm.approve_trade(entry=100,
                          stop=95,
                          account_balance=10_000,
                          day_start_balance=10_000,
                          direction="long",
                          symbol="AAPL")

  if size.valid:
      place_order(symbol, size.units, size.stop_price)
      store_tps(size.take_profits, size.tp_units)

Per-bar stop management (after entry)
──────────────────────────────────────
  pos = OpenPosition(symbol="AAPL", direction="long",
                     entry_price=100, current_stop=95,
                     units_remaining=20)

  pos = rm.update_stop(pos, current_price=106,
                       tp_prices=size.take_profits)
  # → stop moved to breakeven (100) if TP1 was hit,
  #   AND trailed up if trailing_stop_enabled
"""
from __future__ import annotations

import logging
from copy import copy

from risk.models import RiskConfig, PositionSize, OpenPosition

log = logging.getLogger(__name__)


class RiskManager:

    def __init__(self, config: RiskConfig | None = None):
        self.config = config or RiskConfig()

    # ── Position sizing ───────────────────────────────────────────────────────

    def size_trade(
        self,
        entry_price:     float,
        stop_price:      float,
        account_balance: float,
        direction:       str,    # "long" | "short"
        symbol:          str = "",
    ) -> PositionSize:
        """
        Compute units, TP levels, and R:R.  Applies the R:R filter only;
        drawdown / max-positions / correlation are checked in approve_trade().
        """
        cfg     = self.config
        reasons: list[str] = []

        stop_distance = abs(entry_price - stop_price)
        if stop_distance == 0.0:
            return PositionSize(
                symbol=symbol, direction=direction,
                entry_price=entry_price, stop_price=stop_price,
                stop_distance=0.0, units=0.0, risk_amount=0.0, rr_ratio=0.0,
                valid=False, rejection_reasons=["stop distance is zero"],
            )

        risk_amount = account_balance * cfg.risk_pct
        units       = risk_amount / stop_distance
        if not cfg.allow_fractional_units:
            units = float(int(units))   # floor to whole shares

        sign = 1.0 if direction == "long" else -1.0
        tps  = [
            round(entry_price + sign * r * stop_distance, 8)
            for r in cfg.tp_r_levels
        ]
        tp_units = [round(units * p, 8) for p in cfg.tp_scale_pcts]

        rr_ratio          = round(abs(tps[-1] - entry_price) / stop_distance, 4)
        breakeven_trigger = tps[0]

        # R:R filter — TP2 (primary exit) must meet the minimum
        if len(tps) >= 2:
            rr_to_tp2 = abs(tps[1] - entry_price) / stop_distance
            if rr_to_tp2 < cfg.min_rr:
                reasons.append(
                    f"R:R to TP2 is 1:{rr_to_tp2:.2f}, "
                    f"minimum required 1:{cfg.min_rr:.2f}"
                )

        return PositionSize(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_price=stop_price,
            stop_distance=round(stop_distance, 8),
            units=round(units, 8),
            risk_amount=round(risk_amount, 4),
            rr_ratio=rr_ratio,
            take_profits=tps,
            tp_units=tp_units,
            breakeven_trigger=round(breakeven_trigger, 8),
            valid=len(reasons) == 0,
            rejection_reasons=reasons,
        )

    # ── Individual filters ────────────────────────────────────────────────────

    def check_drawdown(
        self,
        current_balance:   float,
        day_start_balance: float,
    ) -> tuple[bool, float]:
        """
        Returns (can_trade, drawdown_pct).
        Halts when the day's loss equals or exceeds the configured limit.
        A profitable day (negative drawdown) always permits trading.
        """
        if day_start_balance <= 0:
            return True, 0.0
        drawdown = (day_start_balance - current_balance) / day_start_balance
        return drawdown < self.config.max_daily_drawdown_pct, round(drawdown, 6)

    def check_open_positions(self, n_open: int) -> tuple[bool, str]:
        """Returns (can_add, rejection_reason)."""
        limit = self.config.max_open_positions
        if n_open >= limit:
            return False, f"max open positions ({limit}) reached"
        return True, ""

    def check_correlation(
        self,
        symbol:             str,
        open_symbols:       list[str],
        correlation_matrix: dict | None = None,
    ) -> tuple[bool, str]:
        """
        Returns (passes, rejection_reason).
        Looks up correlation as (alphabetically_first, alphabetically_second)
        so key order in the matrix doesn't matter.
        """
        if not open_symbols or not correlation_matrix:
            return True, ""
        threshold = self.config.correlation_threshold
        for existing in open_symbols:
            key  = (min(symbol, existing), max(symbol, existing))
            corr = correlation_matrix.get(key, 0.0)
            if abs(corr) >= threshold:
                return False, (
                    f"correlation {symbol}/{existing} = {corr:.2f} "
                    f"≥ threshold {threshold:.2f}"
                )
        return True, ""

    # ── Per-bar stop management ───────────────────────────────────────────────

    def update_stop(
        self,
        position:      OpenPosition,
        current_price: float,
        tp_prices:     list[float] | None = None,
    ) -> OpenPosition:
        """
        Return a copy of *position* with current_stop updated.

        Two rules applied in order:
          1. Breakeven — move stop to entry_price the first time TP1 is hit.
          2. Trailing stop — continuously tighten if trailing_stop_enabled.

        The stop can only ever move in the favourable direction.
        Call once per bar close (or tick, for live data).
        """
        pos     = copy(position)
        cfg     = self.config
        is_long = pos.direction == "long"

        # 1. Breakeven
        if tp_prices and not pos.at_breakeven:
            tp1     = tp_prices[0]
            tp1_hit = (is_long and current_price >= tp1) or (
                      not is_long and current_price <= tp1)
            if tp1_hit:
                pos.current_stop = pos.entry_price
                pos.at_breakeven = True

        # 2. Trailing stop
        if cfg.trailing_stop_enabled:
            trail_dist = pos.entry_price * cfg.trailing_pct
            if is_long:
                if pos.high_water_mark == 0.0:
                    pos.high_water_mark = pos.entry_price
                pos.high_water_mark = max(pos.high_water_mark, current_price)
                candidate = pos.high_water_mark - trail_dist
                if candidate > pos.current_stop:
                    pos.current_stop = round(candidate, 8)
            else:
                if pos.high_water_mark == 0.0:
                    pos.high_water_mark = pos.entry_price
                pos.high_water_mark = min(pos.high_water_mark, current_price)
                candidate = pos.high_water_mark + trail_dist
                if candidate < pos.current_stop:
                    pos.current_stop = round(candidate, 8)

        return pos

    # ── Full approval pipeline ────────────────────────────────────────────────

    def approve_trade(
        self,
        entry_price:        float,
        stop_price:         float,
        account_balance:    float,
        day_start_balance:  float,
        direction:          str,
        symbol:             str = "",
        open_positions:     list[OpenPosition] | None = None,
        correlation_matrix: dict | None = None,
    ) -> PositionSize:
        """
        Run the complete risk pipeline and return a PositionSize.

        Checks (in order):
          1. Position sizing + R:R filter
          2. Daily drawdown guard
          3. Max open positions
          4. Correlation with existing positions

        All failures are accumulated into PositionSize.rejection_reasons.
        PositionSize.valid is False if any check fails.
        """
        open_positions = open_positions or []

        size = self.size_trade(entry_price, stop_price, account_balance,
                               direction, symbol)

        can, dd = self.check_drawdown(account_balance, day_start_balance)
        if not can:
            size.valid = False
            size.rejection_reasons.append(
                f"daily drawdown {dd:.1%} ≥ limit "
                f"{self.config.max_daily_drawdown_pct:.1%}"
            )

        ok, reason = self.check_open_positions(len(open_positions))
        if not ok:
            size.valid = False
            size.rejection_reasons.append(reason)

        open_syms = [p.symbol for p in open_positions]
        ok, reason = self.check_correlation(symbol, open_syms, correlation_matrix)
        if not ok:
            size.valid = False
            size.rejection_reasons.append(reason)

        return size
