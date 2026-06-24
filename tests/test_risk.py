"""
Tests for the risk management module.

The final class (TestTradeSizingDemo) prints a human-readable trade
breakdown.  Run it directly to see full output:

    pytest tests/test_risk.py::TestTradeSizingDemo -s
"""
import pytest

from risk.models import RiskConfig, PositionSize, OpenPosition
from risk.manager import RiskManager


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rm(**kw) -> RiskManager:
    return RiskManager(RiskConfig(**kw))


def _open_pos(symbol, direction="long", entry=100.0, stop=95.0, units=10.0):
    return OpenPosition(
        symbol=symbol, direction=direction,
        entry_price=entry, current_stop=stop,
        units_remaining=units,
    )


# ── RiskConfig defaults ───────────────────────────────────────────────────────

class TestRiskConfigDefaults:
    def test_risk_pct_default(self):
        assert RiskConfig().risk_pct == 0.01

    def test_min_rr_default(self):
        assert RiskConfig().min_rr == 2.0

    def test_max_drawdown_default(self):
        assert RiskConfig().max_daily_drawdown_pct == 0.05

    def test_max_open_positions_default(self):
        assert RiskConfig().max_open_positions == 5

    def test_tp_levels_default(self):
        assert RiskConfig().tp_r_levels == (1.0, 2.0, 3.0)

    def test_tp_scale_pcts_sum_to_one(self):
        assert sum(RiskConfig().tp_scale_pcts) == pytest.approx(1.0)

    def test_trailing_stop_off_by_default(self):
        assert RiskConfig().trailing_stop_enabled is False


# ── Position sizing: LONG ─────────────────────────────────────────────────────

class TestPositionSizingLong:
    """
    Core maths:
      entry=$100  stop=$95  account=$10,000  risk=1%
      → risk_amount=$100  stop_dist=$5  units=20
      → TP1=$105  TP2=$110  TP3=$115
      → tp_units=[8, 7, 5]  (40/35/25%)  R:R=1:3
    """

    @pytest.fixture
    def size(self):
        return _rm(risk_pct=0.01).size_trade(
            entry_price=100.0,
            stop_price=95.0,
            account_balance=10_000.0,
            direction="long",
            symbol="AAPL",
        )

    def test_valid(self, size):
        assert size.valid, size.rejection_reasons

    def test_units(self, size):
        assert size.units == pytest.approx(20.0)

    def test_risk_amount(self, size):
        assert size.risk_amount == pytest.approx(100.0)

    def test_stop_distance(self, size):
        assert size.stop_distance == pytest.approx(5.0)

    def test_tp1_price(self, size):
        assert size.take_profits[0] == pytest.approx(105.0)

    def test_tp2_price(self, size):
        assert size.take_profits[1] == pytest.approx(110.0)

    def test_tp3_price(self, size):
        assert size.take_profits[2] == pytest.approx(115.0)

    def test_rr_ratio_to_tp3(self, size):
        assert size.rr_ratio == pytest.approx(3.0)

    def test_breakeven_trigger_equals_tp1(self, size):
        assert size.breakeven_trigger == pytest.approx(105.0)

    def test_tp_units_40pct(self, size):
        assert size.tp_units[0] == pytest.approx(8.0)   # 40% of 20

    def test_tp_units_35pct(self, size):
        assert size.tp_units[1] == pytest.approx(7.0)   # 35% of 20

    def test_tp_units_25pct(self, size):
        assert size.tp_units[2] == pytest.approx(5.0)   # 25% of 20

    def test_tp_units_sum_to_total(self, size):
        assert sum(size.tp_units) == pytest.approx(size.units)

    def test_direction_preserved(self, size):
        assert size.direction == "long"

    def test_symbol_preserved(self, size):
        assert size.symbol == "AAPL"

    def test_three_tp_levels(self, size):
        assert len(size.take_profits) == 3

    def test_all_tps_above_entry_for_long(self, size):
        for tp in size.take_profits:
            assert tp > size.entry_price


# ── Position sizing: SHORT ────────────────────────────────────────────────────

class TestPositionSizingShort:
    """
    entry=$200  stop=$210  account=$10,000  risk=1%
    → stop_dist=$10  units=10
    → TP1=$190  TP2=$180  TP3=$170  R:R=1:3
    """

    @pytest.fixture
    def size(self):
        return _rm(risk_pct=0.01).size_trade(
            entry_price=200.0,
            stop_price=210.0,
            account_balance=10_000.0,
            direction="short",
            symbol="TSLA",
        )

    def test_valid(self, size):
        assert size.valid, size.rejection_reasons

    def test_units(self, size):
        assert size.units == pytest.approx(10.0)

    def test_tp1_below_entry(self, size):
        assert size.take_profits[0] == pytest.approx(190.0)

    def test_tp2_below_entry(self, size):
        assert size.take_profits[1] == pytest.approx(180.0)

    def test_tp3_below_entry(self, size):
        assert size.take_profits[2] == pytest.approx(170.0)

    def test_rr_ratio(self, size):
        assert size.rr_ratio == pytest.approx(3.0)

    def test_breakeven_trigger_equals_tp1(self, size):
        assert size.breakeven_trigger == pytest.approx(190.0)

    def test_all_tps_below_entry_for_short(self, size):
        for tp in size.take_profits:
            assert tp < size.entry_price


# ── Risk % scaling ────────────────────────────────────────────────────────────

class TestRiskPercentage:
    def test_two_pct_doubles_units(self):
        rm = _rm(risk_pct=0.02)
        size = rm.size_trade(100.0, 95.0, 10_000.0, "long")
        assert size.risk_amount == pytest.approx(200.0)
        assert size.units       == pytest.approx(40.0)

    def test_half_pct_halves_units(self):
        rm = _rm(risk_pct=0.005)
        size = rm.size_trade(100.0, 95.0, 10_000.0, "long")
        assert size.risk_amount == pytest.approx(50.0)
        assert size.units       == pytest.approx(10.0)

    def test_wider_stop_reduces_units(self):
        rm = _rm(risk_pct=0.01)
        # stop_dist=10 vs. 5 → half the units
        size = rm.size_trade(100.0, 90.0, 10_000.0, "long")
        assert size.units == pytest.approx(10.0)

    def test_larger_account_increases_units(self):
        rm = _rm(risk_pct=0.01)
        size = rm.size_trade(100.0, 95.0, 20_000.0, "long")
        assert size.units == pytest.approx(40.0)


# ── No fractional units ───────────────────────────────────────────────────────

class TestNoFractionalUnits:
    def test_floors_to_whole_units(self):
        rm = _rm(risk_pct=0.01, allow_fractional_units=False)
        # exact units = 100/7 ≈ 14.28 → floored to 14
        size = rm.size_trade(100.0, 93.0, 10_000.0, "long")
        assert size.units == pytest.approx(14.0)

    def test_fractional_allowed_by_default(self):
        rm = _rm(risk_pct=0.01)
        size = rm.size_trade(100.0, 93.0, 10_000.0, "long")
        assert size.units > 14.0


# ── R:R filter ────────────────────────────────────────────────────────────────

class TestRRFilter:
    def test_default_rr_levels_pass_default_min(self):
        # TP2 = 2R, min_rr = 2.0 → passes
        rm = _rm(min_rr=2.0)
        size = rm.size_trade(100.0, 95.0, 10_000.0, "long")
        assert size.valid

    def test_min_rr_higher_than_tp2_rejects(self):
        # min_rr=3.0 but TP2 is only 2R → fails
        rm = _rm(min_rr=3.0)
        size = rm.size_trade(100.0, 95.0, 10_000.0, "long")
        assert not size.valid
        assert any("R:R" in r for r in size.rejection_reasons)

    def test_rejection_reason_shows_actual_and_required(self):
        rm = _rm(min_rr=3.0)
        size = rm.size_trade(100.0, 95.0, 10_000.0, "long")
        reason = size.rejection_reasons[0]
        assert "2.00" in reason   # actual R:R to TP2
        assert "3.00" in reason   # required minimum

    def test_custom_tp_levels_that_fail(self):
        # TP2 at 1R with min_rr=2.0 → fails
        rm = _rm(tp_r_levels=(0.5, 1.0, 1.5), min_rr=2.0)
        size = rm.size_trade(100.0, 95.0, 10_000.0, "long")
        assert not size.valid

    def test_zero_stop_distance_rejected(self):
        rm = _rm()
        size = rm.size_trade(100.0, 100.0, 10_000.0, "long")
        assert not size.valid
        assert any("zero" in r for r in size.rejection_reasons)


# ── Daily drawdown filter ─────────────────────────────────────────────────────

class TestDrawdownFilter:
    def test_no_loss_allows_trading(self):
        rm = _rm(max_daily_drawdown_pct=0.05)
        can, pct = rm.check_drawdown(10_000.0, 10_000.0)
        assert can
        assert pct == pytest.approx(0.0)

    def test_small_loss_allows_trading(self):
        rm = _rm(max_daily_drawdown_pct=0.05)
        can, pct = rm.check_drawdown(9_700.0, 10_000.0)   # 3% loss
        assert can
        assert pct == pytest.approx(0.03)

    def test_exactly_at_limit_halts(self):
        rm = _rm(max_daily_drawdown_pct=0.05)
        can, _ = rm.check_drawdown(9_500.0, 10_000.0)     # exactly 5%
        assert not can

    def test_beyond_limit_halts(self):
        rm = _rm(max_daily_drawdown_pct=0.05)
        can, pct = rm.check_drawdown(9_400.0, 10_000.0)   # 6%
        assert not can
        assert pct == pytest.approx(0.06)

    def test_profitable_day_always_allowed(self):
        rm = _rm(max_daily_drawdown_pct=0.05)
        can, pct = rm.check_drawdown(10_500.0, 10_000.0)  # up on the day
        assert can
        assert pct < 0

    def test_approve_trade_blocked_by_drawdown(self):
        rm = _rm(max_daily_drawdown_pct=0.05)
        size = rm.approve_trade(
            entry_price=100.0, stop_price=95.0,
            account_balance=9_400.0,       # 6% below day start
            day_start_balance=10_000.0,
            direction="long",
        )
        assert not size.valid
        assert any("drawdown" in r for r in size.rejection_reasons)

    def test_drawdown_reason_shows_percentages(self):
        rm = _rm(max_daily_drawdown_pct=0.05)
        size = rm.approve_trade(
            entry_price=100.0, stop_price=95.0,
            account_balance=9_400.0, day_start_balance=10_000.0,
            direction="long",
        )
        reason = next(r for r in size.rejection_reasons if "drawdown" in r)
        assert "6.0%" in reason
        assert "5.0%" in reason


# ── Max open positions filter ─────────────────────────────────────────────────

class TestMaxOpenPositions:
    def test_below_limit_passes(self):
        ok, _ = _rm(max_open_positions=5).check_open_positions(4)
        assert ok

    def test_at_limit_rejects(self):
        ok, reason = _rm(max_open_positions=5).check_open_positions(5)
        assert not ok
        assert "5" in reason

    def test_above_limit_rejects(self):
        ok, _ = _rm(max_open_positions=5).check_open_positions(7)
        assert not ok

    def test_zero_limit_always_rejects(self):
        ok, _ = _rm(max_open_positions=0).check_open_positions(0)
        assert not ok

    def test_approve_trade_blocked_at_max_positions(self):
        rm   = _rm(max_open_positions=3)
        open_pos = [
            _open_pos("A"), _open_pos("B"), _open_pos("C"),
        ]
        size = rm.approve_trade(
            entry_price=100.0, stop_price=95.0,
            account_balance=10_000.0, day_start_balance=10_000.0,
            direction="long", symbol="D",
            open_positions=open_pos,
        )
        assert not size.valid
        assert any("max open" in r for r in size.rejection_reasons)


# ── Correlation filter ────────────────────────────────────────────────────────

CORR_MATRIX = {
    ("AAPL", "MSFT"): 0.85,
    ("AAPL", "GOOG"): 0.80,
    ("MSFT", "GOOG"): 0.75,
    ("AAPL", "XOM"):  0.20,
    ("GOOG", "XOM"):  0.15,
}


class TestCorrelationFilter:
    def test_no_open_positions_always_passes(self):
        ok, _ = _rm().check_correlation("AAPL", [], CORR_MATRIX)
        assert ok

    def test_no_matrix_always_passes(self):
        ok, _ = _rm().check_correlation("AAPL", ["MSFT"], None)
        assert ok

    def test_low_correlation_passes(self):
        ok, _ = _rm(correlation_threshold=0.70).check_correlation(
            "AAPL", ["XOM"], CORR_MATRIX)
        assert ok

    def test_high_correlation_rejects(self):
        ok, reason = _rm(correlation_threshold=0.70).check_correlation(
            "AAPL", ["MSFT"], CORR_MATRIX)
        assert not ok
        assert "MSFT" in reason
        assert "0.85" in reason

    def test_key_order_is_symmetric(self):
        # ("MSFT", "AAPL") must find the ("AAPL", "MSFT") entry
        ok, _ = _rm(correlation_threshold=0.70).check_correlation(
            "MSFT", ["AAPL"], CORR_MATRIX)
        assert not ok

    def test_multiple_open_any_match_rejects(self):
        # XOM is fine but GOOG is correlated → should reject
        ok, reason = _rm(correlation_threshold=0.70).check_correlation(
            "AAPL", ["XOM", "GOOG"], CORR_MATRIX)
        assert not ok
        assert "GOOG" in reason

    def test_approve_trade_blocked_by_correlation(self):
        rm       = _rm(correlation_threshold=0.70)
        open_pos = [OpenPosition("MSFT", "long", 300, 290, 5)]
        size     = rm.approve_trade(
            entry_price=100.0, stop_price=95.0,
            account_balance=10_000.0, day_start_balance=10_000.0,
            direction="long", symbol="AAPL",
            open_positions=open_pos,
            correlation_matrix=CORR_MATRIX,
        )
        assert not size.valid
        assert any("corr" in r.lower() for r in size.rejection_reasons)


# ── Trailing stop ─────────────────────────────────────────────────────────────

class TestTrailingStop:
    def _long_pos(self, entry=100.0, stop=95.0):
        return OpenPosition("AAPL", "long", entry, stop, 20,
                            high_water_mark=entry)

    def _short_pos(self, entry=200.0, stop=210.0):
        return OpenPosition("META", "short", entry, stop, 10,
                            high_water_mark=entry)

    def test_disabled_stop_unchanged(self):
        rm = _rm(trailing_stop_enabled=False)
        pos = rm.update_stop(self._long_pos(), current_price=110.0)
        assert pos.current_stop == pytest.approx(95.0)

    def test_long_stop_trails_up_on_new_high(self):
        rm  = _rm(trailing_stop_enabled=True, trailing_pct=0.02)
        pos = rm.update_stop(self._long_pos(), current_price=110.0)
        # trail_dist = 100*0.02 = 2  →  new_stop = 110 - 2 = 108
        assert pos.current_stop == pytest.approx(108.0)

    def test_long_stop_never_retreats(self):
        rm  = _rm(trailing_stop_enabled=True, trailing_pct=0.02)
        pos = self._long_pos()
        pos = rm.update_stop(pos, current_price=110.0)   # stop → 108
        pos = rm.update_stop(pos, current_price=106.0)   # pullback — stop stays 108
        assert pos.current_stop == pytest.approx(108.0)

    def test_long_stop_keeps_advancing(self):
        rm  = _rm(trailing_stop_enabled=True, trailing_pct=0.02)
        pos = self._long_pos()
        pos = rm.update_stop(pos, current_price=110.0)   # stop → 108
        pos = rm.update_stop(pos, current_price=120.0)   # stop → 118
        assert pos.current_stop == pytest.approx(118.0)

    def test_short_stop_trails_down_on_new_low(self):
        rm  = _rm(trailing_stop_enabled=True, trailing_pct=0.02)
        pos = rm.update_stop(self._short_pos(), current_price=180.0)
        # trail_dist = 200*0.02 = 4  →  new_stop = 180 + 4 = 184
        assert pos.current_stop == pytest.approx(184.0)

    def test_short_stop_never_retreats(self):
        rm  = _rm(trailing_stop_enabled=True, trailing_pct=0.02)
        pos = self._short_pos()
        pos = rm.update_stop(pos, current_price=180.0)   # stop → 184
        pos = rm.update_stop(pos, current_price=185.0)   # bounce — stop stays 184
        assert pos.current_stop == pytest.approx(184.0)

    def test_high_water_mark_initialised_to_entry_if_zero(self):
        rm  = _rm(trailing_stop_enabled=True, trailing_pct=0.02)
        pos = OpenPosition("AAPL", "long", 100.0, 95.0, 20)  # hwm=0
        pos = rm.update_stop(pos, current_price=110.0)
        # hwm was 0 → initialised to entry (100) → max(100, 110) = 110
        assert pos.current_stop == pytest.approx(108.0)


# ── Breakeven after TP1 ───────────────────────────────────────────────────────

class TestBreakevenAfterTP1:
    TP_PRICES = [105.0, 110.0, 115.0]

    def _pos(self):
        return OpenPosition("AAPL", "long", 100.0, 95.0, 20)

    def test_stop_moves_to_entry_on_tp1_hit(self):
        rm  = _rm()
        pos = rm.update_stop(self._pos(), 105.0, self.TP_PRICES)
        assert pos.current_stop   == pytest.approx(100.0)
        assert pos.at_breakeven   is True

    def test_stop_unchanged_before_tp1(self):
        rm  = _rm()
        pos = rm.update_stop(self._pos(), 104.99, self.TP_PRICES)
        assert pos.current_stop == pytest.approx(95.0)
        assert pos.at_breakeven is False

    def test_breakeven_not_re_applied(self):
        rm  = _rm()
        pos = rm.update_stop(self._pos(), 105.0, self.TP_PRICES)
        # manually advance stop (e.g. from trailing)
        pos.current_stop = 107.0
        pos = rm.update_stop(pos, 108.0, self.TP_PRICES)   # already at_breakeven=True
        assert pos.current_stop == pytest.approx(107.0)    # NOT reverted to 100

    def test_breakeven_short_direction(self):
        rm  = _rm()
        pos = OpenPosition("META", "short", 200.0, 210.0, 10)
        pos = rm.update_stop(pos, 190.0, [190.0, 180.0, 170.0])
        assert pos.current_stop == pytest.approx(200.0)    # entry
        assert pos.at_breakeven is True

    def test_breakeven_short_not_triggered_early(self):
        rm  = _rm()
        pos = OpenPosition("META", "short", 200.0, 210.0, 10)
        pos = rm.update_stop(pos, 191.0, [190.0, 180.0, 170.0])
        assert pos.current_stop == pytest.approx(210.0)    # unchanged
        assert pos.at_breakeven is False


# ── Multiple rejections accumulate ───────────────────────────────────────────

class TestMultipleRejections:
    def test_drawdown_and_max_positions_both_recorded(self):
        rm = _rm(max_open_positions=0, max_daily_drawdown_pct=0.0)
        size = rm.approve_trade(
            entry_price=100.0, stop_price=95.0,
            account_balance=9_999.0, day_start_balance=10_000.0,
            direction="long", symbol="AAPL",
        )
        assert not size.valid
        assert len(size.rejection_reasons) >= 2

    def test_all_four_checks_can_fail(self):
        rm = _rm(
            min_rr=3.0,                     # R:R will fail (TP2=2R < 3R)
            max_daily_drawdown_pct=0.0,     # any loss halts
            max_open_positions=0,           # no positions allowed
            correlation_threshold=0.0,      # any correlation fails
        )
        corr = {("AAPL", "MSFT"): 0.01}
        open_pos = [OpenPosition("MSFT", "long", 300, 290, 5)]
        size = rm.approve_trade(
            entry_price=100.0, stop_price=95.0,
            account_balance=9_999.0, day_start_balance=10_000.0,
            direction="long", symbol="AAPL",
            open_positions=open_pos, correlation_matrix=corr,
        )
        assert not size.valid
        assert len(size.rejection_reasons) == 4


# ── Full pipeline approval ────────────────────────────────────────────────────

class TestApproveTradePipeline:
    def test_clean_trade_approved(self):
        rm   = _rm()
        size = rm.approve_trade(
            entry_price=150.0, stop_price=145.0,
            account_balance=20_000.0, day_start_balance=20_000.0,
            direction="long", symbol="SPY",
        )
        assert size.valid
        assert size.units == pytest.approx(40.0)   # risk=$200, dist=$5

    def test_returns_position_size_type(self):
        rm     = _rm()
        result = rm.approve_trade(100.0, 95.0, 10_000.0, 10_000.0, "long")
        assert isinstance(result, PositionSize)

    def test_valid_trade_has_no_rejection_reasons(self):
        rm   = _rm()
        size = rm.approve_trade(100.0, 95.0, 10_000.0, 10_000.0, "long")
        assert size.rejection_reasons == []


# ── Human-readable demo (run with -s to see output) ──────────────────────────

class TestTradeSizingDemo:
    """
    pytest tests/test_risk.py::TestTradeSizingDemo -s
    """

    def _header(self, text):
        return "\n  " + "─" * 58 + f"\n  {text}\n  " + "─" * 58

    def test_long_trade_aapl(self, capsys):
        """entry=$100  stop=$95  account=$10k  risk=1%"""
        rm   = RiskManager(RiskConfig(risk_pct=0.01))
        size = rm.approve_trade(
            entry_price=100.00, stop_price=95.00,
            account_balance=10_000.00, day_start_balance=10_000.00,
            direction="long", symbol="AAPL",
        )
        assert size.valid, size.rejection_reasons

        print(self._header("LONG  AAPL  —  1% risk on $10,000 account"))
        print(f"  Entry price     :  ${size.entry_price:.2f}")
        print(f"  Stop loss       :  ${size.stop_price:.2f}  "
              f"(${size.stop_distance:.2f} below entry)")
        print(f"  Risk amount     :  ${size.risk_amount:.2f}  "
              f"({size.risk_amount / 10_000:.0%} of account)")
        print(f"  Position size   :  {size.units:.0f} units")
        print(f"  R:R ratio       :  1:{size.rr_ratio:.1f}  (to TP3)")
        print()
        scale_pcts = RiskConfig().tp_scale_pcts
        for i, (tp, u, pct) in enumerate(
                zip(size.take_profits, size.tp_units, scale_pcts), 1):
            profit = (tp - size.entry_price) * u
            print(f"  TP{i}  ${tp:.2f}   close {u:.0f} units ({pct:.0%})"
                  f"   profit ≈ ${profit:.2f}   [{i}R]")
        print(f"\n  Breakeven trigger: price ≥ TP1 (${size.take_profits[0]:.2f})"
              f"  →  stop moves to ${size.entry_price:.2f}")
        print("  " + "─" * 58)

        out = capsys.readouterr().out
        assert "$105.00" in out  # TP1
        assert "$110.00" in out  # TP2
        assert "$115.00" in out  # TP3
        assert "20 units" in out

    def test_short_trade_meta(self, capsys):
        """entry=$200  stop=$208  account=$10k  risk=1.5%"""
        rm   = RiskManager(RiskConfig(risk_pct=0.015))
        size = rm.approve_trade(
            entry_price=200.00, stop_price=208.00,
            account_balance=10_000.00, day_start_balance=10_000.00,
            direction="short", symbol="META",
        )
        assert size.valid, size.rejection_reasons

        print(self._header("SHORT  META  —  1.5% risk on $10,000 account"))
        print(f"  Entry price     :  ${size.entry_price:.2f}")
        print(f"  Stop loss       :  ${size.stop_price:.2f}  "
              f"(${size.stop_distance:.2f} above entry)")
        print(f"  Risk amount     :  ${size.risk_amount:.2f}  "
              f"({size.risk_amount / 10_000:.1%} of account)")
        print(f"  Position size   :  {size.units:.4g} units")
        print(f"  R:R ratio       :  1:{size.rr_ratio:.1f}  (to TP3)")
        print()
        scale_pcts = RiskConfig().tp_scale_pcts
        for i, (tp, u, pct) in enumerate(
                zip(size.take_profits, size.tp_units, scale_pcts), 1):
            profit = (size.entry_price - tp) * u
            print(f"  TP{i}  ${tp:.2f}   close {u:.4g} units ({pct:.0%})"
                  f"   profit ≈ ${profit:.2f}   [{i}R]")
        print(f"\n  Breakeven trigger: price ≤ TP1 (${size.take_profits[0]:.2f})"
              f"  →  stop moves to ${size.entry_price:.2f}")
        print("  " + "─" * 58)

        out = capsys.readouterr().out
        # entry=200, stop_dist=8 → TP1=192, TP2=184, TP3=176
        assert "$192.00" in out
        assert "$184.00" in out
        assert "$176.00" in out

    def test_trade_blocked_by_drawdown(self, capsys):
        """Show the rejection message when daily loss limit is hit."""
        rm   = RiskManager(RiskConfig(max_daily_drawdown_pct=0.05))
        size = rm.approve_trade(
            entry_price=100.00, stop_price=95.00,
            account_balance=9_300.00,      # down 7% on the day
            day_start_balance=10_000.00,
            direction="long", symbol="AAPL",
        )
        assert not size.valid

        print(self._header("BLOCKED — daily drawdown limit reached"))
        for reason in size.rejection_reasons:
            print(f"  ✗  {reason}")
        print("  " + "─" * 58)

        out = capsys.readouterr().out
        assert "drawdown" in out.lower()
        assert "7.0%" in out
