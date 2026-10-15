"""Offline tests for persistent, fail-closed SPOT risk checks."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from bot.risk_manager import (
    AccountSnapshot,
    ExecutionCosts,
    InstrumentLimits,
    RiskManager,
    RiskStateError,
    RiskStatus,
)


NOW = datetime(2026, 10, 15, 12, 0, tzinfo=timezone.utc)
ACCOUNT = AccountSnapshot(
    as_of_utc=NOW,
    day_utc=NOW.date(),
    equity_usdt=Decimal("10000"),
    day_start_capital_usdt=Decimal("10000"),
    available_usdt=Decimal("10000"),
    realized_pnl_today_usdt=Decimal("0"),
    unrealized_pnl_usdt=Decimal("0"),
    open_positions=0,
    pending_orders=0,
    entries_today=0,
)
INSTRUMENT = InstrumentLimits(
    instrument_id="BTC-USDT",
    state="live",
    min_size=Decimal("0.0001"),
    lot_size=Decimal("0.0001"),
    tick_size=Decimal("0.01"),
    max_size=Decimal("1000"),
)
COSTS = ExecutionCosts(
    entry_fee_rate=Decimal("0.001"),
    exit_fee_rate=Decimal("0.001"),
    entry_slippage_rate=Decimal("0.001"),
    stop_slippage_rate=Decimal("0.001"),
)


class RiskManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.state_path = Path(directory.name) / "risk_state.json"
        self.manager = RiskManager(self.state_path)
        self.manager.initialize_day(ACCOUNT, NOW)

    def assess(self, **overrides: object):
        inputs = {
            "account": ACCOUNT,
            "instrument": INSTRUMENT,
            "costs": COSTS,
            "entry_price": Decimal("100"),
            "now_utc": NOW,
        }
        inputs.update(overrides)
        return self.manager.assess_buy(**inputs)

    def test_normal_buy_uses_risk_fees_slippage_and_balance(self) -> None:
        result = self.assess()
        self.assertEqual(result.status, RiskStatus.ALLOW)
        self.assertEqual(result.entry_price, Decimal("100"))
        self.assertEqual(result.stop_price, Decimal("99"))
        self.assertEqual(result.take_profit_price, Decimal("102"))
        # Per BTC: 100 * 1.001^2 - 99 * 0.999^2 = 1.398001 USDT.
        self.assertEqual(result.quantity, Decimal("35.7653"))
        self.assertEqual(result.estimated_risk_usdt, result.quantity * Decimal("1.398001"))
        self.assertLessEqual(result.estimated_risk_usdt, Decimal("50"))
        self.assertGreater((result.quantity + INSTRUMENT.lot_size) * Decimal("1.398001"), Decimal("50"))
        self.assertLessEqual(result.estimated_cost_usdt, ACCOUNT.available_usdt)

    def test_insufficient_available_usdt_denies_minimum_lot(self) -> None:
        result = self.assess(account=replace(ACCOUNT, available_usdt=Decimal("0.005")))
        self.assertEqual(result.status, RiskStatus.DENY)
        self.assertEqual(result.quantity, Decimal("0"))
        self.assertIn("USDT", result.reason)

    def test_available_balance_caps_quantity_before_risk_budget(self) -> None:
        result = self.assess(account=replace(ACCOUNT, available_usdt=Decimal("100")))
        self.assertEqual(result.status, RiskStatus.ALLOW)
        self.assertLessEqual(result.estimated_cost_usdt, Decimal("100"))
        self.assertGreater((result.quantity + INSTRUMENT.lot_size) * Decimal("100.2001"), Decimal("100"))
        self.assertLess(result.estimated_risk_usdt, Decimal("50"))

    def test_minimum_lot_above_risk_budget_is_denied(self) -> None:
        instrument = replace(INSTRUMENT, min_size=Decimal("40"))
        result = self.assess(instrument=instrument)
        self.assertEqual(result.status, RiskStatus.DENY)
        self.assertIn("Минимальный лот", result.reason)

    def test_daily_realized_and_unrealized_loss_lock_survives_restart(self) -> None:
        loss = replace(ACCOUNT, realized_pnl_today_usdt=Decimal("-150"), unrealized_pnl_usdt=Decimal("-50"))
        denied = self.assess(account=loss)
        self.assertEqual(denied.status, RiskStatus.DENY)
        self.assertIn("Достигнут дневной лимит", denied.reason)
        state = json.loads(self.state_path.read_text(encoding="utf-8"))
        self.assertTrue(state["loss_locked"])
        self.assertEqual(state["start_capital_usdt"], "10000")

        restarted = RiskManager(self.state_path)
        recovered = restarted.assess_buy(account=ACCOUNT, instrument=INSTRUMENT, costs=COSTS, entry_price=Decimal("100"), now_utc=NOW)
        self.assertEqual(recovered.status, RiskStatus.DENY)
        self.assertIn("уже заблокировал", recovered.reason)
        with self.assertRaisesRegex(RiskStateError, "сброс в те же сутки запрещён"):
            restarted.initialize_day(ACCOUNT, NOW)

    def test_realized_loss_is_not_hidden_by_unrealized_gain(self) -> None:
        account = replace(ACCOUNT, realized_pnl_today_usdt=Decimal("-200"), unrealized_pnl_usdt=Decimal("100"))
        self.assertEqual(self.assess(account=account).status, RiskStatus.DENY)
        self.assertTrue(json.loads(self.state_path.read_text(encoding="utf-8"))["loss_locked"])

    def test_open_position_and_pending_order_block_new_buy(self) -> None:
        for account in (replace(ACCOUNT, open_positions=1), replace(ACCOUNT, pending_orders=1)):
            with self.subTest(account=account):
                self.assertEqual(self.assess(account=account).status, RiskStatus.DENY)

    def test_five_daily_entries_block_another_entry(self) -> None:
        account = replace(ACCOUNT, entries_today=5)
        result = self.assess(account=account)
        self.assertEqual(result.status, RiskStatus.DENY)
        self.assertIn("суточный лимит", result.reason)
        restarted = RiskManager(self.state_path)
        self.assertEqual(restarted.assess_buy(account=ACCOUNT, instrument=INSTRUMENT, costs=COSTS, entry_price=Decimal("100"), now_utc=NOW).status, RiskStatus.DENY)

    def test_repeat_entry_is_blocked_until_confirmed_reconciliation(self) -> None:
        self.assertEqual(self.assess().status, RiskStatus.ALLOW)
        self.assertIn("зарезервирован", self.assess().reason)
        restarted = RiskManager(self.state_path)
        repeated = restarted.assess_buy(account=ACCOUNT, instrument=INSTRUMENT, costs=COSTS, entry_price=Decimal("100"), now_utc=NOW)
        self.assertEqual(repeated.status, RiskStatus.DENY)
        confirmed_open = replace(ACCOUNT, entries_today=1, open_positions=1)
        self.assertTrue(restarted.reconcile_confirmed_entry(confirmed_open, NOW))
        self.assertEqual(self.assess(account=confirmed_open).status, RiskStatus.DENY)
        self.assertIn("противоречит", self.assess(account=ACCOUNT).reason)
        confirmed_closed = replace(confirmed_open, open_positions=0)
        self.assertEqual(self.assess(account=confirmed_closed).status, RiskStatus.ALLOW)

    def test_quantity_and_prices_round_conservatively_to_okx_steps(self) -> None:
        instrument = replace(INSTRUMENT, lot_size=Decimal("0.1"), min_size=Decimal("0.1"), tick_size=Decimal("0.05"), max_size=Decimal("10.19"))
        result = self.assess(instrument=instrument, entry_price=Decimal("100.02"))
        self.assertEqual(result.status, RiskStatus.ALLOW)
        self.assertEqual(result.entry_price, Decimal("100.05"))
        self.assertEqual(result.stop_price, Decimal("99.00"))
        self.assertEqual(result.take_profit_price, Decimal("102.10"))
        self.assertEqual(result.quantity, Decimal("10.1"))
        self.assertLessEqual(result.estimated_risk_usdt, Decimal("50"))

    def test_remaining_daily_budget_caps_trade_risk(self) -> None:
        account = replace(ACCOUNT, realized_pnl_today_usdt=Decimal("-199"))
        result = self.assess(account=account)
        self.assertEqual(result.status, RiskStatus.ALLOW)
        self.assertLessEqual(result.estimated_risk_usdt, Decimal("1"))
        self.assertGreater((result.quantity + INSTRUMENT.lot_size) * Decimal("1.398001"), Decimal("1"))

    def test_missing_corrupt_stale_and_contradictory_data_deny(self) -> None:
        self.assertEqual(self.manager.assess_buy().status, RiskStatus.DENY)
        bad_cases = {
            "missing_account": {"account": None},
            "missing_instrument": {"instrument": None},
            "missing_costs": {"costs": None},
            "missing_price": {"entry_price": None},
            "missing_time": {"now_utc": None},
            "negative_price": {"entry_price": Decimal("-1")},
            "stale_account": {"account": replace(ACCOUNT, as_of_utc=NOW - timedelta(minutes=5))},
            "wrong_day_start": {"account": replace(ACCOUNT, day_start_capital_usdt=Decimal("9999"))},
            "balance_exceeds_equity": {"account": replace(ACCOUNT, available_usdt=Decimal("10001"))},
            "bad_pnl": {"account": replace(ACCOUNT, unrealized_pnl_usdt=Decimal("NaN"))},
            "bad_instrument": {"instrument": replace(INSTRUMENT, max_size=Decimal("0.00001"))},
            "bad_fee": {"costs": replace(COSTS, entry_fee_rate=Decimal("NaN"))},
        }
        for name, changes in bad_cases.items():
            with self.subTest(case=name):
                result = self.assess(**changes)
                self.assertEqual(result.status, RiskStatus.DENY)
                self.assertEqual(result.quantity, Decimal("0"))
        self.state_path.write_text("{broken", encoding="utf-8")
        self.assertEqual(self.assess().status, RiskStatus.DENY)

    def test_missing_state_and_stale_lock_fail_closed(self) -> None:
        self.state_path.unlink()
        self.assertEqual(self.assess().status, RiskStatus.DENY)
        self.manager.initialize_day(ACCOUNT, NOW)
        Path(f"{self.state_path}.lock").write_text("stale", encoding="utf-8")
        self.assertEqual(self.assess().status, RiskStatus.DENY)

    def test_failed_reservation_write_leaves_fail_closed_lock(self) -> None:
        with patch.object(self.manager, "_save", side_effect=OSError("disk full")):
            self.assertEqual(self.assess().status, RiskStatus.DENY)
        self.assertTrue(Path(f"{self.state_path}.lock").exists())
        restarted = RiskManager(self.state_path)
        self.assertEqual(restarted.assess_buy(account=ACCOUNT, instrument=INSTRUMENT, costs=COSTS, entry_price=Decimal("100"), now_utc=NOW).status, RiskStatus.DENY)

    def test_utc_day_does_not_reset_until_explicit_new_baseline(self) -> None:
        tomorrow = datetime(2026, 10, 16, 0, 0, tzinfo=timezone.utc)
        next_day = replace(ACCOUNT, as_of_utc=tomorrow, day_utc=date(2026, 10, 16), equity_usdt=Decimal("9000"), day_start_capital_usdt=Decimal("9000"), available_usdt=Decimal("9000"))
        self.assertEqual(self.assess(account=next_day, now_utc=tomorrow).status, RiskStatus.DENY)
        self.manager.initialize_day(next_day, tomorrow)
        self.assertEqual(self.assess(account=next_day, now_utc=tomorrow).status, RiskStatus.ALLOW)


if __name__ == "__main__":
    unittest.main()
