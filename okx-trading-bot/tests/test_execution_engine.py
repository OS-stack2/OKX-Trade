"""Offline execution-preparation state machine; never sends an exchange request."""

from __future__ import annotations

from dataclasses import replace
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import sqlite3
import unittest
from unittest.mock import patch

import main
from bot.execution_engine import (ExecutionEngine, ExecutionError, ExecutionState,
                                  RECOVERY_PHRASE, RecoveryProof)
from bot.journal import CycleDraft, CycleStatus, DecisionJournal
from bot.market_data import SpotInstrument
from bot.reconcile import ReconcileReport, ReconcileStatus
from bot.risk_manager import AccountSnapshot, ExecutionCosts, InstrumentLimits, RiskManager
from bot.strategy_budget import BudgetBaseline, StrategyBudgetStore


NOW = datetime(2026, 10, 15, 12, 30, tzinfo=timezone.utc)
D = Decimal
LIMITS = InstrumentLimits("BTC-USDT", "live", D("0.0001"), D("0.0001"), D("0.1"), D("0.1"))
SPOT = SpotInstrument("BTC-USDT", D("0.1"), D("0.0001"), D("0.0001"), "live", "BTC", "USDT")
COSTS = ExecutionCosts(D("0.001"), D("0.001"), D("0.001"), D("0.001"))


class ExecutionEngineTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.journal = DecisionJournal(self.root / "journal.sqlite3")
        self.manager = RiskManager(self.root / "risk.json")
        self.store = StrategyBudgetStore(self.journal.path)
        self.manager.initialize_day(AccountSnapshot(
            NOW, NOW.date(), D("5000"), D("5000"), D("5000"), D(0), D(0), 0, 0, 0), NOW)
        self.journal.append(CycleDraft(NOW - timedelta(seconds=20),
                           NOW - timedelta(minutes=30), "HOLD", CycleStatus.HOLD, "earlier"))
        self.store.confirm_once(BudgetBaseline(NOW - timedelta(minutes=2), D("5000"),
            D("1"), D("1"), D("5000"), D("5000"), D("80000"), D("85000"),
            D("-0.0008"), D("-0.001"), 1))
        self.report = ReconcileReport(ReconcileStatus.READY, (), checked_at_utc=NOW,
            btc_total=D("1"), btc_available=D("1"), usdt_total=D("5000"),
            usdt_available=D("5000"), external_btc=D("1"), bot_btc=D(0),
            open_order_count=0, fill_count=0, instrument=SPOT,
            confirmed_realized_pnl_usdt=D(0), confirmed_unrealized_pnl_usdt=D(0),
            daily_loss_usdt=D(0), daily_loss_limit_usdt=D("100"),
            daily_risk_remaining_usdt=D("100"), strategy_capital_usdt=D("5000"),
            strategy_equity_usdt=D("5000"), per_trade_risk_limit_usdt=D("25"),
            taker_fee_rate=D("-0.001"))
        self.engine = ExecutionEngine(journal=self.journal, risk_manager=self.manager,
                                      budget_store=self.store, clock=lambda: NOW)
        self.decision = self.manager.preview_buy(account=AccountSnapshot(
            NOW, NOW.date(), D("5000"), D("5000"), D("5000"), D(0), D(0), 0, 0, 0),
            instrument=LIMITS, costs=COSTS, entry_price=D("80000"), now_utc=NOW)
        self.assertEqual(self.decision.status.value, "ALLOW")
        self.record = self.journal.append(CycleDraft(
            NOW, NOW - timedelta(minutes=15), "BUY", CycleStatus.ALLOW, "test BUY",
            quantity=self.decision.quantity, entry_price=self.decision.entry_price,
            stop_price=self.decision.stop_price, take_profit_price=self.decision.take_profit_price,
            estimated_risk_usdt=self.decision.estimated_risk_usdt))

    def plan(self):
        return self.engine.plan_buy(decision_id=self.record.id, reconciliation=self.report,
            instrument=LIMITS, costs=COSTS, entry_price=D("80000"),
            max_size_btc_verified=True, slippage_verified=True)

    def validate(self, intent):
        return self.engine.validate_intent(intent.client_order_id, reconciliation=self.report,
            instrument=LIMITS, costs=COSTS, max_size_btc_verified=True,
            slippage_verified=True)

    def reserve(self):
        decision = self.manager.assess_buy(account=AccountSnapshot(
            NOW, NOW.date(), D("5000"), D("5000"), D("5000"), D(0), D(0), 0, 0, 0),
            instrument=LIMITS, costs=COSTS, entry_price=D("80000"), now_utc=NOW)
        self.assertEqual(decision.status.value, "ALLOW")
        return decision

    def test_plan_is_durable_unique_and_does_not_change_risk_or_budget(self):
        risk_before = self.manager.state_path.read_bytes()
        budget_before = self.store.read()
        intent = self.plan()
        self.assertRegex(intent.client_order_id, r"^TB[a-f0-9]{30}$")
        self.assertEqual(intent.state, ExecutionState.PLANNED)
        self.assertEqual(intent.stop_price, D("79200.0"))
        self.assertEqual(intent.take_profit_price, D("81600.0"))
        self.assertEqual(self.manager.state_path.read_bytes(), risk_before)
        self.assertEqual(self.store.read(), budget_before)
        restarted = ExecutionEngine(journal=self.journal, risk_manager=self.manager,
                                    budget_store=self.store, clock=lambda: NOW)
        self.assertTrue(restarted.diagnose().blocked)
        self.assertEqual(restarted.diagnose().intents[0].client_order_id, intent.client_order_id)
        with self.assertRaises(ExecutionError):
            restarted.plan_buy(decision_id=self.record.id, reconciliation=self.report,
                instrument=LIMITS, costs=COSTS, entry_price=D("80000"),
                max_size_btc_verified=True, slippage_verified=True)

    def test_validation_and_submitting_require_separate_risk_reservation(self):
        intent = self.plan()
        with self.assertRaises(ExecutionError):
            self.engine.mark_submitting(intent.client_order_id, risk_confirmation=self.decision)
        validated = self.validate(intent)
        self.assertEqual(validated.state, ExecutionState.VALIDATED)
        with self.assertRaises(ExecutionError):
            self.engine.mark_submitting(intent.client_order_id, risk_confirmation=self.decision)
        self.assertFalse(self.manager.inspect_daily_state(NOW).pending_entry)

    def test_stale_plan_does_not_advance_to_validation(self):
        intent = self.plan()
        self.engine.clock = lambda: NOW + timedelta(minutes=2)
        with self.assertRaises(ExecutionError):
            self.validate(intent)
        self.assertEqual(self.engine.diagnose().intents[0].state, ExecutionState.PLANNED)

    def test_wrong_risk_confirmation_cannot_cross_send_boundary(self):
        intent = self.validate(self.plan())
        reservation = self.reserve()
        with self.assertRaises(ExecutionError):
            self.engine.mark_submitting(intent.client_order_id,
                risk_confirmation=replace(reservation, quantity=reservation.quantity + D("0.0001")))
        self.assertEqual(self.engine.diagnose().intents[0].state, ExecutionState.VALIDATED)
        self.assertTrue(self.manager.inspect_daily_state(NOW).pending_entry)

    def test_timeout_and_ambiguous_response_remain_blocked_after_restart(self):
        intent = self.validate(self.plan())
        # This is a local transition test. No transport is attached or invoked.
        submitting = self.engine.mark_submitting(intent.client_order_id,
                                                  risk_confirmation=self.reserve())
        self.assertEqual(submitting.state, ExecutionState.SUBMITTING)
        self.assertTrue(self.engine.diagnose().blocked)
        uncertain = self.engine.record_uncertain(intent.client_order_id, timeout=True)
        self.assertEqual(uncertain.state, ExecutionState.UNKNOWN)
        self.assertEqual(uncertain.reason_code, "NETWORK_TIMEOUT")
        with self.assertRaises(ExecutionError):
            self.engine.record_ack(intent.client_order_id, "123")
        with self.assertRaises(ExecutionError):
            self.plan()
        self.assertTrue(ExecutionEngine(journal=self.journal).diagnose().blocked)
        with self.assertRaises(ExecutionError):
            self.engine.record_rejection(intent.client_order_id)
        with self.assertRaises(ExecutionError):
            self.engine.record_rejection(intent.client_order_id, proof=RecoveryProof(
                "other", NOW, True, True, True, "OKX_DEMO_READ_ONLY"))
        cleared = self.engine.record_rejection(intent.client_order_id, proof=RecoveryProof(
            intent.client_order_id, NOW, True, True, True, "OKX_DEMO_READ_ONLY"))
        self.assertEqual(cleared.state, ExecutionState.REJECTED)
        self.assertFalse(self.engine.diagnose().blocked)

    def test_same_candle_cannot_create_second_intent_after_local_rejection(self):
        intent = self.plan()
        self.engine.record_rejection(intent.client_order_id)
        with self.assertRaises(ExecutionError):
            self.plan()

    def test_btc_fee_creates_unprotectable_dust_and_halts(self):
        intent = self.validate(self.plan())
        self.engine.mark_submitting(intent.client_order_id, risk_confirmation=self.reserve())
        self.engine.record_ack(intent.client_order_id, "ord1")
        first = self.engine.record_fill(intent.client_order_id, trade_id="fill1",
            quantity=D("0.01"), price=D("80000"), fee=D("-0.00001"),
            fee_ccy="BTC", exchange_order_id="ord1")
        self.assertEqual(first.state, ExecutionState.HALTED)
        self.assertEqual(first.owned_btc, D("0.00999"))
        # The BTC fee creates step-incompatible quantity, so protection is unsafe.
        self.assertTrue(self.engine.diagnose().halted)

    def test_step_aligned_partial_and_final_fill(self):
        intent = self.validate(self.plan())
        self.engine.mark_submitting(intent.client_order_id, risk_confirmation=self.reserve())
        self.engine.record_ack(intent.client_order_id, "ord2")
        first = self.engine.record_fill(intent.client_order_id, trade_id="fill1",
            quantity=D("0.01"), price=D("80000"), fee=D("-0.8"),
            fee_ccy="USDT", exchange_order_id="ord2")
        self.assertEqual(first.state, ExecutionState.PARTIALLY_FILLED)
        self.assertEqual(first.fee_usdt, D("-0.8"))
        duplicate = self.engine.record_fill(intent.client_order_id, trade_id="fill1",
            quantity=D("0.01"), price=D("80000"), fee=D("-0.8"),
            fee_ccy="USDT", exchange_order_id="ord2")
        self.assertEqual(duplicate.filled_quantity, first.filled_quantity)
        remaining = intent.quantity - first.filled_quantity
        # Final remainder may not have step precision; the original intent is step-aligned.
        final = self.engine.record_fill(intent.client_order_id, trade_id="fill2",
            quantity=remaining, price=D("80100"), fee=D("-1"),
            fee_ccy="USDT", exchange_order_id="ord2")
        self.assertEqual(final.state, ExecutionState.FILLED)
        self.assertEqual(final.filled_quantity, intent.quantity)
        self.assertGreater(final.average_fill_price, D("80000"))
        self.assertTrue(self.engine.diagnose().blocked)

    def test_subminimum_partial_fill_halts_and_survives_restart(self):
        intent = self.validate(self.plan())
        self.engine.mark_submitting(intent.client_order_id, risk_confirmation=self.reserve())
        self.engine.record_ack(intent.client_order_id, "ord3")
        halted = self.engine.record_fill(intent.client_order_id, trade_id="dust",
            quantity=D("0.00001"), price=D("80000"), fee=D("0"),
            fee_ccy="USDT", exchange_order_id="ord3")
        self.assertEqual(halted.state, ExecutionState.HALTED)
        self.assertEqual(halted.filled_quantity, D("0.00001"))
        later = self.engine.record_fill(intent.client_order_id, trade_id="later",
            quantity=D("0.00999"), price=D("80000"), fee=D("0"),
            fee_ccy="USDT", exchange_order_id="ord3")
        self.assertEqual(later.state, ExecutionState.HALTED)
        self.assertEqual(later.filled_quantity, D("0.01"))
        self.assertTrue(ExecutionEngine(journal=self.journal).diagnose().halted)

    def test_halt_requires_manual_reconciliation_and_cannot_erase_fills(self):
        intent = self.plan()
        self.engine.emergency_halt()
        self.assertTrue(self.engine.diagnose().halted)
        proof = RecoveryProof(intent.client_order_id, NOW, True, True, True, "OKX_DEMO_READ_ONLY")
        with self.assertRaises(ExecutionError):
            self.engine.manual_recover(phrase="yes", proof=proof)
        self.engine.manual_recover(phrase=RECOVERY_PHRASE, proof=proof)
        self.assertFalse(self.engine.diagnose().halted)
        self.assertEqual(self.engine.diagnose().intents[0].state, ExecutionState.REJECTED)

    def test_deleted_halt_control_cannot_be_recreated_as_clear(self):
        self.plan()
        self.engine.emergency_halt()
        with sqlite3.connect(self.journal.path) as db:
            db.execute("DELETE FROM execution_control")
        self.assertTrue(self.engine.diagnose().blocked)
        with self.assertRaises(ExecutionError):
            self.plan()

    def test_manual_recovery_rejects_stale_or_mismatched_proof(self):
        intent = self.plan()
        self.engine.emergency_halt()
        for proof in (RecoveryProof("wrong", NOW, True, True, True, "OKX_DEMO_READ_ONLY"),
                      RecoveryProof(intent.client_order_id, NOW - timedelta(minutes=2),
                                    True, True, True, "OKX_DEMO_READ_ONLY"),
                      RecoveryProof(intent.client_order_id, NOW, True, False, True, "OKX_DEMO_READ_ONLY")):
            with self.subTest(proof=proof):
                with self.assertRaises(ExecutionError):
                    self.engine.manual_recover(phrase=RECOVERY_PHRASE, proof=proof)
        self.assertTrue(self.engine.diagnose().halted)

    def test_unknown_evidence_or_external_sell_cannot_be_planned(self):
        for report in (replace(self.report, status=ReconcileStatus.BLOCKED),
                       replace(self.report, checked_at_utc=NOW - timedelta(minutes=2)),
                       replace(self.report, open_order_count=1),
                       replace(self.report, bot_btc=None),
                       replace(self.report, daily_loss_usdt="0"),
                       replace(self.report, taker_fee_rate=None),
                       replace(self.report, usdt_available=D("4999"))):
            with self.subTest(report=report):
                with self.assertRaises(ExecutionError):
                    self.engine.plan_buy(decision_id=self.record.id, reconciliation=report,
                        instrument=LIMITS, costs=COSTS, entry_price=D("80000"),
                        max_size_btc_verified=True, slippage_verified=True)
        self.assertFalse(hasattr(self.engine, "plan_sell"))
        self.assertFalse(hasattr(self.engine, "submit_order"))

    def test_underestimated_fee_or_unverified_size_blocks(self):
        with self.assertRaises(ExecutionError):
            self.engine.plan_buy(decision_id=self.record.id, reconciliation=self.report,
                instrument=LIMITS, costs=replace(COSTS, entry_fee_rate=D("0.0005")),
                entry_price=D("80000"), max_size_btc_verified=True, slippage_verified=True)
        with self.assertRaises(ExecutionError):
            self.engine.plan_buy(decision_id=self.record.id, reconciliation=self.report,
                instrument=LIMITS, costs=COSTS, entry_price=D("80000"),
                max_size_btc_verified=False, slippage_verified=True)
        self.assertEqual(self.engine.diagnose().intents, ())

    def test_conflicting_fill_id_halts_without_duplicate(self):
        intent = self.validate(self.plan())
        self.engine.mark_submitting(intent.client_order_id, risk_confirmation=self.reserve())
        self.engine.record_ack(intent.client_order_id, "ord4")
        self.engine.record_fill(intent.client_order_id, trade_id="fill1", quantity=D("0.01"),
            price=D("80000"), fee=D("0"), fee_ccy="USDT", exchange_order_id="ord4")
        self.engine.record_fill(intent.client_order_id, trade_id="fill1", quantity=D("0.02"),
            price=D("80000"), fee=D("0"), fee_ccy="USDT", exchange_order_id="ord4")
        self.assertTrue(self.engine.diagnose().halted)
        with sqlite3.connect(self.journal.path) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM execution_fills").fetchone()[0], 1)

    def test_cli_offline_and_no_secrets_in_ledger(self):
        with patch("bot.demo_client.DemoClient.from_env_file", side_effect=AssertionError("network")):
            with patch("main.ExecutionEngine", return_value=self.engine):
                with redirect_stdout(StringIO()) as output:
                    self.assertEqual(main.main(["--execution-check"]), 0)
            self.assertIn("CLEAR", output.getvalue())
        self.plan()
        with self.journal.path.open("rb") as file:
            content = file.read()
        for secret in (b"fake-secret", b"OK-ACCESS-KEY", b"x-simulated-trading", b"passphrase"):
            self.assertNotIn(secret, content)


if __name__ == "__main__":
    unittest.main()
