"""Offline execution-preparation state machine; never sends an exchange request."""

from __future__ import annotations

from dataclasses import replace
from contextlib import closing, redirect_stdout
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import sqlite3
import unittest
from unittest.mock import patch

import main
from bot.execution_engine import (ExecutionEngine, ExecutionError, ExecutionState,
                                  RECOVERY_PHRASE, RecoveryProof)
from bot.demo_client import DemoError, DemoFill, DemoOrder
from bot.journal import CycleDraft, CycleStatus, DecisionJournal
from bot.market_data import MarketSnapshot, SpotInstrument
from bot.order_observer import DemoOrderObserver
from bot.order_preview import PROTECTION_BLOCK, preview_demo_buy
from bot.reconcile import ReconcileReport, ReconcileStatus
from bot.risk_manager import AccountSnapshot, ExecutionCosts, InstrumentLimits, RiskManager
from bot.strategy import Signal, StrategyResult
from bot.strategy_budget import BudgetBaseline, StrategyBudgetStore
from config import Config


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
        # Connection.__exit__ handles the transaction but does not close the file.
        with closing(sqlite3.connect(self.journal.path)) as db, db:
            with closing(db.execute("DELETE FROM execution_control")):
                pass
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
        with closing(sqlite3.connect(self.journal.path)) as db:
            with closing(db.execute("SELECT COUNT(*) FROM execution_fills")) as cursor:
                self.assertEqual(cursor.fetchone()[0], 1)

    def test_sqlite_resources_close_on_success_and_exception(self):
        class TrackedConnection(sqlite3.Connection):
            closed = False

            def close(self):
                self.closed = True
                super().close()

        path = self.root / "resource_check.sqlite3"
        with closing(sqlite3.connect(path, factory=TrackedConnection)) as db, db:
            with closing(db.execute("CREATE TABLE sample (value INTEGER)")) as cursor:
                pass
        self.assertTrue(db.closed)
        with self.assertRaises(sqlite3.ProgrammingError):
            db.execute("SELECT 1")
        with self.assertRaises(sqlite3.ProgrammingError):
            cursor.fetchone()

        with self.assertRaisesRegex(RuntimeError, "stop before cleanup"):
            with closing(sqlite3.connect(path, factory=TrackedConnection)) as failed_db, failed_db:
                with closing(failed_db.execute("INSERT INTO sample VALUES (1)")) as failed_cursor:
                    raise RuntimeError("stop before cleanup")
        self.assertTrue(failed_db.closed)
        with self.assertRaises(sqlite3.ProgrammingError):
            failed_db.execute("SELECT 1")
        with self.assertRaises(sqlite3.ProgrammingError):
            failed_cursor.fetchone()
        with closing(sqlite3.connect(path)) as check_db:
            with closing(check_db.execute("SELECT COUNT(*) FROM sample")) as cursor:
                self.assertEqual(cursor.fetchone()[0], 0)

    def test_engine_transaction_closes_connection_and_rolls_back_on_error(self):
        class TrackedConnection(sqlite3.Connection):
            closed = False

            def close(self):
                self.closed = True
                super().close()

        opened = []
        real_connect = sqlite3.connect

        def tracked_connect(*args, **kwargs):
            connection = real_connect(*args, factory=TrackedConnection, **kwargs)
            opened.append(connection)
            return connection

        with patch("bot.execution_engine.sqlite3.connect", side_effect=tracked_connect):
            with self.engine._transaction() as db:
                with closing(db.execute("CREATE TABLE resource_check (value INTEGER)")):
                    pass
            with self.assertRaisesRegex(RuntimeError, "abort transaction"):
                with self.engine._transaction() as db:
                    with closing(db.execute("INSERT INTO resource_check VALUES (1)")):
                        raise RuntimeError("abort transaction")

        self.assertEqual(len(opened), 2)
        self.assertTrue(all(connection.closed for connection in opened))
        with closing(sqlite3.connect(self.journal.path)) as db:
            with closing(db.execute("SELECT COUNT(*) FROM resource_check")) as cursor:
                self.assertEqual(cursor.fetchone()[0], 0)

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

    def test_complete_buy_candidate_still_blocks_unproven_atomic_protection(self):
        market = MarketSnapshot(SPOT, D("80000"), (), NOW)
        signal = StrategyResult(Signal.BUY, D("2"), D("1"), D("1"), D("2"),
                                NOW - timedelta(minutes=15), "cross")
        before = self.manager.state_path.read_bytes()
        preview = preview_demo_buy(market=market, signal=signal,
            reconciliation=self.report, engine=self.engine,
            verified_max_size_btc=D("0.1"), verified_costs=COSTS, now_utc=NOW)
        self.assertEqual(preview.status, "BLOCK")
        self.assertIn(PROTECTION_BLOCK, preview.reasons)
        self.assertEqual(preview.quantity_btc, self.decision.quantity)
        self.assertEqual((preview.estimated_stop_usdt, preview.estimated_take_usdt),
                         (D("79200.0"), D("81600.0")))
        self.assertEqual(self.manager.state_path.read_bytes(), before)
        self.assertEqual(self.engine.diagnose().intents, ())

    def test_preview_missing_evidence_and_sell_never_target_external_btc(self):
        market = MarketSnapshot(SPOT, D("80000"), (), NOW)
        sell = StrategyResult(Signal.SELL, D("1"), D("2"), D("2"), D("1"),
                              NOW - timedelta(minutes=15), "cross")
        preview = preview_demo_buy(market=market, signal=sell,
            reconciliation=self.report, engine=self.engine, now_utc=NOW)
        self.assertEqual(preview.status, "BLOCK")
        self.assertIsNone(preview.quantity_btc)
        self.assertEqual(preview.external_btc, D("1"))
        self.assertTrue(any("BTC" in reason for reason in preview.reasons))

    def test_preview_rejects_minimum_size_and_invalid_price(self):
        signal = StrategyResult(Signal.BUY, D("2"), D("1"), D("1"), D("2"),
                                NOW - timedelta(minutes=15), "cross")
        too_large_minimum = MarketSnapshot(replace(SPOT, min_size=D("0.2")),
                                           D("80000"), (), NOW)
        denied = preview_demo_buy(market=too_large_minimum, signal=signal,
            reconciliation=self.report, engine=self.engine,
            verified_max_size_btc=D("0.1"), verified_costs=COSTS, now_utc=NOW)
        self.assertEqual(denied.status, "BLOCK")
        self.assertIsNone(denied.quantity_btc)
        invalid_price = preview_demo_buy(market=MarketSnapshot(SPOT, D("NaN"), (), NOW),
            signal=signal, reconciliation=self.report, engine=self.engine, now_utc=NOW)
        self.assertIsNone(invalid_price.estimated_entry_usdt)
        self.assertTrue(any("Цена" in reason for reason in invalid_price.reasons))

    def _submitting_intent(self):
        intent = self.validate(self.plan())
        self.engine.mark_submitting(intent.client_order_id, risk_confirmation=self.reserve())
        return intent

    def test_read_only_order_observation_is_idempotent_and_halts_partial_fill(self):
        intent = self._submitting_intent()
        class Reader:
            def get_order_by_client_id(self, client_id):
                return DemoOrder(client_id, "order1", "partially_filled", intent.quantity,
                                 D("0.01"), intent.entry_price, NOW)
            def get_order_fills(self, order_id):
                return (DemoFill("trade1", order_id, D("0.01"), D("80000"),
                                 D("-0.8"), "USDT", NOW),)
        observer = DemoOrderObserver(client=Reader(), engine=self.engine, clock=lambda: NOW)
        first = observer.observe(intent.client_order_id)
        second = observer.observe(intent.client_order_id)
        self.assertTrue(first.halted)
        self.assertTrue(second.halted)
        self.assertEqual(second.intents[0].filled_quantity, D("0.01"))
        with self.assertRaises(ExecutionError):
            self.engine.manual_recover(phrase=RECOVERY_PHRASE, proof=RecoveryProof(
                intent.client_order_id, NOW, True, True, True, "OKX_DEMO_READ_ONLY"))
        with closing(sqlite3.connect(self.journal.path)) as db:
            with closing(db.execute("SELECT COUNT(*) FROM execution_fills")) as cursor:
                self.assertEqual(cursor.fetchone()[0], 1)

    def test_read_only_order_timeout_halts_and_survives_restart(self):
        intent = self._submitting_intent()
        class Reader:
            def get_order_by_client_id(self, client_id):
                raise DemoError("read-only timeout")
        result = DemoOrderObserver(client=Reader(), engine=self.engine,
                                   clock=lambda: NOW).observe(intent.client_order_id)
        self.assertTrue(result.halted)
        self.assertEqual(result.intents[0].state, ExecutionState.HALTED)
        self.assertTrue(ExecutionEngine(journal=self.journal).diagnose().halted)
        proof = RecoveryProof(intent.client_order_id, NOW, True, True, True, "OKX_DEMO_READ_ONLY")
        with self.assertRaises(ExecutionError):
            self.engine.manual_recover(phrase="incorrect", proof=proof)
        self.engine.manual_recover(phrase=RECOVERY_PHRASE, proof=proof)
        self.assertEqual(self.engine.diagnose().intents[0].state, ExecutionState.REJECTED)

    def test_unknown_order_lookup_is_not_resubmitted_and_halts(self):
        intent = self._submitting_intent()
        self.engine.record_uncertain(intent.client_order_id, timeout=True)
        class Reader:
            def get_order_by_client_id(self, client_id):
                return DemoOrder(client_id, "order2", "live", intent.quantity,
                                 D(0), intent.entry_price, NOW)
            def get_order_fills(self, order_id):
                return ()
        result = DemoOrderObserver(client=Reader(), engine=self.engine,
                                   clock=lambda: NOW).observe(intent.client_order_id)
        self.assertTrue(result.halted)
        self.assertEqual(result.intents[0].exchange_order_id, "order2")

    def test_incomplete_partial_fill_history_halts_without_guessing(self):
        intent = self._submitting_intent()
        class Reader:
            def get_order_by_client_id(self, client_id):
                return DemoOrder(client_id, "order3", "partially_filled", intent.quantity,
                                 D("0.01"), intent.entry_price, NOW)
            def get_order_fills(self, order_id):
                return ()
        result = DemoOrderObserver(client=Reader(), engine=self.engine,
                                   clock=lambda: NOW).observe(intent.client_order_id)
        self.assertTrue(result.halted)
        self.assertEqual(result.intents[0].filled_quantity, D(0))

    def test_partially_filled_then_canceled_records_owned_btc_and_halts(self):
        intent = self._submitting_intent()
        class Reader:
            def get_order_by_client_id(self, client_id):
                return DemoOrder(client_id, "order-canceled", "canceled", intent.quantity,
                                 D("0.01"), intent.entry_price, NOW)
            def get_order_fills(self, order_id):
                return (DemoFill("trade-canceled", order_id, D("0.01"), D("80000"),
                                 D("-0.8"), "USDT", NOW),)
        result = DemoOrderObserver(client=Reader(), engine=self.engine,
                                   clock=lambda: NOW).observe(intent.client_order_id)
        self.assertTrue(result.halted)
        self.assertEqual(result.intents[0].owned_btc, D("0.01"))
        with closing(sqlite3.connect(self.journal.path)) as db:
            with closing(db.execute("SELECT reason_code FROM execution_control WHERE id=1")) as cursor:
                self.assertEqual(cursor.fetchone()[0], "PARTIAL_BUY_CANCELED")

    def test_claimed_full_fill_with_partial_volume_halts_without_recording(self):
        intent = self._submitting_intent()
        class Reader:
            def get_order_by_client_id(self, client_id):
                return DemoOrder(client_id, "order-contradictory", "filled", intent.quantity,
                                 D("0.01"), intent.entry_price, NOW)
            def get_order_fills(self, order_id):
                return (DemoFill("trade-contradictory", order_id, D("0.01"), D("80000"),
                                 D("-0.8"), "USDT", NOW),)
        result = DemoOrderObserver(client=Reader(), engine=self.engine,
                                   clock=lambda: NOW).observe(intent.client_order_id)
        self.assertTrue(result.halted)
        self.assertEqual(result.intents[0].filled_quantity, D(0))

    def test_order_preview_cli_is_read_only_and_shows_block_reasons(self):
        market = MarketSnapshot(SPOT, D("80000"), (), NOW)
        signal = StrategyResult(Signal.BUY, D("2"), D("1"), D("1"), D("2"),
                                NOW - timedelta(minutes=15), "cross")
        risk_before = self.manager.state_path.read_bytes()
        journal_before = self.journal.path.read_bytes()
        client = SimpleNamespace(config=Config(region="eea", mode="DEMO"))
        output = StringIO()
        with (patch.object(main.DemoClient, "from_env_file", return_value=client),
              patch.object(main, "get_market_snapshot", return_value=market),
              patch.object(main, "evaluate_signal", return_value=signal),
              patch.object(main, "DecisionJournal", return_value=self.journal),
              patch.object(main, "RiskManager", return_value=self.manager),
              patch.object(main, "StrategyBudgetStore", return_value=self.store),
              patch.object(main, "ExecutionEngine", return_value=self.engine),
              patch.object(main, "DemoReconciler") as reconciler,
              redirect_stdout(output)):
            reconciler.return_value.run_once.return_value = self.report
            self.assertEqual(main.main(["--order-preview"]), 1)
        self.assertIn("BLOCK", output.getvalue())
        self.assertIn("SL 1%", output.getvalue())
        self.assertEqual(self.manager.state_path.read_bytes(), risk_before)
        self.assertEqual(self.journal.path.read_bytes(), journal_before)


if __name__ == "__main__":
    unittest.main()
