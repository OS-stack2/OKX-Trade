"""Offline emergency recovery, with fake Demo GET and Telegram transports."""

from __future__ import annotations

from contextlib import closing, redirect_stdout
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
from io import StringIO
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import main
from bot.demo_client import CurrencyBalance, DemoAlgoOrder, DemoClient, DemoFill, DemoOrder
from bot.execution_engine import ExecutionEngine, ExecutionState
from bot.journal import CycleDraft, CycleStatus, DecisionJournal
from bot.protection_model import AlgoLeg, BuyObservation, BuyState, ProtectionJournal
from bot.recovery_manager import RecoveryManager
from bot.risk_manager import AccountSnapshot, ExecutionCosts, InstrumentLimits, RiskManager
from bot.strategy_budget import BudgetBaseline, StrategyBudgetStore
from bot.telegram_alerts import TelegramAlerter
from bot.reconcile import ReconcileReport, ReconcileStatus
from bot.market_data import SpotInstrument
from config import Config


NOW = datetime(2026, 10, 15, 12, 30, tzinfo=timezone.utc)
LIMITS = InstrumentLimits("BTC-USDT", "live", D("0.0001"), D("0.0001"), D("0.1"), D("0.1"))
SPOT = SpotInstrument("BTC-USDT", D("0.1"), D("0.0001"), D("0.0001"), "live", "BTC", "USDT")
COSTS = ExecutionCosts(D("0.001"), D("0.001"), D("0.001"), D("0.001"))


class FakeDemo(DemoClient):
    def __init__(self, intent):
        self._config = Config(mode="DEMO")
        self.intent = intent
        self.order_state = "partially_filled"
        self.fills = ()
        self.btc = CurrencyBalance(D("1"), D("1"))
        self.usdt = CurrencyBalance(D("5000"), D("5000"))
        self.pending = ()
        self.algos = {}
        self.foreign_order = False
        self.fail = False
        self.calls = []

    def get_balances(self):
        self.calls.append("GET balances")
        if self.fail:
            raise TimeoutError("secret-bearing transport error must be hidden")
        return self.btc, self.usdt

    def get_open_orders(self):
        self.calls.append("GET open")
        orders = [{"ordId": "order-1"}] if self.order_state in {"live", "partially_filled"} else []
        if self.foreign_order:
            orders.append({"ordId": "foreign"})
        return orders

    def get_pending_algo_orders(self):
        self.calls.append("GET pending algo")
        return self.pending

    def get_order_by_client_id(self, client_order_id):
        self.calls.append("GET order")
        return DemoOrder(client_order_id, "order-1", self.order_state,
                         self.intent.quantity, sum((f.size_btc for f in self.fills), D(0)),
                         self.intent.entry_price, NOW)

    def get_order_fills(self, order_id):
        self.calls.append("GET fills")
        return self.fills

    def get_fills_history(self):
        self.calls.append("GET history")
        return [{"instId": "BTC-USDT", "ts": str(int(NOW.timestamp() * 1000)),
                 "tradeId": f.trade_id} for f in self.fills]

    def get_algo_order_by_id(self, algo_id):
        self.calls.append("GET algo ID")
        return self.algos[algo_id]


class RecordingAlerts:
    def __init__(self, fail=False):
        self.events = []
        self.fail = fail

    def notify(self, event):
        self.events.append(event)
        if self.fail:
            raise RuntimeError("telegram secret must be hidden")
        return True


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.journal = DecisionJournal(self.root / "journal.sqlite3")
        self.risk = RiskManager(self.root / "risk.json")
        self.store = StrategyBudgetStore(self.journal.path)
        self.risk.initialize_day(AccountSnapshot(NOW, NOW.date(), D("5000"), D("5000"),
            D("5000"), D(0), D(0), 0, 0, 0), NOW)
        self.journal.append(CycleDraft(NOW - timedelta(seconds=20), NOW - timedelta(minutes=30),
                                       "HOLD", CycleStatus.HOLD, "earlier"))
        self.store.confirm_once(BudgetBaseline(NOW - timedelta(minutes=2), D("5000"),
            D("1"), D("1"), D("5000"), D("5000"), D("80000"), D("85000"),
            D("-0.0008"), D("-0.001"), 1))
        report = ReconcileReport(ReconcileStatus.READY, (), checked_at_utc=NOW,
            btc_total=D("1"), btc_available=D("1"), usdt_total=D("5000"),
            usdt_available=D("5000"), external_btc=D("1"), bot_btc=D(0),
            open_order_count=0, fill_count=0, instrument=SPOT,
            confirmed_realized_pnl_usdt=D(0), confirmed_unrealized_pnl_usdt=D(0),
            daily_loss_usdt=D(0), daily_loss_limit_usdt=D("100"),
            daily_risk_remaining_usdt=D("100"), strategy_capital_usdt=D("5000"),
            strategy_equity_usdt=D("5000"), per_trade_risk_limit_usdt=D("25"),
            taker_fee_rate=D("-0.001"))
        self.engine = ExecutionEngine(journal=self.journal, risk_manager=self.risk,
                                      budget_store=self.store, clock=lambda: NOW)
        decision = self.risk.preview_buy(account=AccountSnapshot(NOW, NOW.date(), D("5000"),
            D("5000"), D("5000"), D(0), D(0), 0, 0, 0), instrument=LIMITS,
            costs=COSTS, entry_price=D("80000"), now_utc=NOW)
        original = self.journal.append(CycleDraft(NOW, NOW - timedelta(minutes=15), "BUY",
            CycleStatus.ALLOW, "test BUY", quantity=decision.quantity,
            entry_price=decision.entry_price, stop_price=decision.stop_price,
            take_profit_price=decision.take_profit_price,
            estimated_risk_usdt=decision.estimated_risk_usdt))
        self.intent = self.engine.plan_buy(decision_id=original.id, reconciliation=report,
            instrument=LIMITS, costs=COSTS, entry_price=D("80000"),
            max_size_btc_verified=True, slippage_verified=True)
        self.demo = FakeDemo(self.intent)
        self.alerts = RecordingAlerts()
        self.recovery = RecoveryManager(engine=self.engine, client_loader=lambda: self.demo,
                                        alerter_loader=lambda: self.alerts, budget_store=self.store)

    def _partial(self):
        self.demo.fills = (DemoFill("trade-1", "order-1", D("0.01"), D("80000"),
                                    D("-0.0001"), "BTC", NOW),)
        self.demo.btc = CurrencyBalance(D("1.0099"), D("1.0099"))
        self.demo.usdt = CurrencyBalance(D("4200"), D("4200"))

    def _save_protection(self, *, stop_id="sl-1", take_id="tp-1"):
        self._partial()
        self.recovery.check_once()
        current = self.engine.diagnose().intents[0]
        def leg(algo_id, kind, price):
            return None if algo_id is None else AlgoLeg(
                algo_id, kind, "order-1", current.client_order_id, "oco-1",
                current.owned_btc, price, "live")
        ProtectionJournal(self.engine).record(BuyObservation(
            current.client_order_id, "order-1", BuyState.PARTIAL_ACTIVE,
            current.filled_quantity, NOW,
            leg(stop_id, "STOP", current.stop_price),
            leg(take_id, "TAKE", current.take_profit_price), True))

    def test_timeout_halts_before_get_and_never_guesses_fill(self):
        self.demo.fail = True
        report = self.recovery.check_once()
        self.assertTrue(report.halted)
        self.assertIsNone(report.bot_btc)
        self.assertIn("CONNECTION_LOST", self.alerts.events)
        self.assertEqual(self.engine.halt_reason_code(), "RECOVERY_NETWORK_ERROR")
        self.assertEqual(self.engine.diagnose().intents[0].filled_quantity, 0)
        self.assertNotIn("secret-bearing", " ".join(report.reasons))

    def test_network_loss_with_local_btc_alerts_unprotected_without_guessing_exchange(self):
        self._partial()
        self.recovery.check_once()
        self.demo.fail = True
        self.alerts.events.clear()
        report = self.recovery.check_once()
        self.assertIsNone(report.bot_btc)
        self.assertIn("SQLite показывает BTC бота", " ".join(report.reasons))
        self.assertIn("UNPROTECTED", self.alerts.events)
        self.assertEqual(self.engine.halt_reason_code(), "RECOVERY_NETWORK_ERROR")

    def test_connection_restored_does_not_clear_halt(self):
        self.demo.fail = True
        self.recovery.check_once()
        self.demo.fail = False
        self.demo.order_state = "live"
        report = self.recovery.check_once()
        self.assertTrue(report.connection_restored)
        self.assertIn("CONNECTION_RESTORED", self.alerts.events)
        self.assertTrue(self.engine.diagnose().halted)
        self.assertEqual(self.engine.diagnose().intents[0].state, ExecutionState.HALTED)

    def test_partial_buy_fee_and_external_btc_remain_separate(self):
        self._partial()
        report = self.recovery.check_once()
        self.assertEqual(report.bot_btc, D("0.0099"))
        self.assertEqual(report.external_btc, D("1"))
        self.assertEqual(report.uncovered_btc, D("0.0099"))
        self.assertEqual(report.confirmed_sl_btc, 0)
        self.assertEqual(self.engine.halt_reason_code(), "RECOVERY_UNPROTECTED")
        self.assertIn("UNPROTECTED", self.alerts.events)
        self.assertTrue(all(call.startswith("GET") for call in self.demo.calls))

    def test_repeated_run_is_idempotent_and_preserves_budget_and_risk(self):
        self._partial()
        budget_before = self.store.read()
        risk_before = self.risk.state_path.read_bytes()
        self.recovery.check_once()
        self.recovery.check_once()
        with closing(sqlite3.connect(self.journal.path)) as db:
            count = db.execute("SELECT COUNT(*) FROM execution_fills").fetchone()[0]
        self.assertEqual(count, 1)
        self.assertEqual(self.store.read(), budget_before)
        self.assertEqual(self.risk.state_path.read_bytes(), risk_before)
        self.assertEqual(self.alerts.events.count("UNPROTECTED"), 1)

    def test_partial_canceled_remains_unprotected(self):
        self._partial()
        self.demo.order_state = "canceled"
        report = self.recovery.check_once()
        self.assertEqual(report.open_order_count, 0)
        self.assertTrue(report.halted)
        self.assertEqual(report.uncovered_btc, D("0.0099"))

    def test_missing_algo_id_and_rejected_sl_never_prove_protection(self):
        self._save_protection(stop_id=None)
        self.demo.pending = (DemoAlgoOrder("tp-1", "oco", "live", D("0.0099"), "sell"),)
        self.demo.algos["tp-1"] = self.demo.pending[0]
        report = self.recovery.check_once()
        self.assertTrue(report.halted)
        self.assertIn("Algo ID ещё не подтверждён", " ".join(report.reasons))
        self.assertEqual(report.confirmed_sl_btc, 0)

        self.demo.pending = (DemoAlgoOrder("sl-1", "conditional", "live", D("0.0099"), "sell"),
                             DemoAlgoOrder("tp-1", "oco", "live", D("0.0099"), "sell"))
        self.demo.algos["sl-1"] = DemoAlgoOrder("sl-1", "conditional", "canceled", D("0.0099"), "sell")
        self._save_protection(stop_id="sl-1")
        report = self.recovery.check_once()
        self.assertIn("Algo ID, статус", " ".join(report.reasons))
        self.assertEqual(report.confirmed_sl_btc, 0)

    def test_delayed_stop_algo_id_stays_halted_after_restart_and_repeat(self):
        self._save_protection(stop_id=None)
        self.demo.pending = (DemoAlgoOrder("tp-1", "oco", "live", D("0.0099"), "sell"),)
        self.demo.algos["tp-1"] = self.demo.pending[0]
        risk_before = self.risk.state_path.read_bytes()
        budget_before = self.store.read()
        restarted = RecoveryManager(
            engine=ExecutionEngine(journal=self.journal, risk_manager=self.risk,
                                   budget_store=self.store, clock=lambda: NOW),
            client_loader=lambda: self.demo, alerter_loader=lambda: self.alerts,
            budget_store=self.store,
        )
        first = restarted.check_once()
        second = restarted.check_once()
        for report in (first, second):
            self.assertTrue(report.halted)
            self.assertEqual(report.external_btc, D("1"))
            self.assertEqual(report.confirmed_sl_btc, 0)
            self.assertIn("Algo ID ещё не подтверждён", " ".join(report.reasons))
        self.assertEqual(self.risk.state_path.read_bytes(), risk_before)
        self.assertEqual(self.store.read(), budget_before)
        self.assertEqual(len(self.engine.diagnose().intents), 1)
        self.assertFalse(any("POST" in call or "SELL" in call for call in self.demo.calls))

    def test_exact_algo_get_still_cannot_prove_parent_or_prevent_double_sale(self):
        self._save_protection()
        sl = DemoAlgoOrder("sl-1", "conditional", "live", D("0.0099"), "sell")
        tp = DemoAlgoOrder("tp-1", "oco", "live", D("0.0099"), "sell")
        self.demo.pending = (sl, tp)
        self.demo.algos = {"sl-1": sl, "tp-1": tp}
        report = self.recovery.check_once()
        self.assertTrue(report.halted)
        self.assertEqual(report.confirmed_sl_btc, 0)
        self.assertIn("исключение двойной продажи", " ".join(report.reasons))

    def test_duplicate_algo_id_blocks(self):
        self._save_protection(stop_id="same", take_id="same")
        self.demo.pending = (DemoAlgoOrder("same", "conditional", "live", D("0.0099"), "sell"),)
        self.demo.algos["same"] = self.demo.pending[0]
        report = self.recovery.check_once()
        self.assertIn("повторно зарезервирован", " ".join(report.reasons))

    def test_unmatched_recent_fill_blocks_unknown_account_movement(self):
        self.demo.order_state = "live"
        self.demo.get_fills_history = lambda: [{"instId": "BTC-USDT", "ts": str(int(NOW.timestamp() * 1000)),
                                                 "tradeId": "outside"}]
        report = self.recovery.check_once()
        self.assertIn("неучтённое BTC-USDT", " ".join(report.reasons))

    def test_unknown_order_without_intent_persists_halt(self):
        with self.engine._transaction() as db:
            db.execute("DELETE FROM execution_events")
            db.execute("DELETE FROM execution_intents")
        self.demo.foreign_order = True
        report = self.recovery.check_once()
        self.assertTrue(report.halted)
        self.assertEqual(self.engine.halt_reason_code(), "RECOVERY_MISMATCH")
        self.assertIn("MISMATCH", self.alerts.events)

    def test_missing_sqlite_reports_block_without_claiming_persisted_halt(self):
        absent = ExecutionEngine(journal=DecisionJournal(self.root / "absent.sqlite3"),
                                 clock=lambda: NOW)
        report = RecoveryManager(engine=absent, client_loader=lambda: self.demo,
                                 alerter_loader=lambda: self.alerts).check_once()
        self.assertEqual(report.status, "BLOCKED")
        self.assertFalse(report.halted)
        self.assertFalse(absent.path.exists())

    def test_conflicting_balance_or_foreign_order_blocks(self):
        self._partial()
        self.demo.btc = CurrencyBalance(D("1.01"), D("1.01"))
        report = self.recovery.check_once()
        self.assertIn("Движение BTC/USDT", " ".join(report.reasons))
        self.demo.btc = CurrencyBalance(D("1.0099"), D("1.0099"))
        self.demo.foreign_order = True
        report = self.recovery.check_once()
        self.assertIn("чужой", " ".join(report.reasons))

    def test_telegram_exception_does_not_prevent_durable_halt(self):
        self.alerts.fail = True
        self.demo.fail = True
        report = self.recovery.check_once()
        self.assertEqual(report.notification, "доставка не подтверждена")
        self.assertEqual(self.engine.halt_reason_code(), "RECOVERY_NETWORK_ERROR")

    def test_restart_reads_persisted_halt_and_never_sells_external_btc(self):
        self._partial()
        self.recovery.check_once()
        restarted = ExecutionEngine(journal=self.journal, risk_manager=self.risk,
                                    budget_store=self.store, clock=lambda: NOW)
        manager = RecoveryManager(engine=restarted, client_loader=lambda: self.demo,
                                  alerter_loader=lambda: self.alerts, budget_store=self.store)
        report = manager.check_once()
        self.assertTrue(report.halted)
        self.assertEqual(report.external_btc, D("1"))
        self.assertEqual(report.bot_btc, D("0.0099"))
        self.assertFalse(any("POST" in call or "SELL" in call for call in self.demo.calls))

    def test_cli_missing_configuration_never_exposes_secret(self):
        with patch("main.RecoveryManager") as manager:
            manager.return_value.check_once.return_value = SimpleNamespace(
                status="BLOCKED", halted=True, bot_btc=None, external_btc=None,
                confirmed_sl_btc=None, uncovered_btc=None, open_order_count=None,
                reasons=("Demo GET недоступен",), owner_action="Проверить Demo", notification="не отправлено")
            output = StringIO()
            with redirect_stdout(output):
                code = main.main(["--recovery-check"])
        self.assertEqual(code, 1)
        self.assertIn("BLOCKED", output.getvalue())


class TelegramTests(unittest.TestCase):
    def test_local_env_only_and_no_secret_repr_or_response(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / ".env"
            token = "123456:" + "abcDEF_1234567890abcDEF_123"
            path.write_text("OKX_MODE=DEMO\nOKX_TELEGRAM_BOT_TOKEN=" + token +
                            "\nOKX_TELEGRAM_CHAT_ID=1234\n", encoding="utf-8")
            alerter = TelegramAlerter.from_env_file(path)
            self.assertIsNotNone(alerter)
            self.assertNotIn(token, repr(alerter))
            seen = []
            def fake(request, timeout):
                seen.append((request.full_url, request.get_method(), request.data))
                return {"ok": True}
            alerter = TelegramAlerter(alerter.token, alerter.chat_id, fake)
            self.assertTrue(alerter.notify("HALTED"))
            self.assertEqual(seen[0][1], "POST")
            self.assertIn(b"OKX Demo BTC-USDT", seen[0][2])
            self.assertNotIn(token.encode(), seen[0][2])
            path.write_text("OKX_MODE=LIVE\nOKX_TELEGRAM_BOT_TOKEN=" + token +
                            "\nOKX_TELEGRAM_CHAT_ID=1234\n", encoding="utf-8")
            self.assertIsNone(TelegramAlerter.from_env_file(path))
