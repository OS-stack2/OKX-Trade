"""Offline proof that Demo budget confirmation is explicit and fail-closed."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

import main
from bot.budget_setup import BudgetInitializer
from bot.demo_client import DemoClient, DemoCredentials
from bot.journal import CycleDraft, CycleStatus, DecisionJournal
from bot.market_data import MarketSnapshot, PublicQuoteEvidence, SpotInstrument
from bot.reconcile import DemoReconciler, ReconcileStatus
from bot.risk_manager import RiskManager, RiskSettings
from bot.strategy_budget import BudgetError, CONFIRMATION_PHRASE, StrategyBudgetStore
from config import Config


NOW = datetime(2026, 10, 15, 12, 30, tzinfo=timezone.utc)
INSTRUMENT = SpotInstrument("BTC-USDT", Decimal("0.1"), Decimal("0.00000001"),
                            Decimal("0.00001"), "live", "BTC", "USDT",
                            Decimal("99999999"), Decimal("100000"))


class BudgetTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.journal = DecisionJournal(self.root / "decisions.sqlite3")
        self.journal.append(CycleDraft(NOW, NOW - timedelta(minutes=15), "HOLD", CycleStatus.HOLD, "no cross"))
        self.manager = RiskManager(self.root / "risk.json")
        self.store = StrategyBudgetStore(self.journal.path)
        self.btc_total = "1"
        self.btc_available = "1"
        self.usdt_total = "5000"
        self.usdt_available = "5000"
        self.orders = []
        self.fills = []
        self.fee = "-0.001"
        self.price = Decimal("81047.8")
        self.instrument = INSTRUMENT
        self.requests = []
        self.client = DemoClient(
            Config(region="global", mode="DEMO"),
            DemoCredentials("dummy-key", "dummy-secret", "dummy-passphrase"),
            transport=self.transport, clock=lambda: NOW,
        )

    def transport(self, request, timeout):
        self.requests.append(request)
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(dict(request.header_items())["X-simulated-trading"], "1")
        self.assertEqual(urlsplit(request.full_url).netloc, "www.okx.com")
        rows = {
            "/api/v5/account/balance": [{"details": [
                {"ccy": "BTC", "cashBal": self.btc_total, "availBal": self.btc_available},
                {"ccy": "USDT", "cashBal": self.usdt_total, "availBal": self.usdt_available},
            ]}],
            "/api/v5/trade/orders-pending": self.orders,
            "/api/v5/trade/fills-history": self.fills,
            "/api/v5/account/trade-fee": [{"maker": "-0.0008", "taker": self.fee}],
        }[urlsplit(request.full_url).path]
        return {"code": "0", "data": rows}

    def market(self):
        return MarketSnapshot(self.instrument, self.price, (), NOW)

    def initializer(self):
        return BudgetInitializer(
            market_provider=self.market, demo_client=self.client, journal=self.journal,
            risk_manager=self.manager, store=self.store, clock=lambda: NOW,
        )

    def reconciler(self):
        return DemoReconciler(
            market_provider=self.market, demo_client=self.client, journal=self.journal,
            risk_manager=self.manager, budget_store=self.store, clock=lambda: NOW,
        )

    def test_preview_is_read_only_and_exact_confirmation_sets_strategy_only(self):
        journal_before = self.journal.path.read_bytes()
        preview = self.initializer().preview()
        self.assertTrue(preview.ready_to_confirm)
        self.assertEqual(len(preview.token), 16)
        self.assertIsNone(self.store.read())
        self.assertFalse(self.manager.state_path.exists())
        self.assertEqual(self.journal.path.read_bytes(), journal_before)
        with self.assertRaises(BudgetError):
            self.initializer().confirm("yes", preview.token)
        with self.assertRaises(BudgetError):
            self.initializer().confirm(CONFIRMATION_PHRASE, "wrong-token")
        baseline = self.initializer().confirm(CONFIRMATION_PHRASE, preview.token)
        self.assertEqual(baseline.strategy_capital_usdt, Decimal("5000"))
        self.assertEqual(baseline.demo_btc_total, Decimal("1"))
        self.assertEqual(baseline.demo_usdt_total, Decimal("5000"))
        self.assertEqual(baseline.portfolio_btc_usdt_value, Decimal("86047.8"))
        self.assertIn("Owner allocation", baseline.source_description)
        self.assertEqual(self.store.read(), baseline)
        state = self.manager.inspect_daily_state(NOW)
        self.assertEqual(state.start_capital_usdt, Decimal("5000"))
        self.assertEqual(state.entries_used, 0)
        report = self.reconciler().run_once()
        self.assertEqual(report.status, ReconcileStatus.READY)
        self.assertEqual(report.strategy_equity_usdt, Decimal("5000"))
        self.assertEqual(report.confirmed_realized_pnl_usdt, Decimal("0"))
        self.assertEqual(report.confirmed_unrealized_pnl_usdt, Decimal("0"))
        self.assertEqual(report.per_trade_risk_limit_usdt, Decimal("25.000"))
        self.assertEqual(report.daily_loss_limit_usdt, Decimal("100.00"))
        self.assertEqual(report.external_btc, Decimal("1"))
        self.assertTrue(all(request.get_method() == "GET" for request in self.requests))

    def test_external_btc_price_change_does_not_change_strategy_capital_or_pnl(self):
        token = self.initializer().preview().token
        self.initializer().confirm(CONFIRMATION_PHRASE, token)
        self.price = Decimal("90000")
        report = self.reconciler().run_once()
        self.assertEqual(report.status, ReconcileStatus.READY)
        self.assertEqual(report.portfolio_value_usdt, Decimal("95000"))
        self.assertEqual(report.strategy_equity_usdt, Decimal("5000"))
        self.assertEqual(report.daily_loss_usdt, Decimal("0"))

    def test_repeat_confirmation_cannot_reset_existing_loss_lock(self):
        token = self.initializer().preview().token
        self.initializer().confirm(CONFIRMATION_PHRASE, token)
        payload = json.loads(self.manager.state_path.read_text(encoding="utf-8"))
        payload["loss_locked"] = True
        self.manager.state_path.write_text(json.dumps(payload), encoding="utf-8")
        before = self.manager.state_path.read_bytes()
        with self.assertRaises(BudgetError):
            self.initializer().confirm(CONFIRMATION_PHRASE, token)
        self.assertEqual(self.manager.state_path.read_bytes(), before)
        self.assertEqual(self.reconciler().run_once().status, ReconcileStatus.BLOCKED)

    def test_unexplained_balance_movement_blocks_pnl_without_changing_baseline(self):
        token = self.initializer().preview().token
        baseline = self.initializer().confirm(CONFIRMATION_PHRASE, token)
        self.usdt_total = self.usdt_available = "4999"
        report = self.reconciler().run_once()
        self.assertEqual(report.status, ReconcileStatus.BLOCKED)
        self.assertIsNone(report.confirmed_realized_pnl_usdt)
        self.assertIsNone(report.strategy_equity_usdt)
        self.assertTrue(any("Баланс Demo изменился" in reason for reason in report.reasons))
        self.assertEqual(self.store.read(), baseline)

    def test_foreign_order_and_fill_block_initialization(self):
        self.orders = [{"instType": "SPOT", "instId": "BTC-USDT", "ordId": "foreign", "state": "live"}]
        self.assertFalse(self.initializer().preview().ready_to_confirm)
        self.orders = []
        self.fills = [{"instType": "SPOT", "instId": "BTC-USDT", "tradeId": "foreign",
                       "feeCcy": "USDT", "fee": "-0.01", "fillSz": "0.001", "fillPx": "80000",
                       "side": "buy", "ts": str(int(NOW.timestamp() * 1000))}]
        self.assertFalse(self.initializer().preview().ready_to_confirm)
        self.assertFalse(self.manager.state_path.exists())
        self.assertIsNone(self.store.read())

    def test_insufficient_or_reserved_usdt_and_reserved_btc_block(self):
        for total, available, btc_available in (
            ("4999", "4999", "1"), ("5000", "4999", "1"), ("5001", "5000", "1"),
            ("5000", "5000", "0.9"),
        ):
            with self.subTest(total=total, available=available, btc_available=btc_available):
                self.usdt_total, self.usdt_available, self.btc_available = total, available, btc_available
                self.assertFalse(self.initializer().preview().ready_to_confirm)
        self.assertIsNone(self.store.read())

    def test_bad_fee_limits_and_missing_journal_block(self):
        self.fee = "invalid"
        self.assertFalse(self.initializer().preview().ready_to_confirm)
        self.fee = "-0.001"
        self.instrument = SpotInstrument("BTC-USDT", Decimal("0.1"), Decimal("0.00000001"),
                                         Decimal("0.00001"), "live", "BTC", "USDT",
                                         Decimal("0.000001"), Decimal("100000"))
        self.assertFalse(self.initializer().preview().ready_to_confirm)
        self.instrument = INSTRUMENT
        self.journal.path.unlink()
        self.assertFalse(self.initializer().preview().ready_to_confirm)
        self.assertFalse(self.manager.state_path.exists())

    def test_preview_separates_stale_price_and_missing_limit_without_writes(self):
        before = self.journal.path.read_bytes()
        evidence = PublicQuoteEvidence(
            replace(INSTRUMENT, max_market_size=None), self.price,
            NOW - timedelta(seconds=61), (),
        )
        preview = BudgetInitializer(
            market_provider=lambda: evidence, demo_client=self.client, journal=self.journal,
            risk_manager=self.manager, store=self.store, clock=lambda: NOW,
        ).preview()
        self.assertFalse(preview.ready_to_confirm)
        self.assertEqual(dict(preview.report.diagnostic_checks)["QUOTE_TIME"], "BLOCKED")
        self.assertEqual(dict(preview.report.diagnostic_checks)["LIMITS"], "BLOCKED")
        self.assertTrue(any("котировка старше" in x for x in preview.reasons))
        self.assertTrue(any("maxMktSz" in x for x in preview.reasons))
        self.assertEqual(self.journal.path.read_bytes(), before)
        self.assertIsNone(self.store.read())
        self.assertFalse(self.manager.state_path.exists())

    def test_token_rechecks_changes_and_expires_at_utc_window(self):
        preview = self.initializer().preview()
        self.usdt_total = self.usdt_available = "5001"
        with self.assertRaises(BudgetError):
            self.initializer().confirm(CONFIRMATION_PHRASE, preview.token)
        self.assertIsNone(self.store.read())
        self.usdt_total = self.usdt_available = "5000"
        later = NOW + timedelta(minutes=5)
        expired = BudgetInitializer(
            market_provider=self.market, demo_client=self.client, journal=self.journal,
            risk_manager=self.manager, store=self.store, clock=lambda: later,
        )
        with self.assertRaises(BudgetError):
            expired.confirm(CONFIRMATION_PHRASE, preview.token)

    def test_journal_change_between_preview_and_confirmation_blocks(self):
        token = self.initializer().preview().token
        self.journal.append(CycleDraft(NOW, NOW - timedelta(minutes=30), "HOLD", CycleStatus.HOLD, "another"))
        with self.assertRaises(BudgetError):
            self.initializer().confirm(CONFIRMATION_PHRASE, token)
        self.assertIsNone(self.store.read())
        self.assertFalse(self.manager.state_path.exists())

    def test_partial_risk_state_failure_is_durable_and_blocks_repeat(self):
        token = self.initializer().preview().token
        with patch.object(self.manager, "initialize_day", side_effect=OSError("dummy-secret")):
            with self.assertRaises(BudgetError) as caught:
                self.initializer().confirm(CONFIRMATION_PHRASE, token)
        self.assertNotIn("dummy-secret", str(caught.exception))
        self.assertIsNotNone(self.store.read())
        self.assertFalse(self.initializer().preview().ready_to_confirm)
        self.assertEqual(self.reconciler().run_once().status, ReconcileStatus.BLOCKED)

    def test_unmatched_fill_after_confirmation_blocks_pnl(self):
        token = self.initializer().preview().token
        self.initializer().confirm(CONFIRMATION_PHRASE, token)
        self.fills = [{"instType": "SPOT", "instId": "BTC-USDT", "tradeId": "foreign",
                       "feeCcy": "USDT", "fee": "-0.01", "fillSz": "0.001", "fillPx": "80000",
                       "side": "buy", "ts": str(int(NOW.timestamp() * 1000))}]
        report = self.reconciler().run_once()
        self.assertEqual(report.status, ReconcileStatus.BLOCKED)
        self.assertIsNone(report.confirmed_realized_pnl_usdt)
        self.assertIsNone(report.daily_risk_remaining_usdt)

    def test_existing_risk_state_blocks_new_allocation(self):
        self.manager.state_path.write_text("corrupt", encoding="utf-8")
        self.assertFalse(self.initializer().preview().ready_to_confirm)
        self.assertIsNone(self.store.read())

    def test_changed_risk_settings_block_allocation(self):
        self.manager.settings = RiskSettings(risk_per_trade=Decimal("0.01"))
        preview = self.initializer().preview()
        self.assertFalse(preview.ready_to_confirm)
        self.assertTrue(any("Настройки Risk Manager" in reason for reason in preview.reasons))

    def test_cli_preview_and_confirm_require_exact_phrase_and_token(self):
        def factory(**kwargs):
            return BudgetInitializer(**kwargs, clock=lambda: NOW)
        output = StringIO()
        error = StringIO()
        with patch.object(main.DemoClient, "from_env_file", return_value=self.client), \
             patch.object(main, "DecisionJournal", return_value=self.journal), \
             patch.object(main, "RiskManager", return_value=self.manager), \
             patch.object(main, "get_public_quote_evidence", return_value=self.market()), \
             patch.object(main, "OkxClient"), \
             patch.object(main, "BudgetInitializer", side_effect=factory), \
             redirect_stdout(output), redirect_stderr(error):
            self.assertEqual(main.main(["--budget-preview"]), 0)
            token = output.getvalue().split("--budget-token ")[1].split()[0]
            self.assertFalse(self.manager.state_path.exists())
            self.assertEqual(main.main(["--confirm-budget", CONFIRMATION_PHRASE, "--budget-token", token]), 0)
        self.assertIn("подтверждён один раз", output.getvalue())
        self.assertNotIn("dummy-secret", output.getvalue() + error.getvalue())
        self.assertTrue(self.manager.state_path.exists())

    def test_cli_reconcile_after_confirmation_shows_separate_strategy_risk(self):
        token = self.initializer().preview().token
        self.initializer().confirm(CONFIRMATION_PHRASE, token)
        output = StringIO()
        with patch.object(main.DemoClient, "from_env_file", return_value=self.client), \
             patch.object(main, "DecisionJournal", return_value=self.journal), \
             patch.object(main, "RiskManager", return_value=self.manager), \
             patch.object(main, "get_public_quote_evidence", return_value=self.market()), \
             patch.object(main, "OkxClient"), \
             patch.object(main, "DemoReconciler", side_effect=lambda **kwargs: DemoReconciler(**kwargs, clock=lambda: NOW)), \
             redirect_stdout(output):
            self.assertEqual(main.main(["--reconcile"]), 0)
        rendered = output.getvalue()
        self.assertIn("READY", rendered)
        self.assertIn("Капитал стратегии: 5000 USDT", rendered)
        self.assertIn("Лимит риска на сделку: 25.000 USDT", rendered)
        self.assertIn("предел 100.00", rendered)
        self.assertIn("Внешний BTC: 1", rendered)
        self.assertNotIn("dummy-secret", rendered)


if __name__ == "__main__":
    unittest.main()
