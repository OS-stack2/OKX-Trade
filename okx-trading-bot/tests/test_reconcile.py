"""Offline, read-only Demo reconciliation and fail-closed daily accounting."""

from __future__ import annotations

from dataclasses import replace
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from io import StringIO
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

from bot.demo_client import DemoClient, DemoCredentials
from bot.journal import CycleDraft, CycleStatus, DecisionJournal
from bot.market_data import MarketSnapshot, SpotInstrument
from bot.reconcile import ConfirmedDayEvidence, DemoReconciler, ReconcileStatus
from bot.risk_manager import AccountSnapshot, RiskManager
from config import Config, ConfigError
import main


NOW = datetime(2026, 10, 15, 12, 30, tzinfo=timezone.utc)
INSTRUMENT = SpotInstrument(
    "BTC-USDT", Decimal("0.01"), Decimal("0.0001"), Decimal("0.0001"),
    "live", "BTC", "USDT", Decimal("100"), Decimal("200"),
)


class ReconcileTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.journal = DecisionJournal(self.root / "journal.sqlite3")
        self.journal.append(CycleDraft(NOW, NOW - timedelta(minutes=15), "HOLD", CycleStatus.HOLD, "no cross"))
        self.manager = RiskManager(self.root / "risk.json")
        self.requests = []
        self.orders = []
        self.fills = []
        self.bad_endpoint = None
        self.client = DemoClient(
            Config(region="eea", mode="DEMO"), DemoCredentials("fake-key", "fake-secret", "fake-passphrase"),
            transport=self.transport, clock=lambda: NOW,
        )

    def transport(self, request, timeout):
        self.requests.append(request)
        self.assertEqual(request.get_method(), "GET")
        self.assertEqual(urlsplit(request.full_url).netloc, "eea.okx.com")
        self.assertEqual(dict(request.header_items())["X-simulated-trading"], "1")
        path = urlsplit(request.full_url).path
        if path == self.bad_endpoint:
            return {"code": "50113", "msg": "fake-secret", "data": []}
        data = {
            "/api/v5/account/balance": [{"details": [
                {"ccy": "BTC", "cashBal": "1", "availBal": "1"},
                {"ccy": "USDT", "cashBal": "5000", "availBal": "5000"},
            ]}],
            "/api/v5/trade/orders-pending": self.orders,
            "/api/v5/trade/fills-history": self.fills,
            "/api/v5/account/trade-fee": [{"maker": "-0.0008", "taker": "-0.001"}],
        }[path]
        return {"code": "0", "data": data}

    def confirmed(self):
        return ConfirmedDayEvidence(NOW, Decimal("5100"), Decimal("0"),
                                    Decimal("0"), Decimal("0"), 0, True, Decimal("0"))

    def initialize_day(self):
        self.manager.initialize_day(AccountSnapshot(
            NOW, NOW.date(), Decimal("5100"), Decimal("5100"), Decimal("5000"),
            Decimal("0"), Decimal("0"), 0, 0, 0,
        ), NOW)

    def run_reconcile(self, *, day_provider=None, instrument=INSTRUMENT, price=Decimal("100")):
        return DemoReconciler(
            market_provider=lambda: MarketSnapshot(instrument, price, (), NOW),
            demo_client=self.client, journal=self.journal, risk_manager=self.manager,
            day_provider=day_provider, clock=lambda: NOW,
        ).run_once()

    def test_complete_explicit_evidence_is_ready_without_writing_risk_state(self):
        self.initialize_day()
        before = self.manager.state_path.read_bytes()
        report = self.run_reconcile(day_provider=self.confirmed)
        self.assertEqual(report.status, ReconcileStatus.READY)
        self.assertEqual(report.portfolio_value_usdt, Decimal("5100"))
        self.assertEqual((report.external_btc, report.bot_btc), (Decimal("1"), Decimal("0")))
        self.assertEqual(report.daily_loss_limit_usdt, Decimal("102"))
        self.assertEqual(report.daily_risk_remaining_usdt, Decimal("102"))
        self.assertEqual(report.fees_by_currency, (("BTC", Decimal("0")), ("USDT", Decimal("0"))))
        self.assertEqual(report.maker_fee_rate, Decimal("-0.0008"))
        self.assertEqual(self.manager.state_path.read_bytes(), before)
        self.assertEqual(len(self.requests), 4)
        self.assertTrue(all(request.get_method() == "GET" for request in self.requests))

    def test_existing_external_btc_and_no_baseline_remain_blocked(self):
        report = self.run_reconcile()
        self.assertEqual(report.status, ReconcileStatus.BLOCKED)
        self.assertEqual(report.external_btc, Decimal("1"))
        self.assertIsNone(report.confirmed_realized_pnl_usdt)
        self.assertIsNone(report.daily_risk_remaining_usdt)
        self.assertFalse(self.manager.state_path.exists())

    def test_saved_baseline_without_confirmed_pnl_is_still_blocked(self):
        self.initialize_day()
        before = self.manager.state_path.read_bytes()
        report = self.run_reconcile()
        self.assertEqual(report.status, ReconcileStatus.BLOCKED)
        self.assertIsNone(report.daily_loss_usdt)
        self.assertEqual(self.manager.state_path.read_bytes(), before)

    def test_open_order_and_unmatched_fill_block(self):
        self.initialize_day()
        self.orders = [{"instType": "SPOT", "instId": "BTC-USDT", "ordId": "o1", "state": "live"}]
        self.fills = [{"instType": "SPOT", "instId": "BTC-USDT", "tradeId": "t1",
                       "feeCcy": "USDT", "fee": "-0.1", "fillSz": "0.01", "fillPx": "100",
                       "side": "buy", "ts": str(int(NOW.timestamp() * 1000))}]
        report = self.run_reconcile(day_provider=self.confirmed)
        self.assertEqual(report.status, ReconcileStatus.BLOCKED)
        self.assertEqual(report.fees_by_currency[1], ("USDT", Decimal("-0.1")))
        self.assertTrue(any("ордер" in reason for reason in report.reasons))
        self.assertTrue(any("Исполнения" in reason for reason in report.reasons))
        self.assertIsNone(report.confirmed_realized_pnl_usdt)

    def test_other_spot_fill_and_missing_fee_block_without_guessing_pnl(self):
        self.initialize_day()
        self.fills = [{"instType": "SPOT", "instId": "SOL-USDT", "tradeId": "t1"}]
        report = self.run_reconcile(day_provider=self.confirmed)
        self.assertEqual(report.status, ReconcileStatus.BLOCKED)
        self.assertIsNone(report.fees_by_currency)
        self.assertIsNone(report.daily_loss_usdt)

    def test_demo_api_error_is_safe_and_does_not_fall_back(self):
        self.initialize_day()
        self.bad_endpoint = "/api/v5/account/balance"
        report = self.run_reconcile(day_provider=self.confirmed)
        self.assertEqual(report.status, ReconcileStatus.BLOCKED)
        self.assertNotIn("fake-secret", repr(report))
        self.assertTrue(all(urlsplit(request.full_url).netloc == "eea.okx.com" for request in self.requests))

    def test_stale_price_missing_limits_and_missing_journal_block(self):
        self.initialize_day()
        self.journal.path.unlink()
        report = self.run_reconcile(day_provider=self.confirmed, instrument=replace(INSTRUMENT, max_market_size=None))
        self.assertEqual(report.status, ReconcileStatus.BLOCKED)
        self.assertTrue(any("Журнал" in reason for reason in report.reasons))
        self.assertTrue(any("ограничения" in reason for reason in report.reasons))

    def test_unverified_pnl_and_wrong_baseline_block(self):
        self.initialize_day()
        for evidence in (replace(self.confirmed(), complete_and_verified=False),
                         replace(self.confirmed(), day_start_capital_usdt=Decimal("5200")),
                         replace(self.confirmed(), bot_btc=Decimal("1"))):
            with self.subTest(evidence=evidence):
                report = self.run_reconcile(day_provider=lambda value=evidence: value)
                self.assertEqual(report.status, ReconcileStatus.BLOCKED)
                self.assertIsNone(report.daily_risk_remaining_usdt)

    def test_risk_loss_and_lock_do_not_clear_during_reconciliation(self):
        self.initialize_day()
        evidence = replace(self.confirmed(), realized_pnl_today_usdt=Decimal("-100"),
                           unrealized_pnl_usdt=Decimal("-2"),
                           net_external_cash_flow_usdt=Decimal("102"))
        before = self.manager.state_path.read_bytes()
        report = self.run_reconcile(day_provider=lambda: evidence)
        self.assertEqual(report.status, ReconcileStatus.BLOCKED)
        self.assertEqual(report.daily_loss_usdt, Decimal("102"))
        self.assertEqual(self.manager.state_path.read_bytes(), before)

    def test_cli_reports_blocked_without_secrets_or_risk_writes(self):
        output = StringIO()
        with patch.object(main.DemoClient, "from_env_file", return_value=self.client), \
             patch.object(main, "DecisionJournal", return_value=self.journal), \
             patch.object(main, "RiskManager", return_value=self.manager), \
             patch.object(main, "get_market_snapshot", return_value=MarketSnapshot(INSTRUMENT, Decimal("100"), (), NOW)), \
             patch.object(main, "OkxClient"), redirect_stdout(output), \
             patch.object(main, "DemoReconciler", side_effect=lambda **kwargs: DemoReconciler(**kwargs, clock=lambda: NOW)):
            self.assertEqual(main.main(["--reconcile"]), 1)
        rendered = output.getvalue()
        self.assertIn("BLOCKED", rendered)
        self.assertIn("Оценка BTC+USDT: 5100", rendered)
        self.assertIn("Внешний BTC: 1", rendered)
        self.assertIn("дневная база не устанавливалась", rendered)
        self.assertNotIn("fake-secret", rendered)
        self.assertFalse(self.manager.state_path.exists())

    def test_missing_fill_fee_and_full_page_block(self):
        self.initialize_day()
        self.fills = [{"instType": "SPOT", "instId": "BTC-USDT", "tradeId": "t1"}]
        report = self.run_reconcile(day_provider=self.confirmed)
        self.assertEqual(report.status, ReconcileStatus.BLOCKED)
        self.assertIsNone(report.fees_by_currency)
        self.fills = [{"instType": "SPOT", "instId": "BTC-USDT", "tradeId": str(i)} for i in range(100)]
        report = self.run_reconcile(day_provider=self.confirmed)
        self.assertEqual(report.status, ReconcileStatus.BLOCKED)
        self.assertTrue(any("История" in reason for reason in report.reasons))

    def test_missing_demo_configuration_blocks_before_any_http_request(self):
        error = StringIO()
        with patch.object(main.DemoClient, "from_env_file", side_effect=ConfigError("fake-secret")), \
             redirect_stderr(error):
            self.assertEqual(main.main(["--reconcile"]), 1)
        self.assertIn("BLOCKED", error.getvalue())
        self.assertNotIn("fake-secret", error.getvalue())
        self.assertEqual(self.requests, [])


if __name__ == "__main__":
    unittest.main()
