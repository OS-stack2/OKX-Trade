"""Offline first Demo trade preflight and full synthetic BUY/exit accounting."""
from __future__ import annotations

from dataclasses import replace
from contextlib import redirect_stderr
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
import unittest
import main

from bot.demo_client import DemoClient
from bot.execution_engine import ExecutionEngine
from bot.first_trade import FirstTradePreflight, REVIEW_PHRASE
from bot.journal import CycleDraft, CycleStatus, DecisionJournal
from bot.market_data import Candle, MarketSnapshot, SpotInstrument
from bot.order_preview import PROTECTION_BLOCK, preview_demo_buy
from bot.protection_simulator import SimEvent, SimulationLedger
from bot.reconcile import ReconcileReport, ReconcileStatus
from bot.risk_manager import AccountSnapshot, ExecutionCosts, RiskManager
from bot.strategy import Signal, StrategyResult
from bot.strategy_budget import BudgetBaseline, StrategyBudgetStore
from config import Config

NOW = datetime(2026, 10, 15, 12, 30, tzinfo=timezone.utc)
SPOT = SpotInstrument("BTC-USDT", D("0.1"), D("0.0001"), D("0.0001"),
                      "live", "BTC", "USDT", D("0.1"), D("0.1"))
COSTS = ExecutionCosts(D("0.001"), D("0.001"), D("0.001"), D("0.001"))


class FakeDemo(DemoClient):
    def __init__(self, mode="DEMO"):
        self._config = Config(mode=mode, region="global")


class FirstTradeTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.journal = DecisionJournal(self.root / "journal.sqlite3")
        self.risk = RiskManager(self.root / "risk.json")
        self.budget = StrategyBudgetStore(self.journal.path)
        self.risk.initialize_day(AccountSnapshot(NOW, NOW.date(), D("5000"), D("5000"),
            D("5000"), D(0), D(0), 0, 0, 0), NOW)
        self.journal.append(CycleDraft(NOW - timedelta(seconds=20), NOW - timedelta(minutes=30),
                                       "HOLD", CycleStatus.HOLD, "seed"))
        self.budget.confirm_once(BudgetBaseline(NOW - timedelta(minutes=2), D("5000"),
            D("1"), D("1"), D("5000"), D("5000"), D("80000"), D("85000"),
            D("-0.0008"), D("-0.001"), 1))
        self.engine = ExecutionEngine(journal=self.journal, risk_manager=self.risk,
                                      budget_store=self.budget, clock=lambda: NOW)
        candles = tuple(Candle(NOW - timedelta(minutes=15 * (100 - index)),
                               D("80000"), D("80000"), D("80000"), D("80000"), True)
                        for index in range(100))
        self.market = MarketSnapshot(SPOT, D("80000"), candles, NOW)
        self.signal = StrategyResult(Signal.BUY, D("80001"), D("80000"),
                                     D("79999"), D("80000"), NOW, "synthetic crossover")
        self.report = ReconcileReport(ReconcileStatus.READY, (), checked_at_utc=NOW,
            btc_total=D("1"), btc_available=D("1"), usdt_total=D("5000"),
            usdt_available=D("5000"), external_btc=D("1"), bot_btc=D(0),
            open_order_count=0, fill_count=0, instrument=SPOT,
            confirmed_realized_pnl_usdt=D(0), confirmed_unrealized_pnl_usdt=D(0),
            daily_loss_usdt=D(0), daily_loss_limit_usdt=D("100"),
            daily_risk_remaining_usdt=D("100"), strategy_capital_usdt=D("5000"),
            strategy_equity_usdt=D("5000"), per_trade_risk_limit_usdt=D("25"),
            taker_fee_rate=D("-0.001"))
        self.market_provider = Mock(return_value=self.market)
        self.reconciliation_provider = Mock(return_value=self.report)
        self.preview_provider = lambda **kwargs: preview_demo_buy(
            **kwargs, verified_max_size_btc=D("0.1"), verified_costs=COSTS)

    def preflight(self, *, mode="DEMO", signal=None, report=None):
        if report is not None:
            self.reconciliation_provider.return_value = report
        return FirstTradePreflight(
            demo_client=FakeDemo(mode), engine=self.engine,
            market_provider=self.market_provider,
            reconciliation_provider=self.reconciliation_provider,
            signal_provider=lambda _candles: signal or self.signal,
            preview_provider=self.preview_provider, clock=lambda: NOW)

    def test_buy_risk_preview_uses_budget_but_never_reserves_or_sends(self):
        risk_before = self.risk.state_path.read_bytes()
        journal_before = self.journal.path.read_bytes()
        with patch.object(DemoClient, "_get", side_effect=AssertionError("HTTP forbidden")), \
             patch.object(self.engine, "plan_buy", side_effect=AssertionError("plan forbidden")), \
             patch.object(self.engine, "mark_submitting", side_effect=AssertionError("send forbidden")):
            result = self.preflight().run_once(confirmation=REVIEW_PHRASE)
        self.assertEqual(result.status, "BLOCK")
        self.assertTrue(result.manual_review_confirmed)
        self.assertEqual(result.stages[0], ("EMA", "BUY"))
        self.assertEqual(result.stages[1], ("RISK", "PREVIEW_ONLY"))
        self.assertIsNotNone(result.preview.quantity_btc)
        self.assertLessEqual(result.preview.estimated_risk_usdt, D("25"))
        self.assertEqual(result.reconciliation.external_btc, D("1"))
        self.assertIn(PROTECTION_BLOCK, result.reasons)
        self.assertEqual(self.risk.state_path.read_bytes(), risk_before)
        self.assertEqual(self.journal.path.read_bytes(), journal_before)
        self.assertEqual(self.engine.diagnose().intents, ())

    def test_insufficient_usdt_and_unknown_reconciliation_block(self):
        for report in (replace(self.report, usdt_available=D("1")),
                       replace(self.report, status=ReconcileStatus.BLOCKED)):
            with self.subTest(report=report.status):
                result = self.preflight(report=report).run_once()
                self.assertEqual(result.status, "BLOCK")
                self.assertIsNone(result.preview.quantity_btc)
                self.assertEqual(result.stages[2], ("BUY", "POST_DISABLED"))

    def test_hold_and_sell_never_use_external_btc(self):
        for signal in (Signal.HOLD, Signal.SELL):
            with self.subTest(signal=signal):
                result = self.preflight(signal=replace(self.signal, signal=signal)).run_once()
                self.assertEqual(result.status, "BLOCK")
                self.assertIsNone(result.preview.quantity_btc)
                self.assertEqual(result.reconciliation.external_btc, D("1"))
                self.assertTrue(any("SELL внешнего BTC запрещён" in x for x in result.reasons))

    def test_wrong_phrase_live_mode_and_halt_block_before_market_get(self):
        self.assertEqual(self.preflight().run_once(confirmation="wrong").status, "BLOCK")
        self.market_provider.assert_not_called()
        self.assertEqual(self.preflight(mode="READ_ONLY").run_once().status, "BLOCK")
        self.market_provider.assert_not_called()
        self.engine.emergency_halt("PROTECTION_UNVERIFIED")
        self.assertEqual(self.preflight().run_once().status, "BLOCK")
        self.market_provider.assert_not_called()

    def test_get_error_is_sanitized_and_not_retried_as_order(self):
        self.market_provider.side_effect = RuntimeError("secret and signed URL")
        result = self.preflight().run_once(confirmation=REVIEW_PHRASE)
        self.assertEqual(result.status, "BLOCK")
        self.assertNotIn("secret", " ".join(result.reasons))
        self.assertFalse(self.engine.diagnose().intents)

    def test_full_synthetic_close_pnl_restart_and_duplicate_safety(self):
        path = self.root / "simulation.sqlite3"
        ledger = SimulationLedger(path)
        ledger.append(SimEvent("ack", 0, "BUY_ACK"))
        ledger.append(SimEvent("buy", 1, "BUY_FILL", quantity_btc=D("0.02"),
                               price_usdt=D("80000"), fee_btc=D("-0.00002"), order_id="fill-1"))
        ledger.append(SimEvent("oco", 1, "PROTECTION_ACK", quantity_btc=D("0.01998"),
                               stop_size_btc=D("0.01998"), take_size_btc=D("0.01998"),
                               stop_price_usdt=D("79200"), take_price_usdt=D("81600"),
                               stop_id="stop-1", take_id="take-1", group_id="oco-1", exclusive_oco=True))
        close = SimEvent("close", 30, "PROTECTION_FILL", quantity_btc=D("0.01998"),
                         price_usdt=D("81600"), fee_usdt=D("-1.630368"), order_id="take-1")
        first = ledger.append(close)
        self.assertEqual(first.simulated_realized_pnl_usdt, D("28.737632"))
        self.assertEqual(first.bot_remaining_btc, D(0))
        self.assertEqual(first.external_btc, D("1"))
        self.assertTrue(first.halted)
        self.assertEqual(SimulationLedger(path).append(close), first)
        double = ledger.append(SimEvent("double-sell", 31, "PROTECTION_FILL",
                                        quantity_btc=D("0.01998"), price_usdt=D("79200"),
                                        order_id="stop-1"))
        self.assertIsNone(double.simulated_realized_pnl_usdt)
        self.assertEqual(double.external_btc, D("1"))
        self.assertTrue(double.halted)

    def test_timeout_and_unprotected_fill_have_no_final_pnl(self):
        path = self.root / "simulation.sqlite3"
        ledger = SimulationLedger(path)
        ledger.append(SimEvent("ack", 0, "BUY_ACK"))
        self.assertIsNone(ledger.append(SimEvent("timeout", 1, "BUY_TIMEOUT"))
                          .simulated_realized_pnl_usdt)
        self.assertTrue(SimulationLedger(path).replay().halted)

        partial = SimulationLedger(self.root / "partial.sqlite3")
        partial.append(SimEvent("ack", 0, "BUY_ACK"))
        partial.append(SimEvent("buy", 1, "BUY_FILL", quantity_btc=D("0.02"),
                                price_usdt=D("80000"), fee_btc=D("-0.00002"), order_id="fill-1"))
        partial.append(SimEvent("oco", 1, "PROTECTION_ACK", quantity_btc=D("0.01998"),
                                stop_size_btc=D("0.01998"), take_size_btc=D("0.01998"),
                                stop_price_usdt=D("79200"), take_price_usdt=D("81600"),
                                stop_id="stop-1", take_id="take-1", group_id="oco-1", exclusive_oco=True))
        result = partial.append(SimEvent("partial-exit", 30, "PROTECTION_FILL",
                                         quantity_btc=D("0.005"), price_usdt=D("79200"),
                                         fee_usdt=D("-0.396"), order_id="stop-1"))
        self.assertIsNone(result.simulated_realized_pnl_usdt)
        self.assertEqual(result.bot_remaining_btc, D("0.01498"))
        self.assertTrue(result.halted)

    def test_cli_wrong_phrase_blocks_before_credentials_or_http(self):
        with patch.object(main.DemoClient, "from_env_file",
                          side_effect=AssertionError("credentials must not load")), \
             redirect_stderr(StringIO()) as errors:
            self.assertEqual(main.main(["--first-demo-trade", "--first-demo-confirm", "wrong"]), 1)
        self.assertIn("BLOCK", errors.getvalue())
        self.assertNotIn("credentials", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
