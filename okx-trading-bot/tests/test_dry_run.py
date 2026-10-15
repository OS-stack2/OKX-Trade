"""Offline tests for one non-executing Demo analysis cycle."""

from __future__ import annotations

from contextlib import closing, redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from io import StringIO
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.parse import urlsplit

import main
from bot.demo_client import CurrencyBalance, DemoClient, DemoCredentials, DemoError, DemoSummary
from bot.dry_run import DryRunOrchestrator, RiskEvidence
from bot.journal import CycleStatus, DecisionJournal, JournalError
from bot.market_data import Candle, MarketSnapshot, SpotInstrument
from bot.risk_manager import AccountSnapshot, ExecutionCosts, InstrumentLimits, RiskManager, RiskSettings
from config import Config, ConfigError


NOW = datetime(2026, 10, 15, 12, 30, tzinfo=timezone.utc)
INSTRUMENT = SpotInstrument("BTC-USDT", Decimal("0.01"), Decimal("0.0001"), Decimal("0.0001"))
COSTS = ExecutionCosts(Decimal("0.001"), Decimal("0.001"), Decimal("0.001"), Decimal("0.001"))


def market(signal: str = "BUY", now: datetime = NOW) -> MarketSnapshot:
    prices = {"BUY": [100] * 20 + [90, 120], "SELL": [100] * 20 + [110, 80],
              "HOLD": [100] * 22}[signal]
    last_closed = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
    start = last_closed - timedelta(minutes=15 * len(prices))
    candles = tuple(Candle(
        opened_at=start + timedelta(minutes=15 * index),
        open=Decimal(price), high=Decimal(price), low=Decimal(price),
        close=Decimal(price), confirmed=True,
    ) for index, price in enumerate(prices))
    return MarketSnapshot(INSTRUMENT, Decimal(prices[-1]), candles, now)


def demo(usdt_available: str = "5000", orders: int = 0, now: datetime = NOW) -> DemoSummary:
    return DemoSummary(
        CurrencyBalance(Decimal("1"), Decimal("1")),
        CurrencyBalance(Decimal("5000"), Decimal(usdt_available)),
        orders, 0, now,
    )


def account(usdt_available: str = "5000", orders: int = 0, now: datetime = NOW) -> AccountSnapshot:
    return AccountSnapshot(
        as_of_utc=now, day_utc=now.date(), equity_usdt=Decimal("5120"),
        day_start_capital_usdt=Decimal("5120"), available_usdt=Decimal(usdt_available),
        realized_pnl_today_usdt=Decimal("0"), unrealized_pnl_usdt=Decimal("0"),
        open_positions=0, pending_orders=orders, entries_today=0,
    )


def evidence(snapshot: AccountSnapshot | None = None) -> RiskEvidence:
    return RiskEvidence(
        account=snapshot or account(),
        instrument=InstrumentLimits("BTC-USDT", "live", INSTRUMENT.min_size,
                                    INSTRUMENT.lot_size, INSTRUMENT.tick_size, Decimal("1000")),
        costs=COSTS, provenance_verified=True,
    )


class DryRunTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.state_path = self.root / "risk_state.json"
        self.manager = RiskManager(self.state_path)
        self.manager.initialize_day(account(), NOW)
        self.journal = DecisionJournal(self.root / "decisions.sqlite3")

    def run_cycle(self, signal="BUY", summary=None, verified=None, now=NOW, snapshot=None):
        return DryRunOrchestrator(
            market_provider=lambda: snapshot or market(signal, now),
            demo_provider=lambda: summary or demo(now=now),
            risk_manager=self.manager, journal=self.journal,
            evidence_provider=verified, clock=lambda: now,
        ).run_once()

    def test_buy_with_sufficient_virtual_usdt_is_only_hypothetically_allowed(self):
        before = self.state_path.read_bytes()
        record = self.run_cycle(verified=lambda market, demo, now: evidence())
        self.assertEqual(record.draft.status, CycleStatus.ALLOW)
        self.assertGreater(record.draft.quantity, 0)
        self.assertEqual(record.draft.entry_price, Decimal("120"))
        self.assertLess(record.draft.stop_price, record.draft.entry_price)
        self.assertGreater(record.draft.take_profit_price, record.draft.entry_price)
        self.assertFalse(record.executed)
        self.assertEqual(self.state_path.read_bytes(), before)
        self.assertFalse(Path(f"{self.state_path}.lock").exists())

    def test_buy_with_insufficient_virtual_usdt_is_blocked(self):
        summary = demo(usdt_available="0.005")
        record = self.run_cycle(
            summary=summary,
            verified=lambda market, demo, now: evidence(account(usdt_available="0.005")),
        )
        self.assertEqual(record.draft.status, CycleStatus.BLOCK)
        self.assertIn("USDT", record.draft.reason)
        self.assertIsNone(record.draft.quantity)

    def test_hold_does_not_need_buy_evidence(self):
        record = self.run_cycle(signal="HOLD")
        self.assertEqual(record.draft.status, CycleStatus.HOLD)
        self.assertEqual(record.draft.virtual_btc_available, Decimal("1"))
        self.assertIsNone(record.draft.quantity)

    def test_sell_never_targets_external_demo_btc(self):
        record = self.run_cycle(signal="SELL", verified=lambda market, demo, now: evidence())
        self.assertEqual(record.draft.status, CycleStatus.BLOCK)
        self.assertEqual(record.draft.signal, "SELL")
        self.assertIn("внешний BTC", record.draft.reason)
        self.assertIsNone(record.draft.quantity)

    def test_open_order_blocks_buy_even_with_complete_evidence(self):
        record = self.run_cycle(
            summary=demo(orders=1),
            verified=lambda market, demo, now: evidence(account(orders=1)),
        )
        self.assertEqual(record.draft.status, CycleStatus.BLOCK)
        self.assertIn("SPOT-ордер", record.draft.reason)

    def test_owned_position_daily_loss_and_entry_limit_block_without_state_change(self):
        before = self.state_path.read_bytes()
        cases = (
            (replace(account(), open_positions=1), "открытая позиция"),
            (replace(account(now=NOW + timedelta(minutes=15)), realized_pnl_today_usdt=Decimal("-103")), "дневной лимит"),
            (replace(account(now=NOW + timedelta(minutes=30)), entries_today=5), "суточный лимит"),
        )
        for index, (snapshot, phrase) in enumerate(cases):
            checked_at = NOW + timedelta(minutes=15 * index)
            with self.subTest(phrase=phrase):
                record = self.run_cycle(
                    now=checked_at, summary=demo(now=checked_at),
                    verified=lambda market, demo, now, account_snapshot=snapshot: evidence(account_snapshot),
                )
                self.assertEqual(record.draft.status, CycleStatus.BLOCK)
                self.assertIn(phrase, record.draft.reason)
                self.assertEqual(self.state_path.read_bytes(), before)

    def test_api_failure_and_unknown_risk_state_are_blocked_and_recorded(self):
        failed = DryRunOrchestrator(
            market_provider=lambda: market(), demo_provider=lambda: (_ for _ in ()).throw(DemoError("test-secret")),
            risk_manager=self.manager, journal=self.journal, clock=lambda: NOW,
        ).run_once()
        self.assertEqual(failed.draft.status, CycleStatus.BLOCK)
        self.assertNotIn("test-secret", failed.draft.reason)
        self.assertIsNone(failed.draft.virtual_usdt_available)
        # A new candle is needed because duplicate-candle decisions are deliberately idempotent.
        tomorrow = NOW + timedelta(days=1)
        unknown = self.run_cycle(now=tomorrow, summary=demo(now=tomorrow))
        self.assertEqual(unknown.draft.status, CycleStatus.BLOCK)
        self.assertIn("Нет подтверждённых", unknown.draft.reason)

    def test_missing_or_contradictory_evidence_and_missing_risk_state_block(self):
        bad = self.run_cycle(verified=lambda market, demo, now: evidence(account(usdt_available="4000")))
        self.assertEqual(bad.draft.status, CycleStatus.BLOCK)
        self.assertIn("противоречат", bad.draft.reason)
        later = NOW + timedelta(minutes=15)
        self.state_path.unlink()
        missing = self.run_cycle(now=later, summary=demo(now=later),
                                 verified=lambda market, demo, now: evidence(account(now=later)))
        self.assertEqual(missing.draft.status, CycleStatus.BLOCK)
        self.assertIn("состояние риска отсутствует", missing.draft.reason)
        self.assertFalse(self.state_path.exists())

    def test_stale_market_and_demo_data_block(self):
        stale_candle = market(now=NOW - timedelta(hours=1))
        self.assertIn("свеча устарела", self.run_cycle(snapshot=stale_candle).draft.reason)
        stale_ticker = replace(market(), ticker_as_of_utc=NOW - timedelta(minutes=5))
        self.assertIn("свежего времени", self.run_cycle(snapshot=stale_ticker).draft.reason)
        later = NOW + timedelta(minutes=15)
        stale_demo = self.run_cycle(now=later, summary=demo(now=NOW))
        self.assertEqual(stale_demo.draft.status, CycleStatus.BLOCK)
        self.assertIn("устарел", stale_demo.draft.reason)

    def test_contradictory_demo_balance_blocks(self):
        invalid = replace(demo(), usdt=CurrencyBalance(Decimal("1"), Decimal("2")))
        record = self.run_cycle(summary=invalid)
        self.assertEqual(record.draft.status, CycleStatus.BLOCK)
        self.assertIn("противоречивые", record.draft.reason)

    def test_same_candle_references_original_and_has_no_new_intent(self):
        provider = lambda market, demo, now: evidence()
        first = self.run_cycle(verified=provider)
        second = self.run_cycle(verified=provider)
        self.assertEqual(first.draft.status, CycleStatus.ALLOW)
        self.assertEqual(second.draft.status, CycleStatus.BLOCK)
        self.assertEqual(second.reference_id, first.id)
        self.assertIsNone(second.draft.quantity)
        with closing(sqlite3.connect(self.journal.path)) as connection:
            rows = connection.execute("SELECT id, reference_id, executed FROM decisions ORDER BY rowid").fetchall()
        self.assertEqual(rows, [(first.id, None, 0), (second.id, first.id, 0)])

    def test_journal_survives_restart_and_contains_no_secrets(self):
        first = self.run_cycle(signal="HOLD")
        restarted = DecisionJournal(self.journal.path)
        second = DryRunOrchestrator(
            market_provider=lambda: market("HOLD"), demo_provider=lambda: demo(),
            risk_manager=RiskManager(self.state_path), journal=restarted,
            clock=lambda: NOW,
        ).run_once()
        self.assertEqual(second.reference_id, first.id)
        with closing(sqlite3.connect(self.journal.path)) as connection:
            rows = connection.execute(
                "SELECT status, signal, risk_parameters, executed FROM decisions ORDER BY rowid"
            ).fetchall()
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][0:2], ("HOLD", "HOLD"))
        self.assertIn("risk_per_trade", rows[0][2])
        self.assertTrue(all(row[3] == 0 for row in rows))
        blob = self.journal.path.read_bytes()
        for secret in (b"test-secret", b"OK-ACCESS-KEY", b"test-passphrase"):
            self.assertNotIn(secret, blob)

    def test_demo_http_requests_during_cycle_are_get_only(self):
        requests = []

        def transport(request, timeout):
            requests.append(request)
            path = urlsplit(request.full_url).path
            if path.endswith("/balance"):
                data = [{"details": [{"ccy": "BTC", "cashBal": "1", "availBal": "1"},
                                     {"ccy": "USDT", "cashBal": "5000", "availBal": "5000"}]}]
            else:
                data = []
            return {"code": "0", "data": data}

        client = DemoClient(
            Config(region="global", mode="DEMO"), DemoCredentials("test-key", "test-secret", "test-passphrase"),
            transport=transport, clock=lambda: NOW,
        )
        current = datetime.now(timezone.utc)
        result = DryRunOrchestrator(
            market_provider=lambda: market("HOLD", current), demo_provider=client.check_account,
            risk_manager=self.manager, journal=self.journal,
            clock=lambda: datetime.now(timezone.utc),
        ).run_once()
        self.assertEqual(result.draft.status, CycleStatus.HOLD)
        self.assertEqual(len(requests), 3)
        allowed = {"/api/v5/account/balance", "/api/v5/trade/orders-pending", "/api/v5/trade/fills-history"}
        self.assertEqual({urlsplit(request.full_url).path for request in requests}, allowed)
        for request in requests:
            self.assertEqual(request.get_method(), "GET")
            self.assertEqual(dict(request.header_items())["X-simulated-trading"], "1")
        self.assertNotIn(b"test-secret", self.journal.path.read_bytes())

    def test_cli_dry_run_prints_decision_without_execution(self):
        record = self.run_cycle(signal="HOLD")
        output = StringIO()
        with patch.object(main.DryRunOrchestrator, "run_once", return_value=record), redirect_stdout(output):
            self.assertEqual(main.main(["--dry-run"]), 0)
        self.assertIn("Решение: HOLD", output.getvalue())
        self.assertIn("executed=false", output.getvalue())

    def test_cli_missing_demo_config_records_block_without_network(self):
        output = StringIO()
        with patch.object(main.DemoClient, "from_env_file", side_effect=ConfigError("missing")), \
             patch.object(main, "DecisionJournal", return_value=self.journal), redirect_stdout(output):
            self.assertEqual(main.main(["--dry-run"]), 0)
        self.assertIn("Решение: BLOCK", output.getvalue())
        self.assertIn("OKX_MODE=DEMO", output.getvalue())

    def test_journal_failure_fails_closed(self):
        bad_journal = DecisionJournal(self.root)
        with self.assertRaises(JournalError):
            DryRunOrchestrator(
                market_provider=lambda: market(), demo_provider=lambda: demo(),
                risk_manager=self.manager, journal=bad_journal, clock=lambda: NOW,
            ).run_once()

    def test_invalid_risk_settings_record_block_before_network(self):
        invalid_manager = RiskManager(
            self.state_path, RiskSettings(risk_per_trade=Decimal("NaN"))
        )
        record = DryRunOrchestrator(
            market_provider=lambda: self.fail("network must not be used"),
            demo_provider=lambda: self.fail("network must not be used"),
            risk_manager=invalid_manager, journal=self.journal, clock=lambda: NOW,
        ).run_once()
        self.assertEqual(record.draft.status, CycleStatus.BLOCK)
        self.assertIn("Настройки Risk Manager", record.draft.reason)


if __name__ == "__main__":
    unittest.main()
