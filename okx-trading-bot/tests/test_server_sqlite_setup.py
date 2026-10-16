"""Offline server SQLite bootstrap tests; every file lives in TemporaryDirectory."""

from contextlib import closing, redirect_stdout
from datetime import datetime, timezone
from io import StringIO
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import main
from bot.execution_engine import ExecutionEngine
from bot.journal import DecisionJournal
from bot.risk_manager import RiskManager
from bot.server_sqlite_setup import SQLITE_CONFIRMATION_PHRASE, ServerSqliteSetup, SqliteSetupError
from bot.strategy_budget import StrategyBudgetStore


NOW = datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc)


class ServerSqliteSetupTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.data = Path(temporary.name) / "data"
        self.data.mkdir(mode=0o700)
        self.path = self.data / "dry_run_journal.sqlite3"
        self.risk_path = self.data / "risk_state.json"
        self.journal = DecisionJournal(self.path)
        self.setup = ServerSqliteSetup(
            journal=self.journal, risk_manager=RiskManager(self.risk_path), clock=lambda: NOW)

    def initialize(self):
        token = self.setup.preview().token
        self.assertIsNotNone(token)
        return self.setup.initialize(SQLITE_CONFIRMATION_PHRASE, token)

    def test_read_only_preview_then_explicit_creation_keeps_budget_separate(self):
        report = self.setup.preview()
        self.assertEqual(report.status, "MISSING")
        self.assertEqual(report.schema, "MISSING")
        self.assertEqual(report.budget, "UNCONFIRMED")
        self.assertEqual(report.risk, "UNINITIALIZED")
        self.assertFalse(self.path.exists())
        with self.assertRaises(SqliteSetupError):
            self.setup.initialize("yes", report.token)
        with self.assertRaises(SqliteSetupError):
            self.setup.initialize(SQLITE_CONFIRMATION_PHRASE, "wrong")
        self.assertFalse(self.path.exists())
        done = self.setup.initialize(SQLITE_CONFIRMATION_PHRASE, report.token)
        self.assertEqual(done.status, "EXISTS")
        self.assertEqual(done.intent_count, 0)
        self.assertFalse(done.halted)
        self.assertEqual(self.journal.audit_read_only().decision_count, 1)
        self.assertIsNone(StrategyBudgetStore(self.path).read())
        self.assertFalse(self.risk_path.exists())
        self.assertEqual(list(self.data.glob(".sqlite-bootstrap-*")), [])
        if os.name == "posix":
            self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_repeat_cannot_overwrite_existing_valid_database(self):
        self.initialize()
        original = self.path.read_bytes()
        self.assertEqual(self.setup.preview().status, "EXISTS")
        with self.assertRaises(SqliteSetupError):
            self.setup.initialize(SQLITE_CONFIRMATION_PHRASE, "0" * 16)
        self.assertEqual(self.path.read_bytes(), original)

    def test_concurrent_target_appearing_before_publish_is_not_overwritten(self):
        token = self.setup.preview().token
        original_link = os.link

        def create_competing_file(source, destination):
            Path(destination).write_bytes(b"existing state")
            return original_link(source, destination)

        with patch("bot.server_sqlite_setup.os.link", side_effect=create_competing_file):
            with self.assertRaises(SqliteSetupError):
                self.setup.initialize(SQLITE_CONFIRMATION_PHRASE, token)
        self.assertEqual(self.path.read_bytes(), b"existing state")
        self.assertEqual(list(self.data.glob(".sqlite-bootstrap-*")), [])

    def test_corrupt_existing_database_is_blocked_and_unchanged(self):
        self.path.write_bytes(b"corrupt sqlite")
        self.path.chmod(0o600)
        before = self.path.read_bytes()
        result = self.setup.preview()
        self.assertEqual(result.status, "BLOCKED")
        self.assertEqual(result.schema, "UNKNOWN")
        self.assertIsNone(result.token)
        with self.assertRaises(SqliteSetupError):
            self.setup.initialize(SQLITE_CONFIRMATION_PHRASE, "0" * 16)
        self.assertEqual(self.path.read_bytes(), before)

    @unittest.skipUnless(os.name == "posix", "POSIX permission mode check")
    def test_missing_write_permission_blocks_creation(self):
        self.data.chmod(0o500)
        try:
            result = self.setup.preview()
            self.assertEqual(result.status, "BLOCKED")
            self.assertIsNone(result.token)
            self.assertFalse(self.path.exists())
        finally:
            self.data.chmod(0o700)

    @unittest.skipUnless(os.name == "posix", "POSIX private-file permission check")
    def test_existing_file_with_excess_permissions_is_not_rewritten(self):
        self.initialize()
        self.path.chmod(0o644)
        before = self.path.read_bytes()
        self.assertEqual(self.setup.preview().status, "BLOCKED")
        self.assertEqual(self.path.read_bytes(), before)

    def test_orphaned_bootstrap_file_requires_manual_inspection(self):
        orphan = self.data / ".sqlite-bootstrap-orphan.sqlite3"
        orphan.write_bytes(b"unknown")
        self.assertEqual(self.setup.preview().status, "BLOCKED")
        self.assertFalse(self.path.exists())

    def test_prior_risk_state_without_database_blocks_initialization(self):
        self.risk_path.write_text("{}", encoding="utf-8")
        self.assertEqual(self.setup.preview().status, "BLOCKED")
        self.assertFalse(self.path.exists())

    def test_unfinished_intent_blocks_and_is_never_deleted(self):
        self.initialize()
        engine = ExecutionEngine(journal=self.journal)
        client_id = "TB" + "b" * 30
        with closing(sqlite3.connect(self.path)) as db:
            engine._schema(db)
            db.execute("""INSERT INTO execution_intents (
                client_order_id, decision_id, candle_closed_at, mode, inst_id, side,
                state, quantity, entry_price, stop_price, take_profit_price,
                estimated_risk_usdt, min_size, lot_size, tick_size,
                created_at_utc, updated_at_utc, reason_code)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                client_id, "decision-1", "2026-10-16T11:45:00Z", "DEMO", "BTC-USDT",
                "BUY", "UNKNOWN", "0.01", "80000", "79200", "81600", "8",
                "0.0001", "0.0001", "0.1", "2026-10-16T11:46:00Z",
                "2026-10-16T11:47:00Z", "NETWORK_TIMEOUT"))
            db.commit()
        before = self.path.read_bytes()
        report = self.setup.preview()
        self.assertEqual(report.status, "BLOCKED")
        self.assertEqual(report.intent_count, 1)
        self.assertFalse(report.halted)
        self.assertEqual(self.path.read_bytes(), before)
        with self.assertRaises(SqliteSetupError):
            self.setup.initialize(SQLITE_CONFIRMATION_PHRASE, "0" * 16)
        self.assertEqual(self.path.read_bytes(), before)

    def test_halted_is_preserved_across_preview_and_repeat(self):
        self.initialize()
        ExecutionEngine(journal=self.journal).emergency_halt("MANUAL_HALT")
        before = self.path.read_bytes()
        report = self.setup.preview()
        self.assertEqual(report.status, "BLOCKED")
        self.assertTrue(report.halted)
        self.assertEqual(self.path.read_bytes(), before)
        with self.assertRaises(SqliteSetupError):
            self.setup.initialize(SQLITE_CONFIRMATION_PHRASE, "0" * 16)
        self.assertEqual(self.path.read_bytes(), before)

    def test_cli_preview_and_init_are_offline_and_keep_protection_unverified(self):
        output = StringIO()
        with (patch.object(main, "ServerSqliteSetup", return_value=self.setup),
              patch.object(main.DemoClient, "from_env_file", side_effect=AssertionError("OKX called")),
              redirect_stdout(output)):
            self.assertEqual(main.main(["--sqlite-prepare"]), 0)
        self.assertIn("MISSING", output.getvalue())
        self.assertFalse(self.path.exists())
        output = StringIO()
        with (patch.object(main, "ServerSqliteSetup", return_value=self.setup),
              redirect_stdout(output)):
            self.assertEqual(main.main(["--sqlite-init", SQLITE_CONFIRMATION_PHRASE,
                                        "--sqlite-token", self.setup.preview().token]), 0)
        self.assertIn("UNCONFIRMED", output.getvalue())
        self.assertIn("UNVERIFIED", output.getvalue())


if __name__ == "__main__":
    unittest.main()
