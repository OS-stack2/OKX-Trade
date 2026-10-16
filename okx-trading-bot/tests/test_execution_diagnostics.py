"""The detailed gate report only reads disposable local SQLite fixtures."""

from contextlib import closing, redirect_stdout
from datetime import datetime, timezone
from decimal import Decimal
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import sqlite3
import unittest
import urllib.request
from unittest.mock import patch

import main
from bot.execution_diagnostics import inspect_execution
from bot.execution_engine import ExecutionEngine
from bot.journal import CycleDraft, CycleStatus, DecisionJournal


NOW = datetime(2026, 10, 16, 10, 0, tzinfo=timezone.utc)
CLIENT_ID = "TB" + "a" * 30


class ExecutionDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "journal.sqlite3"
        self.journal = DecisionJournal(self.path)
        self.record = self.journal.append(CycleDraft(
            NOW, NOW, "HOLD", CycleStatus.HOLD, "fixture"))
        self.engine = ExecutionEngine(journal=self.journal, clock=lambda: NOW)

    def add_intent(self, state="PLANNED", reason="PLANNED", filled="0"):
        with closing(sqlite3.connect(self.path)) as db:
            self.engine._schema(db)
            db.execute("""INSERT INTO execution_intents (
                client_order_id, decision_id, candle_closed_at, mode, inst_id, side,
                state, quantity, entry_price, stop_price, take_profit_price,
                estimated_risk_usdt, min_size, lot_size, tick_size,
                created_at_utc, updated_at_utc, filled_quantity, reason_code)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                CLIENT_ID, self.record.id, "2026-10-16T09:45:00Z", "DEMO", "BTC-USDT",
                "BUY", state, "0.01", "80000", "79200", "81600", "8",
                "0.0001", "0.0001", "0.1", "2026-10-16T09:46:00Z",
                "2026-10-16T09:47:00Z", filled, reason))
            db.commit()

    def test_empty_decision_journal_has_no_intent_but_protection_is_unverified(self):
        before = self.path.read_bytes()
        result = inspect_execution(self.engine)
        self.assertEqual(result.sqlite_status, "CLEAR")
        self.assertFalse(result.halted)
        self.assertIsNone(result.latest_intent)
        self.assertEqual(result.protection_status, "UNVERIFIED")
        self.assertEqual(self.path.read_bytes(), before)

    def test_halted_reason_and_timestamp_survive_restart_without_writes(self):
        self.add_intent("PARTIALLY_FILLED", "FILL_OBSERVED", "0.005")
        self.engine.emergency_halt("PARTIAL_BUY_ACTIVE")
        before = self.path.read_bytes()
        report = inspect_execution(ExecutionEngine(journal=self.journal))
        self.assertEqual(report.sqlite_status, "BLOCKED")
        self.assertTrue(report.halted)
        self.assertEqual(report.halt_reason_code, "PARTIAL_BUY_ACTIVE")
        self.assertIsNotNone(report.halt_updated_at_utc)
        self.assertEqual(report.latest_intent.state.value, "HALTED")
        self.assertEqual(report.latest_intent.filled_quantity, Decimal("0.005"))
        self.assertTrue(report.latest_intent.flags)
        self.assertEqual(self.path.read_bytes(), before)

    def test_unknown_status_is_blocked_and_preserves_uncertain_flag(self):
        self.add_intent("UNKNOWN", "NETWORK_TIMEOUT")
        report = inspect_execution(self.engine)
        self.assertEqual(report.sqlite_status, "BLOCKED")
        self.assertFalse(report.halted)
        self.assertEqual(report.latest_intent.reason_code, "NETWORK_TIMEOUT")
        self.assertIn("не подтверждён", report.latest_intent.flags[0])

    def test_halt_without_intent_has_exact_saved_reason(self):
        self.engine.emergency_halt("MANUAL_HALT")
        result = inspect_execution(self.engine)
        self.assertEqual(result.sqlite_status, "BLOCKED")
        self.assertEqual(result.halt_reason_code, "MANUAL_HALT")
        self.assertEqual(result.intent_count, 0)
        self.assertIsNone(result.latest_intent)

    def test_unfinished_submitting_intent_is_distinct_from_sl_tp_block(self):
        self.add_intent("SUBMITTING", "PRE_SEND_BOUNDARY")
        report = inspect_execution(self.engine)
        self.assertEqual(report.sqlite_status, "BLOCKED")
        self.assertEqual(report.latest_intent.state.value, "SUBMITTING")
        self.assertEqual(report.protection_status, "UNVERIFIED")
        self.assertGreaterEqual(len(report.missing_protection_evidence), 4)

    def test_rejected_intent_still_blocks_first_trade_preflight(self):
        self.add_intent("REJECTED", "LOCAL_PLAN_REJECTED")
        report = inspect_execution(self.engine)
        self.assertEqual(report.sqlite_status, "BLOCKED")
        self.assertEqual(report.intent_count, 1)
        self.assertEqual(report.latest_intent.client_order_id, CLIENT_ID)
        self.assertIn("прежнее намерение", report.sqlite_reason.lower())
        self.assertIn("--first-demo-trade", report.latest_intent.flags[0])
        self.assertEqual(report.protection_status, "UNVERIFIED")

    def test_corrupt_or_missing_database_fails_closed_without_creating_it(self):
        self.path.unlink()
        self.assertEqual(inspect_execution(self.engine).sqlite_status, "UNKNOWN")
        self.assertFalse(self.path.exists())
        self.path.write_bytes(b"not sqlite")
        self.assertEqual(inspect_execution(self.engine).sqlite_status, "UNKNOWN")

    def test_untrusted_sqlite_reason_is_never_printed(self):
        self.engine.emergency_halt("MANUAL_HALT")
        with closing(sqlite3.connect(self.path)) as db:
            db.execute("UPDATE execution_control SET reason_code=? WHERE id=1",
                       ("secret-demo-passphrase-123",))
            db.commit()
        output = StringIO()
        with (patch.object(main, "ExecutionEngine", return_value=self.engine),
              redirect_stdout(output)):
            self.assertEqual(main.main(["--execution-diagnose"]), 1)
        self.assertIn("SQLite: UNKNOWN", output.getvalue())
        self.assertNotIn("secret-demo-passphrase-123", output.getvalue())

    def test_cli_is_offline_read_only_and_reports_both_independent_blocks(self):
        self.add_intent("UNKNOWN", "NETWORK_TIMEOUT")
        before = self.path.read_bytes()
        output = StringIO()
        with (patch.object(main, "ExecutionEngine", return_value=self.engine),
              patch.object(main.DemoClient, "from_env_file", side_effect=AssertionError("OKX called")),
              patch.object(urllib.request, "urlopen", side_effect=AssertionError("HTTP called")),
              redirect_stdout(output)):
            self.assertEqual(main.main(["--execution-diagnose"]), 1)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertIn("SQLite: BLOCKED", output.getvalue())
        self.assertIn("Защита частичных BUY: UNVERIFIED", output.getvalue())
        self.assertIn(CLIENT_ID, output.getvalue())
        self.assertIn("NETWORK_TIMEOUT", output.getvalue())


if __name__ == "__main__":
    unittest.main()
