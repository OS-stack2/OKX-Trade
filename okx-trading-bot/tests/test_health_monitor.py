"""Offline Health Monitor checks using only disposable SQLite and fake alerts."""

from __future__ import annotations

from contextlib import closing, redirect_stdout
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from io import StringIO
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import main
from bot.health_monitor import (HealthError, HealthMonitor, HealthSettings, HealthStore,
                                health_settings_from_env)
from bot.watchdog import WatchdogSnapshot, WatchdogStore


NOW = datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc)
SNAPSHOT = WatchdogSnapshot(Decimal("1"), Decimal("1"), Decimal("5000"),
                            Decimal("5000"), (), ())


class FakeAlerts:
    def __init__(self, fail=False):
        self.events = []
        self.fail = fail

    def notify(self, event):
        self.events.append(event)
        if self.fail:
            raise RuntimeError("dummy-secret must never leak")
        return True


class HealthMonitorTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = WatchdogStore(self.root / "watchdog.sqlite3")
        self.settings = HealthSettings(self.source.path, self.root / "health.sqlite3",
                                       timedelta(minutes=15))
        self.store = HealthStore(self.settings)
        self.alerts = FakeAlerts()
        self.current = NOW

    def observe(self, at=NOW, status="OBSERVED_UNVERIFIED"):
        self.source.record(at_utc=at, status=status,
                           snapshot=None if status == "API_UNAVAILABLE" else SNAPSHOT,
                           reasons=())

    def run_once(self):
        return HealthMonitor(self.store, alerter_loader=lambda: self.alerts,
                             clock=lambda: self.current).run_once()

    def state(self):
        with closing(sqlite3.connect(self.settings.health_path)) as db:
            return db.execute("SELECT status,failure_success_id FROM health_state").fetchone()

    def test_fresh_success_is_healthy_but_not_protection_clear(self):
        self.observe()
        result = self.run_once()
        self.assertEqual(result.status, "HEALTHY")
        self.assertEqual(result.exit_code, 0)
        self.assertIn("неизвестны", result.reason)
        self.assertEqual(self.alerts.events, [])
        self.assertTrue(self.settings.health_path.is_file())
        if os.name == "posix":
            self.assertEqual(self.settings.health_path.stat().st_mode & 0o777, 0o600)

    def test_missed_runs_become_stale_and_deduplicate_after_restart(self):
        self.observe()
        self.current += timedelta(minutes=16)
        self.assertEqual(self.run_once().status, "STALE")
        self.assertEqual(HealthMonitor(self.store, alerter_loader=lambda: self.alerts,
                                       clock=lambda: self.current).run_once().status, "STALE")
        self.assertEqual(self.alerts.events, ["HEALTH_STALE"])
        self.assertEqual(self.state(), ("STALE", 1))

    def test_api_error_is_not_healthy_even_if_prior_get_is_fresh(self):
        self.observe()
        self.observe(NOW + timedelta(minutes=1), "API_UNAVAILABLE")
        self.current += timedelta(minutes=2)
        self.assertEqual(self.run_once().status, "API_UNAVAILABLE")
        self.assertEqual(self.alerts.events, ["HEALTH_API_UNAVAILABLE"])

    def test_stale_and_api_recovery_require_new_successful_get(self):
        self.observe()
        self.current += timedelta(minutes=16)
        self.run_once()
        self.observe(NOW + timedelta(minutes=17), "API_UNAVAILABLE")
        self.current = NOW + timedelta(minutes=18)
        self.assertEqual(self.run_once().status, "API_UNAVAILABLE")
        self.observe(NOW + timedelta(minutes=19))
        self.current += timedelta(minutes=2)
        result = self.run_once()
        self.assertEqual(result.status, "HEALTHY")
        self.assertEqual(self.alerts.events,
                         ["HEALTH_STALE", "HEALTH_API_UNAVAILABLE", "HEALTH_RESTORED"])
        self.assertEqual(self.state(), ("HEALTHY", None))
        self.run_once()
        self.assertEqual(len(self.alerts.events), 3)

    def test_clock_rollback_never_claims_recovery(self):
        self.observe()
        self.current += timedelta(minutes=16)
        self.run_once()
        self.current = NOW + timedelta(minutes=1)
        self.assertEqual(self.run_once().status, "UNKNOWN")
        self.assertEqual(self.alerts.events, ["HEALTH_STALE"])
        self.current = NOW + timedelta(minutes=17)
        self.assertEqual(self.run_once().status, "STALE")
        self.assertEqual(self.alerts.events, ["HEALTH_STALE"])
        self.observe(NOW + timedelta(minutes=18))
        self.current = NOW + timedelta(minutes=19)
        self.assertEqual(self.run_once().status, "HEALTHY")
        self.assertEqual(self.alerts.events[-1], "HEALTH_RESTORED")

    def test_future_watchdog_timestamp_is_unknown(self):
        self.observe(NOW + timedelta(hours=1))
        self.assertEqual(self.run_once().status, "UNKNOWN")
        self.assertEqual(self.alerts.events, [])

    def test_watchdog_clock_rollback_is_unknown(self):
        self.observe(NOW)
        self.observe(NOW - timedelta(minutes=1))
        self.assertEqual(self.run_once().status, "UNKNOWN")
        self.assertEqual(self.alerts.events, [])

    def test_missing_and_corrupt_source_fail_closed_without_creating_it(self):
        self.assertEqual(self.run_once().status, "UNKNOWN")
        self.assertFalse(self.source.path.exists())
        self.source.path.write_bytes(b"not SQLite")
        self.assertEqual(self.run_once().status, "UNKNOWN")
        self.assertEqual(self.alerts.events, [])

    def test_corrupt_health_state_fails_closed_without_alert(self):
        self.observe()
        self.settings.health_path.write_bytes(b"not SQLite")
        result = self.run_once()
        self.assertEqual(result.status, "UNKNOWN")
        self.assertFalse(result.state_saved)
        self.assertEqual(self.alerts.events, [])

    def test_lost_health_state_does_not_reset_deduplication(self):
        self.observe()
        self.current += timedelta(minutes=16)
        self.run_once()
        with closing(sqlite3.connect(self.settings.health_path)) as db:
            db.execute("DELETE FROM health_state")
            db.commit()
        result = self.run_once()
        self.assertEqual(result.status, "UNKNOWN")
        self.assertFalse(result.state_saved)
        self.assertEqual(self.alerts.events, ["HEALTH_STALE"])

    def test_telegram_failure_does_not_undo_state_or_retry(self):
        self.observe()
        self.alerts.fail = True
        self.current += timedelta(minutes=16)
        result = self.run_once()
        self.assertEqual(result.status, "STALE")
        self.assertEqual(self.state(), ("STALE", 1))
        self.assertNotIn("dummy-secret", result.notification)
        self.assertEqual(self.run_once().status, "STALE")
        self.assertEqual(self.alerts.events, ["HEALTH_STALE"])
        with closing(sqlite3.connect(self.settings.health_path)) as db:
            self.assertEqual(db.execute("SELECT delivered FROM health_alerts").fetchone(), (0,))

    def test_invalid_configuration_and_paths_fail_closed(self):
        env = self.root / ".env"
        env.write_text(f"OKX_MODE=DEMO\nOKX_WATCHDOG_STATE_PATH={self.source.path}\n"
                       "OKX_HEALTH_STALE_MINUTES=abc\n", encoding="utf-8")
        if os.name == "posix":
            env.chmod(0o600)
        with self.assertRaises(HealthError):
            health_settings_from_env(env)
        env.write_text(f"OKX_MODE=DEMO\nOKX_WATCHDOG_STATE_PATH={self.source.path}\n",
                       encoding="utf-8")
        self.assertEqual(health_settings_from_env(env).stale_after, timedelta(minutes=15))
        with self.assertRaises(HealthError):
            HealthStore(HealthSettings(self.source.path, self.source.path, timedelta(minutes=15)))

    def test_health_does_not_modify_watchdog_or_trading_database(self):
        self.observe()
        working = self.root / "dry_run_journal.sqlite3"
        working.write_bytes(b"untouched trading database")
        before = self.source.path.read_bytes()
        self.run_once()
        self.assertEqual(self.source.path.read_bytes(), before)
        self.assertEqual(working.read_bytes(), b"untouched trading database")
        with patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")):
            self.assertEqual(self.run_once().status, "HEALTHY")
        self.assertNotIn("HALTED", self.settings.health_path.read_bytes().decode("latin1"))

    def test_cli_reports_status_without_api_or_secret(self):
        self.observe()
        with patch("main.health_settings_from_env", return_value=self.settings), \
             patch("main.HealthMonitor", side_effect=lambda store: HealthMonitor(
                 store, alerter_loader=lambda: None, clock=lambda: NOW)), \
             redirect_stdout(StringIO()) as output:
            self.assertEqual(main.main(["--health-once"]), 0)
        self.assertIn("HEALTHY", output.getvalue())
        self.assertNotIn("dummy-secret", output.getvalue())

    def test_systemd_units_are_separate_and_bounded(self):
        units = Path(__file__).resolve().parents[1] / "deploy" / "systemd"
        service = (units / "okx-health.service").read_text(encoding="utf-8")
        timer = (units / "okx-health.timer").read_text(encoding="utf-8")
        for required in ("User=okx-watchdog", "--health-once", "ProtectSystem=strict",
                         "StateDirectory=okx-watchdog", "MemoryMax=64M", "TimeoutStartSec=30s"):
            self.assertIn(required, service)
        self.assertIn("OnCalendar=*-*-* *:2/5:00", timer)
        self.assertNotIn("WantedBy=", service)
        self.assertNotIn("--watchdog-once", service)


if __name__ == "__main__":
    unittest.main()
