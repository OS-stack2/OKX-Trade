"""Offline, separate-journal Watchdog checks; no live credentials or network."""

from __future__ import annotations

from contextlib import closing, redirect_stdout
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
from io import StringIO
import os
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import main
from bot.demo_client import CurrencyBalance, DemoAlgoOrder, DemoClient, DemoCredentials
from bot.okx_client import OkxClient
from bot.watchdog import (Watchdog, WatchdogError, WatchdogSnapshot, WatchdogStore,
                          watchdog_path_from_env)
from config import Config


NOW = datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc)


class FakeDemo(DemoClient):
    def __init__(self):
        self._config = Config(mode="DEMO", region="global")
        self.btc = CurrencyBalance(D("1"), D("1"))
        self.usdt = CurrencyBalance(D("5000"), D("5000"))
        self.orders = []
        self.algos = ()
        self.fail = False
        self.calls = []

    def get_balances(self):
        self.calls.append("balances")
        if self.fail:
            raise TimeoutError("private key and chat id must not leak")
        return self.btc, self.usdt

    def get_open_orders(self):
        self.calls.append("pending SPOT")
        return self.orders

    def get_pending_algo_orders(self):
        self.calls.append("pending Algo")
        return self.algos


class FakePublic:
    def __init__(self):
        self.fail = False
        self.calls = 0

    def get_ticker(self):
        self.calls += 1
        if self.fail:
            raise TimeoutError("public URL must not leak")
        return [{"instId": "BTC-USDT", "last": "80000", "ts": str(int(NOW.timestamp() * 1000))}]


class FakeAlerts:
    def __init__(self):
        self.events = []
        self.fail = False

    def notify(self, event):
        self.events.append(event)
        if self.fail:
            raise RuntimeError("secret-bearing Telegram error")
        return True


class WatchdogTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.store = WatchdogStore(self.root / "watchdog.sqlite3")
        self.demo = FakeDemo()
        self.public = FakePublic()
        self.alerts = FakeAlerts()
        self.runner = Watchdog(store=self.store, demo_loader=lambda: self.demo,
                               public_factory=lambda _config: self.public,
                               alerter_loader=lambda: self.alerts, clock=lambda: NOW)

    def test_first_snapshot_is_observed_but_never_clear_or_protected(self):
        report = self.runner.run_once()
        self.assertEqual(report.status, "OBSERVED_UNVERIFIED")
        self.assertEqual(report.exit_code, 0)
        self.assertIn("Принадлежность BTC", " ".join(report.reasons))
        self.assertEqual(report.snapshot.btc_total, D("1"))
        self.assertEqual(report.snapshot.algo_order_count, 0)
        self.assertEqual(self.alerts.events, [])
        self.assertEqual(self.demo.calls, ["balances", "pending SPOT", "pending Algo"])

    def test_balance_change_alerts_once_and_survives_restart(self):
        self.runner.run_once()
        self.demo.usdt = CurrencyBalance(D("4999"), D("4999"))
        changed = self.runner.run_once()
        self.assertEqual(changed.status, "SUSPICIOUS")
        self.assertTrue(changed.changed)
        self.assertEqual(self.alerts.events, ["WATCHDOG_SUSPICIOUS"])
        restarted = Watchdog(store=WatchdogStore(self.store.path),
                             demo_loader=lambda: self.demo,
                             public_factory=lambda _: self.public,
                             alerter_loader=lambda: self.alerts, clock=lambda: NOW)
        same = restarted.run_once()
        self.assertFalse(same.changed)
        self.assertEqual(len(self.alerts.events), 1)
        with closing(sqlite3.connect(self.store.path)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM watchdog_observations").fetchone()[0], 3)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM watchdog_alerts").fetchone()[0], 1)

    def test_timeout_dedup_and_recovery_notification(self):
        self.demo.fail = True
        first = self.runner.run_once()
        second = self.runner.run_once()
        self.assertEqual(first.status, "API_UNAVAILABLE")
        self.assertEqual(second.status, "API_UNAVAILABLE")
        self.assertEqual(self.alerts.events, ["WATCHDOG_API_DOWN"])
        self.assertNotIn("private key", " ".join(first.reasons))
        self.demo.fail = False
        recovered = Watchdog(store=WatchdogStore(self.store.path),
                             demo_loader=lambda: self.demo,
                             public_factory=lambda _: self.public,
                             alerter_loader=lambda: self.alerts, clock=lambda: NOW).run_once()
        self.assertEqual(recovered.status, "OBSERVED_UNVERIFIED")
        self.assertEqual(self.alerts.events[-1], "WATCHDOG_RESTORED")
        self.demo.fail = True
        self.runner.run_once()
        self.assertEqual(self.alerts.events.count("WATCHDOG_API_DOWN"), 2)

    def test_recovery_compares_last_successful_snapshot_after_outage(self):
        self.runner.run_once()
        self.demo.fail = True
        self.runner.run_once()
        self.demo.fail = False
        self.demo.btc = CurrencyBalance(D("1.01"), D("1.01"))
        report = self.runner.run_once()
        self.assertEqual(report.status, "SUSPICIOUS")
        self.assertTrue(report.changed)
        self.assertEqual(self.alerts.events,
                         ["WATCHDOG_API_DOWN", "WATCHDOG_RESTORED", "WATCHDOG_SUSPICIOUS"])

    def test_public_timeout_and_stale_ticker_are_unavailable(self):
        self.public.fail = True
        self.assertEqual(self.runner.run_once().status, "API_UNAVAILABLE")
        self.public.fail = False
        self.public.get_ticker = lambda: [{"instId": "BTC-USDT", "last": "80000",
                                           "ts": str(int((NOW - timedelta(minutes=10)).timestamp() * 1000))}]
        self.assertEqual(self.runner.run_once().status, "API_UNAVAILABLE")
        self.assertEqual(self.alerts.events, ["WATCHDOG_API_DOWN"])

    def test_non_demo_client_is_rejected_before_any_get(self):
        self.demo._config = Config(mode="READ_ONLY")
        report = self.runner.run_once()
        self.assertEqual(report.status, "API_UNAVAILABLE")
        self.assertEqual(self.demo.calls, [])
        self.assertEqual(self.public.calls, 0)

    def test_open_orders_and_algos_are_suspicious_without_ownership_claim(self):
        self.demo.orders = [{"instType": "SPOT", "instId": "BTC-USDT", "ordId": "order1",
                             "state": "live", "side": "buy", "sz": "0.01", "accFillSz": "0"}]
        self.demo.algos = (DemoAlgoOrder("algo1", "oco", "live", D("0.01"), "sell"),)
        report = self.runner.run_once()
        self.assertEqual(report.status, "SUSPICIOUS")
        self.assertIn("неизвестного происхождения", " ".join(report.reasons))
        self.assertEqual(report.snapshot.spot_order_count, 1)
        self.assertEqual(report.snapshot.algo_order_count, 1)
        self.assertEqual(self.alerts.events, ["WATCHDOG_SUSPICIOUS"])
        with closing(sqlite3.connect(self.store.path)) as db:
            payload = db.execute("SELECT snapshot_json FROM watchdog_state WHERE id=1").fetchone()[0]
        self.assertNotIn("order1", payload)
        self.assertNotIn("algo1", payload)

    def test_telegram_failure_does_not_undo_snapshot_or_repeat_alert(self):
        self.demo.fail = True
        self.alerts.fail = True
        first = self.runner.run_once()
        self.assertTrue(first.journaled)
        self.assertIn("доставка не подтверждена", " ".join(first.notifications))
        self.runner.run_once()
        self.assertEqual(self.alerts.events, ["WATCHDOG_API_DOWN"])
        with closing(sqlite3.connect(self.store.path)) as db:
            self.assertEqual(db.execute("SELECT health FROM watchdog_state WHERE id=1").fetchone()[0], "DOWN")
            self.assertEqual(db.execute("SELECT delivered FROM watchdog_alerts").fetchone()[0], 0)

    def test_bad_balance_or_partial_journal_schema_fail_closed(self):
        self.demo.btc = CurrencyBalance(D("1"), D("2"))
        self.assertEqual(self.runner.run_once().status, "API_UNAVAILABLE")
        with closing(sqlite3.connect(self.store.path)) as db:
            db.execute("DROP TABLE watchdog_alerts")
            db.commit()
        report = self.runner.run_once()
        self.assertEqual(report.status, "JOURNAL_ERROR")
        self.assertFalse(report.journaled)

    def test_missing_persisted_state_cannot_reset_alert_deduplication(self):
        self.runner.run_once()
        with closing(sqlite3.connect(self.store.path)) as db:
            db.execute("DELETE FROM watchdog_state")
            db.commit()
        report = self.runner.run_once()
        self.assertEqual(report.status, "JOURNAL_ERROR")
        self.assertEqual(self.alerts.events, [])

    def test_log_retention_and_private_file_mode(self):
        with patch("bot.watchdog.MAX_OBSERVATIONS", 3):
            for _ in range(6):
                self.runner.run_once()
        with closing(sqlite3.connect(self.store.path)) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM watchdog_observations").fetchone()[0], 3)
        if os.name == "posix":
            self.assertEqual(self.store.path.stat().st_mode & 0o777, 0o600)

    def test_path_guard_rejects_working_journal_symlink_and_hardlink(self):
        main_path = self.root / "dry_run_journal.sqlite3"
        main_path.write_bytes(b"Windows SQLite placeholder - never opened")
        alias = self.root / "watchdog.sqlite3"
        with patch("bot.watchdog.WINDOWS_JOURNAL_PATH", main_path):
            with self.assertRaises(WatchdogError):
                WatchdogStore(main_path)
            # Creating links may require extra privileges on Windows; the
            # direct-path check still runs there.
            try:
                alias.symlink_to(main_path)
            except OSError:
                pass
            else:
                with self.assertRaises(WatchdogError):
                    WatchdogStore(alias)
                alias.unlink()
            try:
                alias.hardlink_to(main_path)
            except OSError:
                pass
            else:
                with self.assertRaises(WatchdogError):
                    WatchdogStore(alias)
        self.assertEqual(main_path.read_bytes(), b"Windows SQLite placeholder - never opened")

    def test_local_state_path_requires_absolute_path(self):
        env = self.root / ".env"
        env.write_text("OKX_MODE=DEMO\nOKX_WATCHDOG_STATE_PATH=relative/watchdog.sqlite3\n",
                       encoding="utf-8")
        if os.name == "posix":
            env.chmod(0o600)
        with self.assertRaises(WatchdogError):
            watchdog_path_from_env(env)
        env.write_text(f"OKX_MODE=DEMO\nOKX_WATCHDOG_STATE_PATH={self.store.path}\n",
                       encoding="utf-8")
        self.assertEqual(watchdog_path_from_env(env), self.store.path)
        if os.name == "posix":
            env.chmod(0o644)
            with self.assertRaises(WatchdogError):
                watchdog_path_from_env(env)

    def test_real_demo_client_transport_is_get_only_with_demo_header(self):
        seen = []

        def private(request, _timeout):
            seen.append((request.get_method(), request.full_url,
                         request.get_header("X-simulated-trading")))
            if "/account/balance" in request.full_url:
                return {"code": "0", "data": [{"details": [
                    {"ccy": "BTC", "cashBal": "1", "availBal": "1"},
                    {"ccy": "USDT", "cashBal": "5000", "availBal": "5000"}]}]}
            return {"code": "0", "data": []}

        demo = DemoClient(Config(mode="DEMO"), DemoCredentials("fake-key", "fake-secret", "fake-pass"),
                          transport=private, clock=lambda: NOW)
        public = OkxClient(demo.config, transport=lambda _url, _timeout:
                           {"code": "0", "data": [{"instId": "BTC-USDT", "last": "80000",
                                                   "ts": str(int(NOW.timestamp() * 1000))}]})
        report = Watchdog(store=self.store, demo_loader=lambda: demo,
                          public_factory=lambda _: public,
                          alerter_loader=lambda: None, clock=lambda: NOW).run_once()
        self.assertEqual(report.status, "OBSERVED_UNVERIFIED")
        self.assertTrue(seen)
        self.assertTrue(all(method == "GET" and header == "1" for method, _, header in seen))
        self.assertTrue(all(url.startswith("https://www.okx.com/api/v5/") for _, url, _ in seen))

    def test_cli_reports_without_secrets_or_main_sqlite(self):
        report = SimpleNamespace(status="OBSERVED_UNVERIFIED", snapshot=None,
                                 reasons=("Принадлежность BTC неизвестна",),
                                 notifications=(), exit_code=0)
        output = StringIO()
        with (patch("main.WatchdogStore") as store,
              patch("main.Watchdog") as runner,
              patch("main.DecisionJournal", side_effect=AssertionError("Windows journal called")),
              redirect_stdout(output)):
            runner.return_value.run_once.return_value = report
            self.assertEqual(main.main(["--watchdog-once"]), 0)
            store.assert_called_once()
        self.assertIn("OBSERVED_UNVERIFIED", output.getvalue())
        self.assertNotIn("CLEAR", output.getvalue())

    def test_systemd_units_are_isolated_and_bounded(self):
        root = Path(__file__).resolve().parents[1] / "deploy" / "systemd"
        service = (root / "okx-watchdog.service").read_text(encoding="utf-8")
        timer = (root / "okx-watchdog.timer").read_text(encoding="utf-8")
        for required in ("User=okx-watchdog", "Type=oneshot", "--watchdog-once",
                         "StateDirectory=okx-watchdog", "ProtectSystem=strict",
                         "UMask=0077", "MemoryMax=128M", "TimeoutStartSec=90s"):
            self.assertIn(required, service)
        self.assertIn("OnCalendar=*-*-* *:0/5:00", timer)
        self.assertNotIn("--recovery-check", service)
