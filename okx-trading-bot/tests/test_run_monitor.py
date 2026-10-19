from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from bot.run_monitor import BOTS, RunMonitor, check_stale


NOW = datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc)


class RunMonitorTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.monitor = RunMonitor(Path(temp.name) / "m.sqlite3")

    def test_third_consecutive_failure_alerts_once_then_recovery(self):
        self.assertIsNone(self.monitor.record("demo-trader", True, NOW))
        self.assertIsNone(self.monitor.record("demo-trader", False, NOW, "e1"))
        self.assertIsNone(self.monitor.record("demo-trader", False, NOW, "e2"))
        alert = self.monitor.record("demo-trader", False, NOW, "Неверное время запроса")
        self.assertIn("3 неудачных запусков", alert)
        self.assertIn("Неверное время", alert)
        self.assertIsNone(self.monitor.record("demo-trader", False, NOW, "e4"))
        self.assertIn("снова работает", self.monitor.record("demo-trader", True, NOW))
        self.assertIsNone(self.monitor.record("demo-trader", True, NOW))

    def test_success_resets_the_failure_count(self):
        for _ in range(2):
            self.monitor.record("demo-trend", False, NOW, "x")
        self.monitor.record("demo-trend", True, NOW)
        for _ in range(2):
            self.assertIsNone(self.monitor.record("demo-trend", False, NOW, "x"))

    def test_stale_bot_alerts_once_only_when_timer_enabled(self):
        self.monitor.record("demo-trader", True, NOW)
        self.monitor.record("demo-trend", True, NOW)
        later = NOW + timedelta(hours=1)
        sent = []
        enabled = lambda timer: timer == "okx-demo-trader.timer"
        messages = check_stale(self.monitor, later, sent.append, enabled)
        self.assertEqual(len(messages), 1)
        self.assertIn("EMA Demo-трейдер", sent[0])
        self.assertEqual(check_stale(self.monitor, later, sent.append, enabled), [])
        self.assertIn("снова работает", self.monitor.record("demo-trader", True, later))

    def test_never_seen_bot_starts_the_clock_instead_of_alerting(self):
        self.assertEqual(check_stale(self.monitor, NOW, None, lambda _t: True), [])
        self.assertEqual(len(check_stale(self.monitor, NOW + timedelta(hours=4), None, lambda _t: True)), len(BOTS))


if __name__ == "__main__":
    unittest.main()
