"""Offline one-shot Telegram diagnostic and persisted emergency-state tests."""

from __future__ import annotations

from contextlib import closing, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from io import StringIO
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import main
from bot.telegram_alerts import TEST_MESSAGE, TelegramAlerter


FAKE_TOKEN = "123456:" + "test_token_1234567890ABCD"
FAKE_CHAT = "-123456789"


class TelegramDiagnosticTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.env = self.root / ".env"
        self.env.write_text(
            f"OKX_MODE=DEMO\nOKX_TELEGRAM_BOT_TOKEN={FAKE_TOKEN}\n"
            f"OKX_TELEGRAM_CHAT_ID={FAKE_CHAT}\n", encoding="utf-8")

    def _alerter(self, transport):
        loaded = TelegramAlerter.from_env_file(self.env)
        self.assertIsNotNone(loaded)
        return TelegramAlerter(loaded.token, loaded.chat_id, transport)

    def test_cli_sends_exactly_one_message_without_okx_or_sqlite(self):
        requests = []

        def transport(request, timeout):
            requests.append(request)
            self.assertEqual(timeout, 5.0)
            return {"ok": True}

        alerter = self._alerter(transport)
        sentinel = self.root / "journal.sqlite3"
        sentinel.write_bytes(b"unchanged local file")
        before = sentinel.read_bytes()
        output, errors = StringIO(), StringIO()
        with (patch("main.TelegramAlerter.from_env_file", return_value=alerter),
              patch("main.DemoClient.from_env_file", side_effect=AssertionError("OKX called")),
              redirect_stdout(output), redirect_stderr(errors)):
            result = main.main(["--telegram-test"])
        self.assertEqual(result, 0)
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].get_method(), "POST")
        self.assertEqual(json.loads(requests[0].data.decode("utf-8")),
                         {"chat_id": FAKE_CHAT, "text": TEST_MESSAGE})
        self.assertTrue(requests[0].full_url.startswith("https://api.telegram.org/bot"))
        self.assertEqual(sentinel.read_bytes(), before)
        self.assertEqual(errors.getvalue(), "")
        self.assertNotIn(FAKE_TOKEN, output.getvalue())
        self.assertNotIn(FAKE_CHAT, output.getvalue())

    def test_repeat_attempt_on_one_instance_is_suppressed_even_after_timeout(self):
        requests = []

        def timeout(request, _):
            requests.append(request)
            raise TimeoutError(f"URL contains {FAKE_TOKEN} and {FAKE_CHAT}")

        alerter = self._alerter(timeout)
        self.assertFalse(alerter.notify("TEST"))
        self.assertFalse(alerter.notify("TEST"))
        self.assertEqual(len(requests), 1)
        self.assertNotIn(FAKE_TOKEN, repr(alerter))
        self.assertNotIn(FAKE_CHAT, repr(alerter))

    def test_missing_or_live_configuration_and_transport_error_are_safe(self):
        stdout, stderr = StringIO(), StringIO()
        with (patch("main.TelegramAlerter.from_env_file", return_value=None),
              redirect_stdout(stdout), redirect_stderr(stderr)):
            self.assertEqual(main.main(["--telegram-test"]), 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertNotIn(FAKE_TOKEN, stderr.getvalue())
        self.assertNotIn(FAKE_CHAT, stderr.getvalue())

        self.env.write_text(
            f"OKX_MODE=LIVE\nOKX_TELEGRAM_BOT_TOKEN={FAKE_TOKEN}\n"
            f"OKX_TELEGRAM_CHAT_ID={FAKE_CHAT}\n", encoding="utf-8")
        self.assertIsNone(TelegramAlerter.from_env_file(self.env))

        alerter = TelegramAlerter(FAKE_TOKEN, FAKE_CHAT,
                                  lambda _request, _timeout: (_ for _ in ()).throw(
                                      RuntimeError(FAKE_TOKEN + FAKE_CHAT)))
        stderr = StringIO()
        with (patch("main.TelegramAlerter.from_env_file", return_value=alerter),
              redirect_stderr(stderr)):
            self.assertEqual(main.main(["--telegram-test"]), 1)
        self.assertNotIn(FAKE_TOKEN, stderr.getvalue())
        self.assertNotIn(FAKE_CHAT, stderr.getvalue())
