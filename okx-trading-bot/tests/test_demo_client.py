"""Offline tests for the private, Demo-only, GET-only account reader."""

from __future__ import annotations

import base64
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import hmac
from io import BytesIO, StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

import main
from bot.demo_client import CurrencyBalance, DemoClient, DemoCredentials, DemoError, DemoSummary
from config import Config, ConfigError


FIXED_TIME = datetime(2024, 1, 2, 3, 4, 5, 678000, tzinfo=timezone.utc)


def fake_response(request, timeout):
    assert timeout == 5
    assert request.get_method() == "GET"
    parsed = urlsplit(request.full_url)
    assert parsed.netloc == "eea.okx.com"
    if parsed.path == "/api/v5/account/balance":
        assert parse_qs(parsed.query) == {"ccy": ["BTC,USDT"]}
        data = [{"details": [
            {"ccy": "BTC", "cashBal": "0.25", "availBal": "0.2"},
            {"ccy": "USDT", "cashBal": "1000", "availBal": "950"},
        ]}]
    elif parsed.path == "/api/v5/trade/orders-pending":
        assert parse_qs(parsed.query)["instType"] == ["SPOT"]
        data = [{"instType": "SPOT", "instId": "BTC-USDT", "ordId": "order-1", "state": "live"}]
    elif parsed.path == "/api/v5/trade/fills-history":
        assert parse_qs(parsed.query)["instType"] == ["SPOT"]
        data = [{"instType": "SPOT", "instId": "BTC-USDT", "tradeId": "fill-1"}]
    else:
        raise AssertionError("Unexpected endpoint")
    return {"code": "0", "msg": "", "data": data}


class DemoClientTests(unittest.TestCase):
    def client(self, transport=fake_response, **kwargs):
        return DemoClient(
            Config(region="eea", mode="DEMO"),
            DemoCredentials("sample-key", "sample-secret", "sample-passphrase"),
            transport=transport,
            clock=lambda: FIXED_TIME,
            **kwargs,
        )

    def test_signature_is_hmac_of_timestamp_method_and_exact_encoded_path(self):
        requests = []

        def transport(request, timeout):
            requests.append(request)
            return fake_response(request, timeout)

        summary = self.client(transport).check_account()
        self.assertEqual(summary.btc.available, Decimal("0.2"))
        self.assertEqual(summary.usdt.available, Decimal("950"))
        self.assertEqual((summary.open_order_count, summary.recent_fill_count), (1, 1))
        self.assertEqual(len(requests), 3)
        for request in requests:
            headers = {name.lower(): value for name, value in request.header_items()}
            self.assertEqual(headers["x-simulated-trading"], "1")
            self.assertEqual(headers["ok-access-key"], "sample-key")
            self.assertEqual(headers["ok-access-passphrase"], "sample-passphrase")
            self.assertEqual(headers["ok-access-timestamp"], "2024-01-02T03:04:05.678Z")
            parsed = urlsplit(request.full_url)
            path = parsed.path + ("?" + parsed.query if parsed.query else "")
            expected = base64.b64encode(hmac.new(
                b"sample-secret", ("2024-01-02T03:04:05.678ZGET" + path).encode(), hashlib.sha256
            ).digest()).decode()
            self.assertEqual(headers["ok-access-sign"], expected)
        self.assertIn("BTC%2CUSDT", requests[0].full_url)

    def test_credentials_only_from_local_env_and_no_live_fallback(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            with self.assertRaisesRegex(ConfigError, "Нет локального .env"):
                DemoClient.from_env_file(path)
            path.write_text(
                "OKX_REGION=eea\nOKX_MODE=DEMO\n"
                "OKX_DEMO_API_KEY=file-key\nOKX_DEMO_API_SECRET=file-secret\n"
                "OKX_DEMO_PASSPHRASE=file-passphrase\n", encoding="utf-8"
            )
            with patch.dict("os.environ", {
                "OKX_MODE": "LIVE", "OKX_REGION": "us", "OKX_DEMO_API_KEY": "environment-key",
                "OKX_DEMO_API_SECRET": "environment-secret", "OKX_DEMO_PASSPHRASE": "environment-passphrase",
            }):
                client = DemoClient.from_env_file(path)
            self.assertEqual(client._config.base_url, "https://eea.okx.com")
            self.assertEqual(client._credentials.api_key, "file-key")
            self.assertNotIn("file-key", repr(client._credentials))
            self.assertNotIn("file-secret", repr(client._credentials))
            path.write_text(path.read_text().replace("OKX_MODE=DEMO", "OKX_MODE=LIVE"), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "LIVE запрещён"):
                DemoClient.from_env_file(path)
            path.write_text(path.read_text().replace("OKX_MODE=LIVE", "OKX_MODE=DEMO").replace("file-passphrase", ""), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "нужны Demo API"):
                DemoClient.from_env_file(path)

    def test_demo_mode_is_required_even_with_credentials(self):
        with self.assertRaisesRegex(ConfigError, "DEMO mode"):
            DemoClient(Config(), DemoCredentials("k", "s", "p"))
        with self.assertRaisesRegex(ConfigError, "LIVE is forbidden"):
            Config(mode="LIVE")

    def test_absent_virtual_usdt_is_zero_and_cli_explains_it(self):
        def transport(request, timeout):
            result = fake_response(request, timeout)
            if urlsplit(request.full_url).path.endswith("/balance"):
                result["data"][0]["details"] = [result["data"][0]["details"][0]]
            return result

        summary = self.client(transport).check_account()
        self.assertEqual(summary.usdt, CurrencyBalance(Decimal("0"), Decimal("0")))
        output = StringIO()
        with patch.object(main.DemoClient, "from_env_file", return_value=self.client(transport)), redirect_stdout(output):
            self.assertEqual(main.main(["--demo-check"]), 0)
        self.assertIn("виртуальных USDT", output.getvalue())
        self.assertIn("Demo Trading", output.getvalue())
        self.assertNotIn("sample-secret", output.getvalue())

    def test_auth_errors_are_safe_and_stop_without_fallback(self):
        calls = []

        def transport(request, timeout):
            calls.append(request.full_url)
            return {"code": "50113", "msg": "sample-secret", "data": []}

        with self.assertRaisesRegex(DemoError, "код 50113") as caught:
            self.client(transport).get_balances()
        self.assertNotIn("sample-secret", str(caught.exception))
        self.assertEqual(len(calls), 1)
        self.assertTrue(all(urlsplit(url).netloc == "eea.okx.com" for url in calls))

    def test_http_permission_error_is_safe_and_closes_response(self):
        body = BytesIO(b"sample-secret")

        def transport(request, timeout):
            raise HTTPError(request.full_url, 403, "sample-secret", None, body)

        with self.assertRaises(DemoError) as caught:
            self.client(transport).get_balances()
        self.assertTrue(body.closed)
        self.assertNotIn("sample-secret", str(caught.exception))

    def test_rate_limit_and_timeout_retry_with_fresh_signature(self):
        calls = []
        delays = []
        moments = iter(FIXED_TIME + timedelta(seconds=offset) for offset in range(3))

        def transport(request, timeout):
            calls.append(request)
            if len(calls) == 1:
                raise TimeoutError()
            if len(calls) == 2:
                return {"code": "50011", "msg": "limited", "data": []}
            return fake_response(request, timeout)

        client = DemoClient(
            Config(region="eea", mode="DEMO", max_attempts=3),
            DemoCredentials("sample-key", "sample-secret", "sample-passphrase"),
            transport=transport, sleeper=delays.append, clock=lambda: next(moments),
        )
        self.assertEqual(client.get_balances()[1].total, Decimal("1000"))
        self.assertEqual(len(calls), 3)
        self.assertEqual(delays, [0.5, 1.0])
        self.assertTrue(all(dict(request.header_items()).get("X-simulated-trading") == "1" for request in calls))
        self.assertEqual(len({dict(request.header_items())["Ok-access-sign"] for request in calls}), 3)

    def test_malformed_and_contradictory_balances_fail(self):
        bad_details = [
            [{"ccy": "USDT", "cashBal": "1", "availBal": "2"}],
            [{"ccy": "BTC", "cashBal": "NaN", "availBal": "0"}],
            [{"ccy": "BTC", "cashBal": "1", "availBal": "1"}, {"ccy": "BTC", "cashBal": "1", "availBal": "1"}],
            [{"ccy": "USDT", "cashBal": "1"}],
        ]
        for details in bad_details:
            with self.subTest(details=details):
                def transport(request, timeout):
                    return {"code": "0", "data": [{"details": details}]}
                with self.assertRaises(DemoError):
                    self.client(transport).get_balances()

    def test_malformed_envelope_and_unexpected_spot_data_fail(self):
        with self.assertRaisesRegex(DemoError, "структуру ответа"):
            self.client(lambda request, timeout: {"data": []}).get_balances()
        def bad_order(request, timeout):
            return {"code": "0", "data": [{"instType": "SWAP", "instId": "BTC-USDT", "ordId": "1", "state": "live"}]}
        with self.assertRaisesRegex(DemoError, "открытый ордер"):
            self.client(bad_order).get_open_orders()
        with self.assertRaisesRegex(DemoError, "историю исполнений"):
            self.client(bad_order).get_fills_history()

    def test_open_orders_use_read_only_pagination(self):
        cursors = []

        def transport(request, timeout):
            params = parse_qs(urlsplit(request.full_url).query)
            cursors.append(params.get("after", [None])[0])
            first = cursors[-1] is None
            numbers = range(100) if first else range(100, 101)
            return {"code": "0", "data": [
                {"instType": "SPOT", "instId": "BTC-USDT", "ordId": str(number), "state": "live"}
                for number in numbers
            ]}

        self.assertEqual(len(self.client(transport).get_open_orders()), 101)
        self.assertEqual(cursors, [None, "99"])

    def test_clock_and_authentication_error_fail_closed(self):
        client = DemoClient(
            Config(region="eea", mode="DEMO"), DemoCredentials("k", "s", "p"),
            transport=lambda request, timeout: self.fail("network must not be used"),
            clock=lambda: datetime(2024, 1, 2),
        )
        with self.assertRaisesRegex(DemoError, "время UTC"):
            client.get_balances()
        with self.assertRaisesRegex(DemoError, "Синхронизируйте|синхронизируйте"):
            self.client(lambda request, timeout: {"code": "50102", "data": []}).get_balances()

    def test_private_order_and_transfer_paths_are_blocked_before_network(self):
        client = self.client(lambda request, timeout: self.fail("network must not be used"))
        for path in ("/api/v5/trade/order", "/api/v5/trade/cancel-order", "/api/v5/asset/transfer"):
            with self.subTest(path=path), self.assertRaisesRegex(DemoError, "Запрещённый"):
                client._get(path, {})
        for name in ("place_order", "amend_order", "cancel_order", "transfer"):
            self.assertFalse(hasattr(client, name))

    def test_cli_reports_missing_credentials_without_public_request(self):
        error = StringIO()
        with patch.object(main.DemoClient, "from_env_file", side_effect=ConfigError("Нет локального .env")), \
             patch.object(main, "OkxClient", side_effect=AssertionError("public request")), redirect_stderr(error):
            self.assertEqual(main.main(["--demo-check"]), 1)
        self.assertIn("Demo-счёта", error.getvalue())


if __name__ == "__main__":
    unittest.main()
