from __future__ import annotations

from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

import main
from bot.market_data import get_market_snapshot
from bot.okx_client import OkxClient, OkxResponseError, OkxTransportError
from config import Config, ConfigError, demo_request_headers


START_MS = int(datetime(2024, 1, 2, tzinfo=timezone.utc).timestamp() * 1000)


def candle_rows(count: int = 101, newest_closed: bool = False) -> list[list[str]]:
    return [
        [
            str(START_MS - index * 900_000),
            "42000", "42100", "41900", "42050", "1", "42050", "42050",
            "1" if newest_closed or index else "0",
        ]
        for index in range(count)
    ]


def response_for(url: str, timeout: float, rows: list[list[str]] | None = None) -> dict:
    assert timeout == 5
    parsed = urlsplit(url)
    assert parsed.scheme == "https"
    params = parse_qs(parsed.query)
    if parsed.path == "/api/v5/market/candles":
        assert params == {"instId": ["BTC-USDT"], "bar": ["15m"], "limit": ["101"]}
        data = rows if rows is not None else candle_rows()
    elif parsed.path == "/api/v5/market/ticker":
        assert params == {"instId": ["BTC-USDT"]}
        data = [{"instId": "BTC-USDT", "last": "42060.5"}]
    elif parsed.path == "/api/v5/public/instruments":
        assert params == {"instType": ["SPOT"], "instId": ["BTC-USDT"]}
        data = [{"instId": "BTC-USDT", "instType": "SPOT", "tickSz": "0.1", "lotSz": "0.00001", "minSz": "0.00001"}]
    else:
        raise AssertionError(f"Unexpected endpoint: {parsed.path}")
    return {"code": "0", "msg": "", "data": data}


class ReadOnlyTests(unittest.TestCase):
    def test_cli_prints_100_closed_candles_and_current_price(self) -> None:
        client = OkxClient(Config(), transport=response_for)
        output = StringIO()
        with patch.object(main, "OkxClient", return_value=client), redirect_stdout(output):
            result = main.main(["--check"])
        self.assertEqual(result, 0)
        self.assertIn("BTC-USDT", output.getvalue())
        self.assertIn("42060.5 USDT", output.getvalue())
        self.assertIn("2024-01-02 00:00:00 UTC", output.getvalue())
        self.assertIn("Количество закрытых свечей: 100", output.getvalue())

    def test_unclosed_candle_is_excluded_even_if_price_is_valid(self) -> None:
        client = OkxClient(Config(), transport=response_for)
        snapshot = get_market_snapshot(client)
        self.assertEqual(len(snapshot.candles), 100)
        self.assertEqual(snapshot.latest_closed_candle.opened_at, datetime(2024, 1, 1, 23, 45, tzinfo=timezone.utc))

    def test_insufficient_closed_candles_fails(self) -> None:
        def transport(url: str, timeout: float) -> dict:
            return response_for(url, timeout, candle_rows(100))

        with self.assertRaisesRegex(OkxResponseError, "Only 99 closed candles"):
            get_market_snapshot(OkxClient(Config(), transport=transport))

    def test_invalid_confirmation_flag_fails(self) -> None:
        rows = candle_rows()
        rows[3][8] = "unknown"
        def transport(url: str, timeout: float) -> dict:
            return response_for(url, timeout, rows)

        with self.assertRaisesRegex(OkxResponseError, "confirmation flag"):
            get_market_snapshot(OkxClient(Config(), transport=transport))

    def test_okx_error_and_malformed_envelope_fail(self) -> None:
        for payload, message in [
            ({"code": "51000", "msg": "bad request", "data": []}, "OKX error 51000"),
            ({"code": "0", "data": "wrong"}, "no data list"),
            ({"data": []}, "Malformed OKX response envelope"),
        ]:
            with self.subTest(payload=payload):
                client = OkxClient(Config(), transport=lambda _url, _timeout: payload)
                with self.assertRaisesRegex(OkxResponseError, message):
                    client.get_ticker()

    def test_timeout_and_rate_limit_are_retried_with_limit(self) -> None:
        attempts = []
        delays = []
        def transport(url: str, timeout: float) -> dict:
            attempts.append(url)
            if len(attempts) == 1:
                raise TimeoutError()
            if len(attempts) == 2:
                raise HTTPError(url, 429, "rate limited", None, None)
            return response_for(url, timeout)

        client = OkxClient(Config(), transport=transport, sleeper=delays.append)
        self.assertEqual(client.get_ticker()[0]["last"], "42060.5")
        self.assertEqual(len(attempts), 3)
        self.assertEqual(delays, [0.5, 1.0])

        # A transport that raises on both attempts must stop rather than loop forever.
        def always_timeout(_url: str, _timeout: float) -> dict:
            raise TimeoutError()
        failing = OkxClient(Config(max_attempts=2), transport=always_timeout, sleeper=lambda _: None)
        with self.assertRaisesRegex(OkxTransportError, "attempts exhausted"):
            failing.get_ticker()

    def test_demo_guard_and_live_rejection(self) -> None:
        with self.assertRaisesRegex(ConfigError, "LIVE is forbidden"):
            Config(mode="LIVE")
        with self.assertRaisesRegex(ConfigError, "Private requests require DEMO"):
            demo_request_headers(Config())
        self.assertEqual(demo_request_headers(Config(mode="DEMO")), {"x-simulated-trading": "1"})

    def test_region_and_env_file_do_not_fall_back_to_another_domain(self) -> None:
        self.assertEqual(Config(region="eea").base_url, "https://eea.okx.com")
        self.assertEqual(Config(region="us").base_url, "https://us.okx.com")
        with self.assertRaises(ConfigError):
            Config(region="unknown")
        with TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("OKX_REGION=us\nOKX_MODE=READ_ONLY\n", encoding="utf-8")
            self.assertEqual(Config.from_environment(path, {}).base_url, "https://us.okx.com")
            self.assertEqual(Config.from_environment(path, {"OKX_REGION": "eea"}).base_url, "https://eea.okx.com")


if __name__ == "__main__":
    unittest.main()
