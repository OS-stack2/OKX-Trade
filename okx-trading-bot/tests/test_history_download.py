"""Public OKX history pagination and CSV safety, with no network or trading API."""

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit

import main
from bot.growth_research import load_candles_csv
from bot.history_download import HistoryError, download_history, save_history_csv
from bot.okx_client import OkxClient, OkxResponseError, OkxTransportError
from config import Config


NOW = datetime(2024, 3, 1, 0, 7, tzinfo=timezone.utc)
OPEN_MS = int(datetime(2024, 3, 1, tzinfo=timezone.utc).timestamp() * 1000)


def rows(count=120, *, open_candle=True):
    answer = []
    for index in range(count + int(open_candle)):
        timestamp = OPEN_MS - index * 900_000
        answer.append([str(timestamp), "42000", "42100", "41900", "42050",
                       "1.5", "63075", "63075", "0" if index == 0 and open_candle else "1"])
    return answer


def client_for(series, *, fail=None, calls=None):
    def transport(url, timeout):
        assert timeout == 5
        parsed = urlsplit(url)
        assert parsed.path == "/api/v5/market/history-candles"
        params = parse_qs(parsed.query)
        assert params["instId"] == ["BTC-USDT"] and params["bar"] == ["15m"]
        assert params["limit"] == ["100"]
        if calls is not None:
            calls.append(params)
        if fail is not None:
            return fail
        cursor = int(params["after"][0]) if "after" in params else None
        eligible = [row for row in series if cursor is None or int(row[0]) < cursor]
        return {"code": "0", "data": eligible[:100]}
    return OkxClient(Config(), transport=transport, sleeper=lambda _: None)


class HistoryDownloadTests(unittest.TestCase):
    def test_closed_paginated_csv_matches_growth_loader_and_does_not_overwrite(self):
        calls = []
        report = download_history(client_for(rows(), calls=calls), bars=120, now=NOW, sleeper=lambda _: None)
        self.assertEqual(len(report.candles), 120)
        self.assertFalse(report.sufficient)
        self.assertEqual(report.split_counts, (72, 24, 24))
        self.assertEqual(len(calls), 2)
        self.assertNotIn("after", calls[0])
        self.assertIn("after", calls[1])
        self.assertEqual(report.candles[-1].closed_at, datetime(2024, 3, 1, tzinfo=timezone.utc))
        with TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            save_history_csv(report, path)
            self.assertEqual(load_candles_csv(path), report.candles)
            original = path.read_bytes()
            with self.assertRaisesRegex(HistoryError, "already exists"):
                save_history_csv(report, path)
            self.assertEqual(path.read_bytes(), original)
            save_history_csv(report, path, overwrite=True)
            self.assertEqual(path.read_bytes(), original)

    def test_unclosed_candle_inside_series_and_future_closed_fail(self):
        bad = rows()
        bad[5][8] = "0"
        with self.assertRaisesRegex(HistoryError, "unconfirmed"):
            download_history(client_for(bad), bars=100, now=NOW, sleeper=lambda _: None)
        bad = rows()
        bad[0][8] = "1"
        with self.assertRaisesRegex(HistoryError, "future candle"):
            download_history(client_for(bad), bars=100, now=NOW, sleeper=lambda _: None)

    def test_gaps_duplicates_page_order_and_stalled_cursor_fail(self):
        bad = rows()
        del bad[55]
        with self.assertRaisesRegex(HistoryError, "Gap"):
            download_history(client_for(bad), bars=120, now=NOW, sleeper=lambda _: None)
        bad = rows()
        bad[55] = bad[54].copy()
        with self.assertRaisesRegex(HistoryError, "Gap|newest-to-oldest"):
            download_history(client_for(bad), bars=120, now=NOW, sleeper=lambda _: None)
        bad = rows()
        bad[51], bad[52] = bad[52], bad[51]
        with self.assertRaisesRegex(HistoryError, "Gap|newest-to-oldest"):
            download_history(client_for(bad), bars=120, now=NOW, sleeper=lambda _: None)
        series = rows()
        def repeating(url, timeout):
            return {"code": "0", "data": series[:100]}
        with self.assertRaisesRegex(HistoryError, "overlap"):
            download_history(OkxClient(Config(), transport=repeating), bars=120, now=NOW, sleeper=lambda _: None)

    def test_invalid_ohlcv_timestamp_flag_and_malformed_record_fail(self):
        defects = ((1, 0, "123", "timestamp"), (1, 2, "1", "OHLC"),
                   (1, 5, "-1", "volume"), (1, 6, "NaN", "volume"),
                   (1, 8, "?", "confirmation"))
        for index, field, value, message in defects:
            with self.subTest(field=field):
                bad = rows()
                bad[index][field] = value
                with self.assertRaisesRegex(HistoryError, message):
                    download_history(client_for(bad), bars=100, now=NOW, sleeper=lambda _: None)
        bad = rows()
        bad[1] = bad[1][:-1]
        with self.assertRaisesRegex(HistoryError, "nine fields"):
            download_history(client_for(bad), bars=100, now=NOW, sleeper=lambda _: None)

    def test_api_error_timeout_and_empty_history_fail_closed(self):
        with self.assertRaisesRegex(OkxResponseError, "OKX error"):
            download_history(client_for(rows(), fail={"code": "51000", "data": [], "msg": "bad"}),
                             bars=100, now=NOW, sleeper=lambda _: None)
        failing = OkxClient(Config(max_attempts=2), transport=lambda *_: (_ for _ in ()).throw(TimeoutError()),
                            sleeper=lambda _: None)
        with self.assertRaisesRegex(OkxTransportError, "attempts exhausted"):
            download_history(failing, bars=100, now=NOW, sleeper=lambda _: None)
        with self.assertRaisesRegex(HistoryError, "Only 0"):
            download_history(client_for([]), bars=100, now=NOW, sleeper=lambda _: None)

    def test_temporary_rate_limit_retries_and_six_month_split_is_sufficient(self):
        series = rows(17_280)
        attempts = []
        def transport(url, timeout):
            params = parse_qs(urlsplit(url).query)
            attempts.append(params)
            if len(attempts) == 1:
                return {"code": "50011", "msg": "rate limited", "data": []}
            cursor = int(params["after"][0]) if "after" in params else None
            eligible = [row for row in series if cursor is None or int(row[0]) < cursor]
            return {"code": "0", "data": eligible[:100]}
        report = download_history(OkxClient(Config(), transport=transport, sleeper=lambda _: None),
                                  bars=17_280, now=NOW, sleeper=lambda _: None)
        self.assertEqual(len(report.candles), 17_280)
        self.assertTrue(report.sufficient)
        self.assertEqual(report.split_counts, (10_368, 3_456, 3_456))
        self.assertEqual(len(attempts), 174)  # One retried rate-limit response.

    def test_cli_never_uses_trading_api_or_overwrites_and_research_costs_are_scenarios(self):
        report = download_history(client_for(rows()), bars=120, now=NOW, sleeper=lambda _: None)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            output = StringIO()
            with patch.object(main, "download_history", return_value=report), \
                 patch.object(main, "OkxClient", return_value=object()), \
                 patch.object(main.Config, "from_environment", return_value=Config()), \
                 redirect_stdout(output):
                code = main.main(["--download-history", "--history-bars", "120", "--history-output", str(path)])
            self.assertEqual(code, 1)  # Valid CSV, but far too short for credible research.
            self.assertTrue(path.exists())
            self.assertIn("НЕДОСТАТОЧНО", output.getvalue())
            error = StringIO()
            with patch.object(main, "download_history", side_effect=AssertionError("must not fetch")), \
                 redirect_stderr(error):
                self.assertEqual(main.main(["--download-history", "--history-output", str(path)]), 1)
            self.assertIn("CSV уже существует", error.getvalue())
            error = StringIO()
            missing = Path(directory) / "missing.csv"
            with patch.object(main, "download_history", side_effect=HistoryError("bad data")), \
                 redirect_stderr(error):
                self.assertEqual(main.main(["--download-history", "--history-output", str(missing)]), 1)
            self.assertFalse(missing.exists())
            output = StringIO()
            with redirect_stdout(output):
                code = main.main(["--growth-research", str(path), "--research-min-sz", "0.0001",
                                  "--research-lot-sz", "0.0001", "--research-tick-sz", "0.1",
                                  "--research-max-mkt-sz", "100"])
            self.assertEqual(code, 0)
            self.assertIn("СЦЕНАРИЙ ВЫСОКИХ ЗАТРАТ", output.getvalue())
            self.assertIn("не подтверждают комиссию", output.getvalue())


if __name__ == "__main__":
    unittest.main()
