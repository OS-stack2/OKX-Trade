"""Independent public quote checks; no private API, trading SQLite or POST."""

from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from io import StringIO
import unittest
from unittest.mock import patch

import main
from bot.market_data import assess_public_quote, get_public_quote_evidence
from config import Config


NOW = datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc)
INSTRUMENT = {"instType": "SPOT", "instId": "BTC-USDT", "state": "live",
              "baseCcy": "BTC", "quoteCcy": "USDT", "minSz": "0.00001",
              "lotSz": "0.00000001", "tickSz": "0.1", "maxLmtSz": "100",
              "maxMktSz": "1000000"}


class FakePublic:
    def __init__(self, *, instrument=None, ticker=None, fail_instrument=False, fail_ticker=False):
        self.instrument = INSTRUMENT if instrument is None else instrument
        self.ticker = ({"instId": "BTC-USDT", "last": "80000", "ts": str(int(NOW.timestamp() * 1000))}
                       if ticker is None else ticker)
        self.fail_instrument = fail_instrument
        self.fail_ticker = fail_ticker
        self.calls = []

    def get_spot_instrument(self):
        self.calls.append("GET instrument")
        if self.fail_instrument:
            raise TimeoutError("secret-bearing HTTP text")
        return [self.instrument]

    def get_ticker(self):
        self.calls.append("GET ticker")
        if self.fail_ticker:
            raise TimeoutError("secret-bearing HTTP text")
        return [self.ticker]

    def get_candles(self, *args, **kwargs):
        raise AssertionError("reconciliation must not request candles")


class MarketDiagnosticTests(unittest.TestCase):
    def test_valid_public_quote_needs_only_instrument_and_ticker(self):
        client = FakePublic()
        assessment = assess_public_quote(get_public_quote_evidence(client), NOW)
        self.assertEqual(client.calls, ["GET instrument", "GET ticker"])
        self.assertEqual(dict(assessment.checks), {"PRICE": "OK", "QUOTE_TIME": "OK",
                                                   "INSTRUMENT": "OK", "LIMITS": "OK"})
        self.assertEqual(assessment.price, Decimal("80000"))
        self.assertEqual(assessment.instrument.max_market_size, Decimal("1000000"))
        self.assertFalse(assessment.reasons)

    def test_failed_endpoints_are_independent_and_do_not_echo_transport_secrets(self):
        client = FakePublic(fail_instrument=True)
        assessment = assess_public_quote(get_public_quote_evidence(client), NOW)
        self.assertEqual(client.calls, ["GET instrument", "GET ticker"])
        self.assertEqual(dict(assessment.checks)["PRICE"], "OK")
        self.assertEqual(dict(assessment.checks)["INSTRUMENT"], "BLOCKED")
        self.assertNotIn("secret-bearing", " ".join(assessment.reasons))
        client = FakePublic(fail_ticker=True)
        assessment = assess_public_quote(get_public_quote_evidence(client), NOW)
        self.assertEqual(dict(assessment.checks)["INSTRUMENT"], "OK")
        self.assertEqual(dict(assessment.checks)["PRICE"], "BLOCKED")
        self.assertNotIn("secret-bearing", " ".join(assessment.reasons))

    def test_stale_future_and_missing_ticker_time_are_separate_from_price(self):
        for timestamp, message in ((NOW - timedelta(seconds=61), "старше 60"),
                                   (NOW + timedelta(seconds=1), "будущем"),
                                   (None, "ticker.ts")):
            with self.subTest(timestamp=timestamp):
                ticker = {"instId": "BTC-USDT", "last": "80000"}
                if timestamp is not None:
                    ticker["ts"] = str(int(timestamp.timestamp() * 1000))
                assessment = assess_public_quote(get_public_quote_evidence(FakePublic(ticker=ticker)), NOW)
                self.assertEqual(dict(assessment.checks)["PRICE"], "OK")
                self.assertEqual(dict(assessment.checks)["QUOTE_TIME"], "BLOCKED")
                self.assertIsNone(assessment.price)
                self.assertTrue(any(message in reason for reason in assessment.reasons))

    def test_missing_or_invalid_instrument_limit_names_the_field(self):
        for field, value in (("maxMktSz", None), ("lotSz", "NaN"),
                             ("maxLmtSz", "0"), ("tickSz", "bad")):
            with self.subTest(field=field):
                instrument = dict(INSTRUMENT)
                if value is None:
                    del instrument[field]
                else:
                    instrument[field] = value
                assessment = assess_public_quote(
                    get_public_quote_evidence(FakePublic(instrument=instrument)), NOW)
                self.assertEqual(dict(assessment.checks)["LIMITS"], "BLOCKED")
                self.assertTrue(any(field in reason for reason in assessment.reasons))
                self.assertIsNone(assessment.instrument)

    def test_market_diagnose_cli_never_reads_demo_or_sqlite(self):
        output = StringIO()
        client = FakePublic(ticker={"instId": "BTC-USDT", "last": "80000", "ts": "1"})
        with (patch.object(main.Config, "from_environment", return_value=Config()),
              patch.object(main, "OkxClient", return_value=client),
              patch.object(main.DemoClient, "from_env_file", side_effect=AssertionError("private GET")),
              redirect_stdout(output)):
            self.assertEqual(main.main(["--market-diagnose"]), 1)
        self.assertEqual(client.calls, ["GET instrument", "GET ticker"])
        self.assertIn("QUOTE_TIME: BLOCKED", output.getvalue())
        self.assertIn("PRICE: OK", output.getvalue())
        self.assertIn("maxMktSz=1000000", output.getvalue())


if __name__ == "__main__":
    unittest.main()
