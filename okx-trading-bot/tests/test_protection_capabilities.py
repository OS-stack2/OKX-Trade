"""Offline GET-only capability inventory; success never enables an order."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from decimal import Decimal as D
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import parse_qs, urlsplit
import unittest
from unittest.mock import patch

import main
from bot.demo_client import DemoClient, DemoCredentials, DemoError, DemoFill, DemoOrder, SpotFeeRates
from bot.execution_engine import ExecutionEngine
from bot.journal import CycleDraft, CycleStatus, DecisionJournal
from bot.okx_client import OkxClient
from bot.protection_capabilities import inspect_protection_capabilities
from bot.risk_manager import RiskManager
from bot.strategy_budget import StrategyBudgetStore
from config import Config, ConfigError


NOW = datetime(2026, 10, 15, 12, 30, tzinfo=timezone.utc)
INSTRUMENT = {"instType": "SPOT", "instId": "BTC-USDT", "state": "live",
              "baseCcy": "BTC", "quoteCcy": "USDT", "minSz": "0.00001",
              "lotSz": "0.00000001", "tickSz": "0.1", "maxLmtSz": "100"}


class CapabilityTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.journal = DecisionJournal(root / "journal.sqlite3")
        self.journal.append(CycleDraft(NOW, NOW, "HOLD", CycleStatus.HOLD, "test"))
        self.engine = ExecutionEngine(journal=self.journal,
            risk_manager=RiskManager(root / "risk.json"),
            budget_store=StrategyBudgetStore(self.journal.path), clock=lambda: NOW)
        self.config = Config(region="eea", mode="DEMO")

    def public(self, row=INSTRUMENT):
        def transport(url, timeout):
            parsed = urlsplit(url)
            self.assertEqual(parsed.netloc, "eea.okx.com")
            self.assertEqual(parsed.path, "/api/v5/public/instruments")
            self.assertEqual(parse_qs(parsed.query), {"instType": ["SPOT"], "instId": ["BTC-USDT"]})
            return {"code": "0", "data": [row]}
        return OkxClient(self.config, transport=transport)

    def demo(self, transport):
        return DemoClient(self.config, DemoCredentials("fake-key", "fake-secret", "fake-phrase"),
                          transport=transport, clock=lambda: NOW)

    def seed_partial_buy(self):
        client_id = "TB" + "a" * 30
        with self.engine._transaction() as db:
            self.engine._schema(db)
            db.execute("""INSERT INTO execution_intents (
                client_order_id,decision_id,candle_closed_at,mode,inst_id,side,state,
                quantity,entry_price,stop_price,take_profit_price,estimated_risk_usdt,
                min_size,lot_size,tick_size,created_at_utc,updated_at_utc,
                exchange_order_id,filled_quantity,fee_btc,fee_usdt,average_fill_price,reason_code
            ) VALUES (?,? ,?,'DEMO','BTC-USDT','BUY','PARTIALLY_FILLED',
                '0.03','80000','79200','81600','25','0.0001','0.0001','0.1',
                ?,?,'buy-order','0.02','0','-1','80000','FILL_OBSERVED')""",
                (client_id, "decision-1", NOW.isoformat(), NOW.isoformat(), NOW.isoformat()))
        return client_id

    def test_public_and_demo_get_observations_remain_blocked_and_read_only(self):
        requests = []
        def transport(request, timeout):
            requests.append(request)
            self.assertEqual(request.get_method(), "GET")
            self.assertEqual(dict((k.lower(), v) for k, v in request.header_items())
                             ["x-simulated-trading"], "1")
            path = urlsplit(request.full_url).path
            if path == "/api/v5/account/trade-fee":
                data = [{"maker": "-0.0008", "taker": "-0.001"}]
            elif path in ("/api/v5/trade/orders-pending", "/api/v5/trade/orders-algo-pending"):
                data = []
            else:
                self.fail(f"Unexpected GET {path}")
            return {"code": "0", "data": data}
        before = self.journal.path.read_bytes()
        report = inspect_protection_capabilities(public_client=self.public(),
            demo_client=self.demo(transport), engine=self.engine)
        self.assertEqual(report.status, "BLOCK")
        self.assertEqual((report.instrument.min_size, report.instrument.lot_size,
                          report.instrument.tick_size, report.instrument.max_limit_size),
                         (D("0.00001"), D("0.00000001"), D("0.1"), D("100")))
        self.assertEqual((report.fees.maker, report.fees.taker), (D("-0.0008"), D("-0.001")))
        self.assertEqual((report.regular_open_orders, report.pending_algo_orders), (0, 0))
        self.assertTrue(any("атомарно" in x for x in report.unknowns))
        self.assertEqual(len(requests), 4)
        self.assertEqual(self.journal.path.read_bytes(), before)
        self.assertFalse(self.engine.risk_manager.state_path.exists())

    def test_algo_get_pending_and_by_id_are_allowlisted_demo_get_only(self):
        seen = []
        def transport(request, timeout):
            seen.append(request)
            parsed = urlsplit(request.full_url)
            if parsed.path == "/api/v5/trade/orders-algo-pending":
                kind = parse_qs(parsed.query)["ordType"][0]
                data = ([{"algoId": "algo-1", "instType": "SPOT", "instId": "BTC-USDT",
                          "ordType": "conditional", "state": "live", "side": "sell", "sz": "0.01"}]
                        if kind == "conditional" else [])
            elif parsed.path == "/api/v5/trade/order-algo":
                self.assertEqual(parse_qs(parsed.query), {"algoId": ["algo-1"]})
                data = [{"algoId": "algo-1", "instType": "SPOT", "instId": "BTC-USDT",
                         "ordType": "conditional", "state": "live", "side": "sell", "sz": "0.01"}]
            else:
                self.fail("Unexpected endpoint")
            return {"code": "0", "data": data}
        client = self.demo(transport)
        self.assertEqual(client.get_pending_algo_orders()[0].algo_id, "algo-1")
        self.assertEqual(client.get_algo_order_by_id("algo-1").size_btc, D("0.01"))
        self.assertEqual(len(seen), 3)
        for request in seen:
            self.assertEqual(request.get_method(), "GET")
            self.assertEqual(dict((k.lower(), v) for k, v in request.header_items())
                             ["x-simulated-trading"], "1")

    def test_duplicate_mismatched_or_missing_algo_evidence_fails_closed(self):
        duplicate = {"algoId": "same", "instType": "SPOT", "instId": "BTC-USDT",
                     "ordType": "conditional", "state": "live", "side": "sell", "sz": "0.01"}
        def repeated(request, timeout):
            if urlsplit(request.full_url).path.endswith("orders-algo-pending"):
                return {"code": "0", "data": [duplicate, duplicate]}
            raise AssertionError("Unexpected endpoint")
        with self.assertRaises(DemoError):
            self.demo(repeated).get_pending_algo_orders()
        def wrong_id(request, timeout):
            return {"code": "0", "data": [{**duplicate, "algoId": "other"}]}
        with self.assertRaises(DemoError):
            self.demo(wrong_id).get_algo_order_by_id("same")
        with self.assertRaises(DemoError):
            self.demo(wrong_id).get_algo_order_by_id("not valid id")
        def duplicate_across_types(request, timeout):
            kind = parse_qs(urlsplit(request.full_url).query)["ordType"][0]
            return {"code": "0", "data": [{**duplicate, "ordType": kind}]}
        with self.assertRaises(DemoError):
            self.demo(duplicate_across_types).get_pending_algo_orders()

    def test_region_or_live_mismatch_stops_before_any_get(self):
        client = self.demo(lambda request, timeout: self.fail("private GET forbidden"))
        public = self.public()
        public.config = Config(region="us", mode="DEMO")
        report = inspect_protection_capabilities(public_client=public,
                                                  demo_client=client, engine=self.engine)
        self.assertEqual(report.status, "BLOCK")
        self.assertEqual(report.api_observations, ())
        with self.assertRaises(ConfigError):
            DemoClient(Config(region="eea", mode="LIVE"),
                       DemoCredentials("fake-key", "fake-secret", "fake-phrase"))

    def test_missing_max_limit_and_algo_error_are_unknown_not_supported(self):
        def transport(request, timeout):
            path = urlsplit(request.full_url).path
            if path == "/api/v5/account/trade-fee":
                return {"code": "0", "data": [{"maker": "-0.0008", "taker": "-0.001"}]}
            if path == "/api/v5/trade/orders-pending":
                return {"code": "0", "data": []}
            return {"code": "51000", "data": []}
        report = inspect_protection_capabilities(public_client=self.public({
            key: value for key, value in INSTRUMENT.items() if key != "maxLmtSz"}),
            demo_client=self.demo(transport), engine=self.engine)
        self.assertIsNone(report.instrument.max_limit_size)
        self.assertIsNone(report.pending_algo_orders)
        self.assertTrue(any("maxLmtSz" in x for x in report.unknowns))
        self.assertTrue(any("Algo-ордеров недоступен" in x for x in report.unknowns))

    def test_cli_is_read_only_redacts_keys_and_returns_block(self):
        def transport(request, timeout):
            path = urlsplit(request.full_url).path
            if path == "/api/v5/account/trade-fee":
                data = [{"maker": "-0.0008", "taker": "-0.001"}]
            else:
                data = []
            return {"code": "0", "data": data}
        client = self.demo(transport)
        output = StringIO()
        before = self.journal.path.read_bytes()
        with (patch.object(main.DemoClient, "from_env_file", return_value=client),
              patch.object(main, "OkxClient", return_value=self.public()),
              patch.object(main, "DecisionJournal", return_value=self.journal),
              patch.object(main, "ExecutionEngine", return_value=self.engine),
              redirect_stdout(output)):
            self.assertEqual(main.main(["--protection-capabilities"]), 1)
        result = output.getvalue()
        self.assertIn("BLOCK", result)
        self.assertIn("Наблюдение GET", result)
        self.assertIn("Не подтверждено", result)
        self.assertNotIn("fake-key", result)
        self.assertNotIn("fake-secret", result)
        self.assertNotIn("fake-phrase", result)
        self.assertEqual(self.journal.path.read_bytes(), before)

    def test_cli_missing_demo_configuration_never_calls_public_api(self):
        output = StringIO()
        with (patch.object(main.DemoClient, "from_env_file", side_effect=ConfigError("secret text")),
              patch.object(main, "OkxClient", side_effect=AssertionError("public GET forbidden")),
              redirect_stderr(output)):
            self.assertEqual(main.main(["--protection-capabilities"]), 1)
        self.assertIn("BLOCK", output.getvalue())
        self.assertNotIn("secret text", output.getvalue())

    def test_partial_active_canceled_and_fee_mismatch_have_distinct_block_reasons(self):
        client_id = self.seed_partial_buy()
        class Reader:
            config = Config(region="eea", mode="DEMO")
            def __init__(self, state, fee):
                self.state = state
                self.fee = fee
            def get_spot_fee_rates(self):
                return SpotFeeRates(D("-0.0008"), D("-0.001"))
            def get_open_orders(self):
                return []
            def get_pending_algo_orders(self):
                return ()
            def get_order_by_client_id(self, requested):
                assert requested == client_id
                return DemoOrder(client_id, "buy-order", self.state,
                                 D("0.03"), D("0.02"), D("80000"), NOW)
            def get_order_fills(self, order_id):
                return (DemoFill("fill-1", order_id, D("0.02"), D("80000"),
                                 self.fee, "USDT", NOW),)
        active = inspect_protection_capabilities(public_client=self.public(),
            demo_client=Reader("partially_filled", D("-1")), engine=self.engine)
        canceled = inspect_protection_capabilities(public_client=self.public(),
            demo_client=Reader("canceled", D("-1")), engine=self.engine)
        mismatch = inspect_protection_capabilities(public_client=self.public(),
            demo_client=Reader("canceled", D("-2")), engine=self.engine)
        self.assertTrue(any("Остаток активного BUY" in x for x in active.unknowns))
        self.assertTrue(any("После отмены остатка BUY" in x for x in canceled.unknowns))
        self.assertTrue(any("исполнения не сверены" in x for x in mismatch.unknowns))
        self.assertEqual((active.status, canceled.status, mismatch.status),
                         ("BLOCK", "BLOCK", "BLOCK"))


if __name__ == "__main__":
    unittest.main()
