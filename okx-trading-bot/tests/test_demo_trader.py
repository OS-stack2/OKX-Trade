from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from urllib.parse import parse_qs, urlsplit

from bot.demo_client import DemoCredentials, DemoError
from bot.demo_trader import (ABANDON_PHRASE, RESUME_PHRASE, DemoTradeClient, DemoTrader, TraderSettings,
                             TraderStore, abandon_position, resume)
from bot.market_data import Candle, MarketSnapshot, SpotInstrument
from bot.strategy import Signal, StrategyResult
from config import Config, ConfigError


NOW = datetime(2026, 10, 16, 12, 1, tzinfo=timezone.utc)
INSTRUMENT = SpotInstrument("BTC-USDT", D("0.1"), D("0.00000001"), D("0.00001"), "live", "BTC", "USDT",
                            D("9000000"), D("1000000"))


class FakeOkx:
    """In-memory OKX Demo: market orders fill instantly at `price`."""

    def __init__(self) -> None:
        self.price = D("100000")
        self.usdt = D("10000")
        self.btc = D("1")           # external BTC never touched by the bot
        self.orders: dict[str, dict] = {}
        self.algos: dict[str, dict] = {}
        self.calls: list[tuple[str, str, object]] = []
        self.fail_algo: str | None = None
        self.timeout_on: str | None = None
        self.reject_buy: str | None = None

    def __call__(self, request, timeout):
        parts = urlsplit(request.full_url)
        assert request.headers["X-simulated-trading"] == "1"
        body = json.loads(request.data) if request.data else None
        self.calls.append((request.get_method(), parts.path, body))
        if self.timeout_on == parts.path:
            self.timeout_on = None
            self._handle(request.get_method(), parts.path, parse_qs(parts.query), body)
            raise TimeoutError()
        return self._handle(request.get_method(), parts.path, parse_qs(parts.query), body)

    def ok(self, rows):
        return {"code": "0", "msg": "", "data": rows}

    def _handle(self, method, path, query, body):
        q = {k: v[0] for k, v in query.items()}
        if method == "GET" and path == "/api/v5/account/balance":
            return self.ok([{"details": [
                {"ccy": "BTC", "cashBal": str(self.btc), "availBal": str(self.btc)},
                {"ccy": "USDT", "cashBal": str(self.usdt), "availBal": str(self.usdt)}]}])
        if method == "GET" and path == "/api/v5/account/trade-fee":
            return self.ok([{"maker": "-0.0008", "taker": "-0.001"}])
        if method == "POST" and path == "/api/v5/trade/order":
            if body["side"] == "buy" and self.reject_buy:
                return {"code": "1", "msg": "", "data": [{"sCode": self.reject_buy, "sMsg": "x", "ordId": ""}]}
            return self.ok([{"ordId": self._fill(body["clOrdId"], body["side"], D(body["sz"])),
                             "clOrdId": body["clOrdId"], "sCode": "0", "sMsg": ""}])
        if method == "GET" and path == "/api/v5/trade/order":
            order = (self.orders.get(q["clOrdId"]) if "clOrdId" in q
                     else next((o for o in self.orders.values() if o["ordId"] == q["ordId"]), None))
            if order is None:
                return {"code": "51603", "msg": "Order does not exist", "data": []}
            return self.ok([order])
        if method == "POST" and path == "/api/v5/trade/order-algo":
            if self.fail_algo:
                return {"code": "1", "msg": "", "data": [{"sCode": self.fail_algo, "sMsg": "x", "algoId": ""}]}
            algo_id = f"A{len(self.algos) + 1}"
            self.algos[algo_id] = {"algoId": algo_id, "instId": "BTC-USDT", "state": "live", "sz": body["sz"],
                                   "tp": D(body["tpTriggerPx"]), "sl": D(body["slTriggerPx"]),
                                   "ordIdList": [], "actualSide": ""}
            return self.ok([{"algoId": algo_id, "sCode": "0", "sMsg": ""}])
        if method == "GET" and path == "/api/v5/trade/order-algo":
            algo = dict(self.algos[q["algoId"]])
            algo.pop("tp"), algo.pop("sl")
            return self.ok([algo])
        if method == "POST" and path == "/api/v5/trade/cancel-algos":
            self.algos[body[0]["algoId"]]["state"] = "canceled"
            return self.ok([{"algoId": body[0]["algoId"], "sCode": "0", "sMsg": ""}])
        raise AssertionError(f"unexpected {method} {path}")

    def _fill(self, client_id: str, side: str, size: D) -> str:
        order_id = f"O{len(self.orders) + 1}"
        if side == "buy":
            fee, ccy = -(size * D("0.001")), "BTC"
            self.usdt -= size * self.price
            self.btc += size + fee
        else:
            fee, ccy = -(size * self.price * D("0.001")), "USDT"
            self.btc -= size
            self.usdt += size * self.price + fee
        self.orders[client_id or order_id] = {
            "instId": "BTC-USDT", "ordId": order_id, "clOrdId": client_id, "side": side, "state": "filled",
            "accFillSz": str(size), "avgPx": str(self.price), "fee": str(fee), "feeCcy": ccy}
        return order_id

    def trigger(self, algo_id: str, side: str) -> None:
        algo = self.algos[algo_id]
        self.price = algo["tp"] if side == "tp" else algo["sl"]
        order_id = self._fill("", "sell", D(algo["sz"]))
        algo.update(state="effective", ordIdList=[order_id], actualSide=side)


def market(price: D = D("100000")) -> MarketSnapshot:
    candle = Candle(NOW - timedelta(minutes=16), price, price, price, price, True)
    return MarketSnapshot(INSTRUMENT, price, (candle,), NOW)


def signal(kind: Signal, closed_at: datetime = NOW - timedelta(minutes=1)) -> StrategyResult:
    return StrategyResult(kind, D("1"), D("1"), D("1"), D("1"), closed_at, "test")


class DemoTraderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.okx = FakeOkx()
        self.client = DemoTradeClient(Config(mode="DEMO"), DemoCredentials("k", "s", "p"),
                                      transport=self.okx, sleeper=lambda _s: None, clock=lambda: self.now)
        self.store = TraderStore(Path(self.tmp.name) / "t.sqlite3")
        self.now = NOW
        self.signal = signal(Signal.BUY)
        self.messages: list[str] = []

    def trader(self) -> DemoTrader:
        return DemoTrader(client=self.client, store=self.store, market_provider=lambda: market(self.okx.price),
                          notifier=self.messages.append, clock=lambda: self.now, sleeper=lambda _s: None,
                          signal_provider=lambda _candles: self.signal)

    def assert_external_btc_untouched(self) -> None:
        # Only sub-lot dust from the BTC-denominated buy fee may remain.
        self.assertTrue(D("1") <= self.okx.btc < D("1") + INSTRUMENT.lot_size, self.okx.btc)

    def advance(self, kind: Signal, minutes: int = 15) -> None:
        self.now += timedelta(minutes=minutes)
        self.signal = signal(kind, self.now - timedelta(minutes=1))

    def test_buy_is_sized_by_risk_and_protected_by_oco(self) -> None:
        report = self.trader().run_once()
        self.assertEqual(report.status, "OPENED", report.lines)
        position = self.store.active_position()
        self.assertEqual(position.state, "OPEN")
        # 0.5% of 5000 USDT budget = 25 USDT risk; ~1.3% risk per BTC unit at 100k.
        risk = position.planned_btc * (D("100000") - position.stop_price)
        self.assertLess(risk, D("25"))
        self.assertGreater(risk, D("15"))
        self.assertEqual(position.stop_price, D("99000.0"))
        self.assertEqual(position.take_price, D("102000.0"))
        net = position.planned_btc * D("0.999")
        self.assertTrue(net - INSTRUMENT.lot_size < position.net_btc <= net)  # floored to lotSz
        algo = self.okx.algos[position.algo_id]
        self.assertEqual(D(algo["sz"]), position.net_btc)
        self.assertIn("ПОКУПКА", self.messages[0])

    def test_take_profit_closes_with_profit(self) -> None:
        self.trader().run_once()
        position = self.store.active_position()
        self.okx.trigger(position.algo_id, "tp")
        self.advance(Signal.HOLD)
        report = self.trader().run_once()
        self.assertEqual(report.status, "CLOSED", report.lines)
        closed = self.store.closed_positions()[0]
        self.assertEqual(closed.exit_reason, "TAKE_PROFIT")
        self.assertGreater(closed.pnl_usdt, D("0"))
        self.assertIsNone(self.store.active_position())

    def test_stop_loss_loss_is_bounded(self) -> None:
        self.trader().run_once()
        self.okx.trigger(self.store.active_position().algo_id, "sl")
        self.advance(Signal.HOLD)
        self.trader().run_once()
        closed = self.store.closed_positions()[0]
        self.assertEqual(closed.exit_reason, "STOP_LOSS")
        self.assertLess(closed.pnl_usdt, D("0"))
        self.assertGreater(closed.pnl_usdt, D("-25"))

    def test_sell_signal_cancels_oco_and_sells_only_bot_btc(self) -> None:
        self.trader().run_once()
        position = self.store.active_position()
        self.advance(Signal.SELL)
        report = self.trader().run_once()
        self.assertEqual(report.status, "CLOSED", report.lines)
        self.assertEqual(self.okx.algos[position.algo_id]["state"], "canceled")
        self.assert_external_btc_untouched()
        self.assertEqual(self.store.closed_positions()[0].exit_reason, "SIGNAL")

    def test_failed_protection_sells_immediately_and_halts(self) -> None:
        self.okx.fail_algo = "51000"
        report = self.trader().run_once()
        self.assertEqual(report.status, "HALTED")
        self.assertEqual(self.store.closed_positions()[0].exit_reason, "EMERGENCY")
        self.assert_external_btc_untouched()
        self.assertEqual(self.trader().run_once().status, "HALTED")
        self.assertTrue(any("АВАРИЙНАЯ" in m for m in self.messages))

    def test_lost_buy_response_halts_then_resume_protects_the_fill(self) -> None:
        self.okx.timeout_on = "/api/v5/trade/order"
        self.assertEqual(self.trader().run_once().status, "HALTED")
        self.assertEqual(self.store.active_position().state, "OPENING")
        buys = [c for c in self.okx.calls if c[:2] == ("POST", "/api/v5/trade/order")]
        self.assertEqual(len(buys), 1)  # never resent blindly
        self.assertTrue(resume(self.store, RESUME_PHRASE, self.now))
        self.advance(Signal.HOLD, minutes=5)
        self.trader().run_once()
        self.assertEqual(self.store.active_position().state, "OPEN")

    def test_rejected_buy_spends_nothing(self) -> None:
        self.okx.reject_buy = "51008"
        report = self.trader().run_once()
        self.assertEqual(report.status, "HOLD")
        self.assertIsNone(self.store.active_position())
        self.assertEqual(self.okx.usdt, D("10000"))

    def test_same_candle_is_acted_on_once(self) -> None:
        self.okx.reject_buy = "51008"
        self.trader().run_once()
        self.okx.reject_buy = None
        self.now += timedelta(minutes=5)
        report = self.trader().run_once()
        self.assertEqual(report.status, "SKIPPED")
        self.assertIsNone(self.store.active_position())

    def test_stale_signal_is_ignored(self) -> None:
        self.signal = signal(Signal.BUY, NOW - timedelta(minutes=40))
        self.assertEqual(self.trader().run_once().status, "SKIPPED")
        self.assertFalse(any(c[0] == "POST" for c in self.okx.calls))

    def test_daily_entry_limit(self) -> None:
        for _ in range(5):
            self.trader().run_once()
            self.okx.trigger(self.store.active_position().algo_id, "tp")
            self.advance(Signal.HOLD)
            self.trader().run_once()
            self.advance(Signal.BUY)
        report = self.trader().run_once()
        self.assertEqual(report.status, "HOLD")
        self.assertIn("лимит входов", " ".join(report.lines))

    def test_externally_canceled_oco_halts(self) -> None:
        self.trader().run_once()
        self.okx.algos[self.store.active_position().algo_id]["state"] = "canceled"
        self.advance(Signal.HOLD)
        self.assertEqual(self.trader().run_once().status, "HALTED")

    def test_abandon_after_manual_close(self) -> None:
        self.trader().run_once()
        self.assertFalse(abandon_position(self.store, "wrong", self.now))
        self.assertTrue(abandon_position(self.store, ABANDON_PHRASE, self.now))
        self.assertIsNone(self.store.active_position())

    def test_owner_close_now(self) -> None:
        self.trader().run_once()
        algo = self.store.active_position().algo_id
        report = self.trader().close_now()
        self.assertEqual(report.status, "CLOSED", report.lines)
        self.assertEqual(self.okx.algos[algo]["state"], "canceled")
        self.assertEqual(self.store.closed_positions()[0].exit_reason, "MANUAL")
        self.assertEqual(self.trader().close_now().lines[-1], "Открытой позиции нет")

    def test_live_mode_and_unknown_endpoints_are_refused(self) -> None:
        with self.assertRaises(ConfigError):
            DemoTradeClient(Config(mode="READ_ONLY"), DemoCredentials("k", "s", "p"))
        with self.assertRaises(ConfigError):
            Config(mode="LIVE")
        with self.assertRaises(DemoError):
            self.client._post("/api/v5/asset/withdrawal", {})

    def test_trade_key_falls_back_to_existing_demo_key(self) -> None:
        env = Path(self.tmp.name) / ".env"
        env.write_text("OKX_MODE=DEMO\nOKX_REGION=global\nOKX_DEMO_API_KEY=a\n"
                       "OKX_DEMO_API_SECRET=b\nOKX_DEMO_PASSPHRASE=c\n", encoding="utf-8")
        self.assertEqual(DemoTradeClient.from_env_file(env)._credentials.api_key, "a")
        with env.open("a", encoding="utf-8") as handle:
            handle.write("OKX_DEMO_TRADE_API_KEY=t\nOKX_DEMO_TRADE_API_SECRET=u\nOKX_DEMO_TRADE_PASSPHRASE=v\n")
        self.assertEqual(DemoTradeClient.from_env_file(env)._credentials.api_key, "t")

    def test_trader_requires_explicit_enable(self) -> None:
        env = Path(self.tmp.name) / ".env"
        env.write_text("OKX_MODE=DEMO\n", encoding="utf-8")
        with self.assertRaises(ConfigError):
            TraderSettings.from_env_file(env)
        env.write_text("OKX_MODE=DEMO\nOKX_TRADER_ENABLED=1\nOKX_TRADER_BUDGET_USDT=1000\n", encoding="utf-8")
        self.assertEqual(TraderSettings.from_env_file(env).budget_usdt, D("1000"))


if __name__ == "__main__":
    unittest.main()
