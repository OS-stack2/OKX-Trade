"""Futures experiment: execution, protection, settlement and the rules the model cannot override."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from urllib.parse import parse_qs, urlsplit

from bot.demo_client import DemoCredentials
from bot.futures_trader import (CLOSE, HOLD, OPEN_LONG, OPEN_SHORT, RESUME_PHRASE, FuturesClient,
                                FuturesDecision, FuturesSettings, FuturesStore, FuturesTrader,
                                SwapInstrument, apply_hard_rules, parse_decision, resume)
from bot.news_radar import Classification, NewsError, NewsItem, NewsStore
from bot.news_trader import NewsFeed
from config import Config


NOW = datetime(2026, 11, 6, 10, 2, tzinfo=timezone.utc)
CT = D("0.01")


class FakeSwap:
    """In-memory OKX Demo swap: market orders fill at `price`, OCO triggers on demand."""

    def __init__(self) -> None:
        self.price = D("100000")
        self.usdt = D("5000")
        self.positions: dict[str, dict] = {}
        self.orders: dict[str, dict] = {}
        self.algos: dict[str, dict] = {}
        self.history: list[dict] = []
        self.calls: list[tuple[str, str, object]] = []
        self.fail_algo = False
        self.now_ms = int(NOW.timestamp() * 1000)

    def ok(self, rows):
        return {"code": "0", "msg": "", "data": rows}

    def __call__(self, request, timeout):
        parts = urlsplit(request.full_url)
        assert request.headers["X-simulated-trading"] == "1"
        body = json.loads(request.data) if request.data else None
        q = {k: v[0] for k, v in parse_qs(parts.query).items()}
        method, path = request.get_method(), parts.path
        self.calls.append((method, path, body))
        if method == "GET" and path == "/api/v5/account/balance":
            return self.ok([{"details": [{"ccy": "USDT", "cashBal": str(self.usdt), "availBal": str(self.usdt)}]}])
        if method == "POST" and path == "/api/v5/account/set-leverage":
            return self.ok([{"lever": body["lever"], "posSide": body["posSide"]}])
        if method == "POST" and path == "/api/v5/trade/order":
            self._fill(body["clOrdId"], body["side"], body["posSide"], D(body["sz"]), self.price)
            return self.ok([{"ordId": "O" + body["clOrdId"][-4:], "clOrdId": body["clOrdId"], "sCode": "0"}])
        if method == "GET" and path == "/api/v5/trade/order":
            if q["clOrdId"] not in self.orders:
                return {"code": "51603", "msg": "", "data": []}
            return self.ok([self.orders[q["clOrdId"]]])
        if method == "POST" and path == "/api/v5/trade/order-algo":
            if self.fail_algo:
                return {"code": "1", "msg": "", "data": [{"sCode": "51000", "sMsg": "x"}]}
            algo_id = f"A{len(self.algos) + 1}"
            self.algos[algo_id] = {"algoId": algo_id, "instId": "BTC-USDT-SWAP", "state": "live",
                                   "posSide": body["posSide"], "sz": body["sz"],
                                   "tp": D(body["tpTriggerPx"]), "sl": D(body["slTriggerPx"])}
            return self.ok([{"algoId": algo_id, "sCode": "0"}])
        if method == "GET" and path == "/api/v5/trade/order-algo":
            algo = self.algos[q["algoId"]]
            return self.ok([{k: v for k, v in algo.items() if k not in ("tp", "sl")}])
        if method == "POST" and path == "/api/v5/trade/cancel-algos":
            algo = self.algos[body[0]["algoId"]]
            if algo["state"] == "live":
                algo["state"] = "canceled"
            return self.ok([{"algoId": algo["algoId"], "sCode": "0"}])
        if method == "GET" and path == "/api/v5/account/positions":
            return self.ok([{"instId": "BTC-USDT-SWAP", "posSide": side, "mgnMode": "isolated",
                             "pos": str(p["pos"]), "avgPx": str(p["avg"]), "upl": str(self.upl(side)),
                             "liqPx": "1"} for side, p in self.positions.items()])
        if method == "GET" and path == "/api/v5/account/positions-history":
            return self.ok(self.history)
        raise AssertionError(f"unexpected {method} {path}")

    def upl(self, side: str) -> D:
        p = self.positions[side]
        sign = 1 if side == "long" else -1
        return (self.price - p["avg"]) * p["pos"] * CT * sign

    def _fill(self, client_id, side, pos_side, size, price) -> None:
        opening = (side == "buy") == (pos_side == "long")
        self.orders[client_id] = {"instId": "BTC-USDT-SWAP", "ordId": "O" + client_id[-4:], "clOrdId": client_id,
                                  "side": side, "posSide": pos_side, "state": "filled",
                                  "accFillSz": str(size), "avgPx": str(price)}
        if opening:
            self.positions[pos_side] = {"pos": size, "avg": price}
            return
        p = self.positions.pop(pos_side)
        sign = 1 if pos_side == "long" else -1
        pnl = (price - p["avg"]) * size * CT * sign - price * size * CT * D("0.001")
        self.history.append({"posSide": pos_side, "mgnMode": "isolated", "type": "2",
                             "realizedPnl": str(pnl), "closeAvgPx": str(price), "uTime": str(self.now_ms)})

    def trigger(self, algo_id: str, leg: str) -> None:
        algo = self.algos[algo_id]
        algo["state"], algo["actualSide"] = "effective", leg
        side = "sell" if algo["posSide"] == "long" else "buy"
        self._fill("FTx" + algo_id, side, algo["posSide"], D(algo["sz"]), algo[leg])


class FakeAnalyst:
    def __init__(self) -> None:
        self.answers: list[FuturesDecision] = []
        self.situations: list[str] = []

    def decide(self, *, settings, situation):
        self.situations.append(situation)
        if not self.answers:
            raise NewsError("нет ответа")
        return self.answers.pop(0)


def long(conf=80, lever=3, margin="50", stop="2", take="4"):
    return FuturesDecision(OPEN_LONG, conf, lever, D(margin), D(stop), D(take), ("тренд вверх",))


class FuturesTraderTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.okx = FakeSwap()
        self.now = NOW
        self.client = FuturesClient(Config(mode="DEMO"), DemoCredentials("k", "s", "p"),
                                    transport=self.okx, sleeper=lambda _s: None, clock=lambda: self.now)
        self.store = FuturesStore(self.root / "futures.sqlite3")
        self.analyst = FakeAnalyst()
        self.messages: list[str] = []
        self.news = NewsStore(self.root / "news.sqlite3")
        self.news.add([], NOW, primed=True)
        self.trader = FuturesTrader(
            client=self.client, store=self.store, settings=FuturesSettings(), analyst=self.analyst,
            instrument=lambda: SwapInstrument(CT, D("0.01"), D("0.01"), D("0.1"), D("100"), "live"),
            price=lambda: self.okx.price, market_text=lambda _now: "context",
            feed=NewsFeed(self.news.path, self.store, directions=("bullish", "bearish")),
            notifier=self.messages.append, clock=lambda: self.now, sleeper=lambda _s: None)

    def tick(self, minutes: float = 0):
        self.now += timedelta(minutes=minutes)
        self.okx.now_ms = int(self.now.timestamp() * 1000)
        return self.trader.step()

    def open_long(self, **kwargs):
        self.analyst.answers.append(long(**kwargs))
        report = self.tick()
        self.assertEqual(report.status, "OPENED", report.lines)
        return self.store.active_position()

    def test_hourly_long_opens_with_leverage_and_exchange_oco(self) -> None:
        position = self.open_long()
        self.assertEqual((position.state, position.side, position.lever), ("OPEN", "long", 3))
        # 50% of 1000 as margin x3 = 1500 notional; the 2% stop + fees costs ~33 USDT < 250 cap.
        self.assertEqual(position.contracts, D("1.5"))
        self.assertEqual((position.stop_price, position.take_price), (D("98000.0"), D("104000.0")))
        self.assertIn(("POST", "/api/v5/account/set-leverage",
                       {"instId": "BTC-USDT-SWAP", "lever": "3", "mgnMode": "isolated", "posSide": "long"}),
                      self.okx.calls)
        self.assertIn("Открыт LONG", self.messages[-1])
        self.assertIn("тренд вверх", self.messages[-1])

    def test_loss_at_stop_is_capped_at_a_quarter_of_the_book(self) -> None:
        position = self.open_long(margin="100", stop="10", take="20")
        loss = position.contracts * CT * D("100000") * (D("0.10") + D("0.001"))
        self.assertLessEqual(loss, D("250.01"))
        self.assertGreater(loss, D("240"))

    def test_take_profit_on_exchange_is_settled_from_history(self) -> None:
        position = self.open_long()
        self.okx.trigger(position.algo_id, "tp")
        report = self.tick(1)
        self.assertEqual(report.status, "CLOSED", report.lines)
        closed = self.store.closed_positions()[0]
        self.assertEqual(closed.exit_reason, "TAKE_PROFIT")
        self.assertEqual(closed.pnl_usdt.quantize(D("0.01")), D("58.44"))   # 1.5 × 0.01 × 4000 − fees
        self.assertIn("PnL +58.44", self.messages[-1])

    def test_close_decision_cancels_oco_and_closes_at_market(self) -> None:
        position = self.open_long()
        self.okx.price = D("101000")
        self.analyst.answers.append(FuturesDecision(CLOSE, 70, reasons_ru=("импульс выдохся",)))
        report = self.tick(60)
        self.assertEqual(report.status, "CLOSED", report.lines)
        self.assertEqual(self.okx.algos[position.algo_id]["state"], "canceled")
        self.assertEqual(self.store.closed_positions()[0].exit_reason, "SIGNAL")
        self.assertEqual(self.okx.positions, {})

    def test_short_side_and_stop_loss(self) -> None:
        self.analyst.answers.append(FuturesDecision(OPEN_SHORT, 75, 2, D("40"), D("1.5"), D("3")))
        self.assertEqual(self.tick().status, "OPENED")
        position = self.store.active_position()
        self.assertEqual((position.side, position.stop_price, position.take_price),
                         ("short", D("101500.0"), D("97000.0")))
        self.okx.trigger(position.algo_id, "sl")
        self.tick(1)
        closed = self.store.closed_positions()[0]
        self.assertEqual(closed.exit_reason, "STOP_LOSS")
        self.assertLess(closed.pnl_usdt, 0)

    def test_book_stop_closes_and_halts_for_good(self) -> None:
        self.open_long(margin="100")
        self.okx.price = D("66000")          # upl ≈ −510 on 1.5 contracts — below the 500 floor
        report = self.tick(0.5)
        self.assertEqual(report.status, "HALTED", report.lines)
        self.assertIn("BOOK_STOP", self.store.get_meta("halted"))
        self.assertEqual(self.store.closed_positions()[0].exit_reason, "BOOK_STOP")
        self.assertIn("СТОП КНИГИ", self.messages[-1])
        self.assertFalse(resume(self.store, RESUME_PHRASE, self.now))
        self.analyst.answers.append(long())
        self.assertIsNone(self.tick(60))

    def test_failed_protection_closes_and_halts(self) -> None:
        self.okx.fail_algo = True
        self.analyst.answers.append(long())
        report = self.tick()
        self.assertEqual(report.status, "HALTED", report.lines)
        self.assertEqual(self.okx.positions, {})
        self.assertTrue(resume(self.store, RESUME_PHRASE, self.now))

    def test_hourly_review_once_per_hour_after_minute_two(self) -> None:
        self.now = NOW.replace(minute=1)
        self.assertIsNone(self.trader.step())
        self.assertEqual(self.analyst.situations, [])
        self.analyst.answers.append(FuturesDecision(HOLD, 40))
        self.assertEqual(self.tick(1).status, "HOLD")
        self.assertIsNone(self.tick(30))
        self.assertEqual(len(self.analyst.situations), 1)
        self.assertEqual(self.messages, [])           # a quiet hourly HOLD is not sent to Telegram

    def test_bearish_news_reaches_the_trader(self) -> None:
        self.analyst.answers.append(FuturesDecision(HOLD, 40))
        self.tick()                                   # hourly review; also primes the news cursor
        item = NewsItem("tg:watcherguru", "tg:watcherguru/1", "https://t.me/x", self.now, "Биржа взломана")
        self.news.add([item], self.now, primed=True)
        self.news.save_classification(item.key, Classification(True, "bearish", 3, True, False, "Взлом биржи"))
        self.analyst.answers.append(FuturesDecision(OPEN_SHORT, 70, 2, D("30"), D("2"), D("5")))
        self.assertEqual(self.tick(0.1).status, "OPENED")
        self.assertIn("classified bearish", self.analyst.situations[-1])
        self.assertIn("новость", self.messages[-1])

    def test_model_failure_is_a_skipped_tick_not_a_halt(self) -> None:
        report = self.tick()
        self.assertEqual(report.status, "ERROR")
        self.assertIsNone(self.store.get_meta("halted"))

    def test_situation_shows_the_open_position_and_history(self) -> None:
        self.open_long()
        self.analyst.answers.append(FuturesDecision(HOLD, 60))
        self.tick(60)
        situation = self.analyst.situations[-1]
        self.assertIn("Open position: LONG 1.5 contracts", situation)
        self.assertIn("OPEN_LONG (80%)", situation)


class HardRuleTests(unittest.TestCase):
    def test_rules(self) -> None:
        settings = FuturesSettings()
        wild = FuturesDecision(OPEN_LONG, 90, 20, D("500"), D("0.01"), D("99"))
        clamped = apply_hard_rules(wild, settings=settings, has_position=False, entries_today=0)
        self.assertEqual((clamped.action, clamped.leverage, clamped.margin_pct), (OPEN_LONG, 3, D("100")))
        self.assertEqual((clamped.stop_loss_pct, clamped.take_profit_pct), (D("0.3"), D("30")))
        for kwargs in ({"has_position": True, "entries_today": 0},
                       {"has_position": False, "entries_today": 6}):
            self.assertEqual(apply_hard_rules(wild, settings=settings, **kwargs).action, HOLD)
        shy = FuturesDecision(OPEN_SHORT, 30, 2, D("20"), D("1"), D("2"))
        self.assertEqual(apply_hard_rules(shy, settings=settings, has_position=False, entries_today=0).action, HOLD)
        close = FuturesDecision(CLOSE, 90)
        self.assertEqual(apply_hard_rules(close, settings=settings, has_position=False, entries_today=0).action, HOLD)

    def test_parse(self) -> None:
        decision = parse_decision('думаю... {"action": "OPEN_SHORT", "confidence": 72, "leverage": 2, '
                                  '"margin_pct": 30, "stop_loss_pct": 1.5, "take_profit_pct": 4, '
                                  '"reasons_ru": ["перегрев"], "lesson_ru": "не гнаться"}')
        self.assertEqual((decision.action, decision.leverage, decision.stop_loss_pct), (OPEN_SHORT, 2, D("1.5")))
        with self.assertRaises(NewsError):
            parse_decision('{"action": "YOLO"}')


if __name__ == "__main__":
    unittest.main()
