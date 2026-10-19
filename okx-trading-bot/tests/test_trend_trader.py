from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal as D
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from bot.demo_client import DemoCredentials
from bot.demo_trader import DemoTradeClient
from bot.okx_client import OkxClient, OkxResponseError
from bot.trend_trader import (RESUME_PHRASE, DailyClose, TrendStore, TrendTrader, read_daily_closes,
                              resume, target_fraction)
from config import Config
from tests.test_demo_trader import INSTRUMENT, FakeOkx


TODAY = date(2026, 10, 16)


def closes(trend: str, days: int = 70) -> list[DailyClose]:
    step = {"up": D("100"), "down": D("-100"), "flat": D("0")}[trend]
    first = TODAY - timedelta(days=days)
    return [DailyClose(first + timedelta(days=i), D("80000") + step * i) for i in range(days)]


class TargetTests(unittest.TestCase):
    def test_fraction_counts_rising_lookbacks(self) -> None:
        self.assertEqual(target_fraction(closes("up")), D("1"))
        self.assertEqual(target_fraction(closes("down")), D("0"))
        mixed = closes("up")
        mixed[-1] = DailyClose(mixed[-1].day, mixed[-36].close)  # above 40/50/60-day closes only
        self.assertEqual(target_fraction(mixed), D("0.6"))

    def test_daily_candles_require_confirmed_contiguous_utc_days(self) -> None:
        start = int(datetime(2026, 10, 16, tzinfo=timezone.utc).timestamp() * 1000)
        rows = [[str(start - i * 86_400_000), "1", "1", "1", str(80000 + i), "0", "0", "0", "0" if i == 0 else "1"]
                for i in range(70)]
        client = OkxClient(Config(), transport=lambda _url, _t: {"code": "0", "data": rows})
        result = read_daily_closes(client)
        self.assertEqual(result[-1].day, date(2026, 10, 15))
        self.assertEqual(len(result), 69)
        del rows[10]
        with self.assertRaises(OkxResponseError):
            read_daily_closes(client)


class TrendTraderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.okx = FakeOkx()
        self.now = datetime(2026, 10, 16, 0, 5, tzinfo=timezone.utc)
        self.client = DemoTradeClient(Config(mode="DEMO"), DemoCredentials("k", "s", "p"),
                                      transport=self.okx, sleeper=lambda _s: None, clock=lambda: self.now)
        self.store = TrendStore(Path(self.tmp.name) / "trend.sqlite3")
        self.series = closes("up")
        self.messages: list[str] = []

    def trader(self) -> TrendTrader:
        return TrendTrader(client=self.client, store=self.store, closes_provider=lambda: self.series,
                           instrument_provider=lambda: INSTRUMENT, price_provider=lambda: self.okx.price,
                           notifier=self.messages.append, clock=lambda: self.now, sleeper=lambda _s: None)

    def next_day(self, trend: str) -> None:
        self.now += timedelta(days=1)
        self.series = [DailyClose(c.day + timedelta(days=1), c.close) for c in closes(trend)]

    def test_uptrend_buys_whole_book_and_downtrend_sells_it(self) -> None:
        report = self.trader().run_once()
        self.assertEqual(report.status, "REBALANCED", report.lines)
        book = self.store.book(D("100"))
        self.assertGreater(book.btc * self.okx.price, D("98"))
        self.assertGreaterEqual(book.usdt, D("0"))
        self.next_day("down")
        self.assertEqual(self.trader().run_once().status, "REBALANCED")
        book = self.store.book(D("100"))
        self.assertLess(book.btc, INSTRUMENT.lot_size)
        self.assertGreater(book.usdt, D("99.5"))   # two fees of ~0.1%
        self.assertLess(book.usdt, D("100"))
        self.assertEqual(self.okx.btc, D("1") + book.btc)  # external BTC untouched
        self.assertEqual(len(self.store.trades()), 2)
        self.assertIn("КУПЛЕНО", self.messages[0])

    def test_each_day_is_processed_once(self) -> None:
        self.trader().run_once()
        self.now += timedelta(hours=1)
        self.assertEqual(self.trader().run_once().status, "SKIPPED")
        self.assertEqual(len(self.store.trades()), 1)

    def test_small_drift_is_not_traded(self) -> None:
        self.trader().run_once()
        self.next_day("up")
        self.okx.price *= D("1.05")
        self.assertEqual(self.trader().run_once().status, "HOLD")

    def test_lost_order_response_halts_and_resume_books_the_fill(self) -> None:
        self.okx.timeout_on = "/api/v5/trade/order"
        self.assertEqual(self.trader().run_once().status, "HALTED")
        self.assertEqual(self.trader().run_once().status, "HALTED")
        self.assertTrue(resume(self.store, RESUME_PHRASE))
        self.now += timedelta(hours=1)
        self.trader().run_once()
        book = self.store.book(D("100"))
        self.assertGreater(book.btc, D("0"))
        posts = [c for c in self.okx.calls if c[0] == "POST"]
        self.assertEqual(len(posts), 1)

    def test_rejected_order_leaves_book_unchanged(self) -> None:
        self.okx.reject_buy = "51008"
        self.assertEqual(self.trader().run_once().status, "HOLD")
        self.assertEqual(self.store.book(D("100")).usdt, D("100"))

    def test_stale_daily_data_is_ignored(self) -> None:
        self.now += timedelta(days=3)
        self.assertEqual(self.trader().run_once().status, "SKIPPED")
        self.assertFalse(any(c[0] == "POST" for c in self.okx.calls))


if __name__ == "__main__":
    unittest.main()


class LimitRebalanceTests(TrendTraderTests):
    """All trend scenarios again, with maker orders that the fake book fills at once by default."""

    def setUp(self) -> None:
        super().setUp()
        self.okx.fill_after_polls = 1

    def trader(self) -> TrendTrader:
        trader = super().trader()
        trader.quote_provider = lambda: (self.okx.price - D("0.1"), self.okx.price + D("0.1"))
        return trader

    def posts(self, kind: str) -> list:
        return [c for c in self.okx.calls if c[0] == "POST" and c[1] == "/api/v5/trade/order"
                and c[2]["ordType"] == kind]

    def test_maker_order_at_bid_fills(self) -> None:
        self.okx.fill_after_polls = 3
        report = self.trader().run_once()
        self.assertEqual(report.status, "REBALANCED", report.lines)
        [order] = self.posts("post_only")
        self.assertEqual(D(order[2]["px"]), D("99999.9"))
        self.assertEqual(self.posts("market"), [])
        [trade] = self.store.trades()
        self.assertEqual(D(trade["fee"]), D(trade["btc"]) * D("0.0008"))
        self.assertEqual(self.store.book(D("100")).last_day, self.series[-1].day.isoformat())

    def test_unfilled_maker_orders_fall_back_to_market_after_three_tries(self) -> None:
        self.okx.fill_after_polls = None
        for attempt in range(1, 4):
            report = self.trader().run_once()
            self.assertEqual(report.status, "HOLD", report.lines)
            self.assertIn(f"попытка {attempt} из 3", " ".join(report.lines))
            self.assertIsNone(self.store.book(D("100")).last_day)   # day stays open
            self.now += timedelta(hours=1)
        self.assertEqual(self.trader().run_once().status, "REBALANCED")
        self.assertEqual(len(self.posts("market")), 1)
        self.assertEqual(self.okx.btc, D("1") + self.store.book(D("100")).btc)

    def test_partial_fill_is_booked_and_continued(self) -> None:
        self.okx.fill_after_polls = 1
        self.okx.fill_fraction = D("0.5")
        report = self.trader().run_once()
        self.assertEqual(report.status, "REBALANCED", report.lines)
        half = self.store.book(D("100"))
        self.assertGreater(half.btc, D("0"))
        self.assertIsNone(half.last_day)
        self.okx.fill_fraction = D("1")
        self.now += timedelta(hours=1)
        self.trader().run_once()
        book = self.store.book(D("100"))
        self.assertGreater(book.btc * self.okx.price, D("98"))
        self.assertIsNotNone(book.last_day)

    def test_interrupted_run_cancels_resting_order_and_books_it(self) -> None:
        self.okx.fill_after_polls = None
        self.okx.timeout_on = "/api/v5/trade/order"
        self.assertEqual(self.trader().run_once().status, "HALTED")
        client_id = self.store.book(D("100")).pending_client_id
        self.okx._fill_resting(client_id, self.okx.resting[client_id]["sz"] / 2)
        self.assertTrue(resume(self.store, RESUME_PHRASE))
        self.trader().run_once()
        book = self.store.book(D("100"))
        self.assertIsNone(book.pending_client_id)
        self.assertEqual(self.okx.btc, D("1") + book.btc)
