from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from bot.demo_client import DemoCredentials
from bot.demo_trader import DemoTradeClient, TraderSettings, TraderStore
from bot.news_radar import Classification, NewsItem, NewsStore
from bot.news_trader import NewsFeed, NewsTrader, NewsTraderSettings
from config import Config
from tests.test_demo_trader import FakeOkx, market


NOW = datetime(2026, 10, 16, 12, 1, tzinfo=timezone.utc)


class NewsTraderTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.okx = FakeOkx()
        self.now = NOW
        self.client = DemoTradeClient(Config(mode="DEMO"), DemoCredentials("k", "s", "p"),
                                      transport=self.okx, sleeper=lambda _s: None, clock=lambda: self.now)
        self.store = TraderStore(self.root / "news_trader.sqlite3")
        self.news = NewsStore(self.root / "news.sqlite3")
        self.news.add([], NOW, primed=True)  # create schema
        self.messages: list[str] = []
        self.brake = None
        self.trader = NewsTrader(
            feed=NewsFeed(self.news.path, self.store),
            news_settings=NewsTraderSettings(TraderSettings(budget_usdt=D("1000"))),
            client=self.client, store=self.store, market_provider=lambda: market(self.okx.price),
            notifier=self.messages.append, clock=lambda: self.now, sleeper=lambda _s: None,
            brake=lambda _now: self.brake)
        self.assertIsNone(self.trader.step())  # first tick only sets the cursor
        self.counter = 0

    def publish(self, *, direction="bullish", impact=3, confirmed=True, stale=False) -> None:
        self.counter += 1
        item = NewsItem("tg:watcherguru", f"tg:watcherguru/{self.counter}", "https://t.me/x", self.now, "USHA")
        self.news.add([item], self.now, primed=True)
        self.news.save_classification(item.key, Classification(True, direction, impact, confirmed, stale,
                                                               "УША переходит на биткоин"))

    def test_bullish_news_buys_20_percent_with_oco(self) -> None:
        self.publish()
        report = self.trader.step()
        self.assertEqual(report.status, "OPENED", report.lines)
        position = self.store.active_position()
        self.assertEqual(position.state, "OPEN")
        notional = position.planned_btc * D("100000")
        self.assertTrue(D("199") < notional <= D("200"), notional)  # 20% of 1000 incl. costs
        self.assertIn("OKX Demo новостной бот: ПОКУПКА", self.messages[0])

    def test_time_exit_after_hold(self) -> None:
        self.publish()
        self.trader.step()
        self.now += timedelta(minutes=29)
        self.assertIsNone(self.trader.step())
        self.now += timedelta(minutes=2)
        report = self.trader.step()
        self.assertEqual(report.status, "CLOSED", report.lines)
        self.assertEqual(self.store.closed_positions()[0].exit_reason, "TIME")
        self.assertEqual(self.okx.btc.quantize(D("0.00001")), D("1.00000"))

    def test_oco_take_profit_before_time(self) -> None:
        self.publish()
        self.trader.step()
        self.okx.trigger(self.store.active_position().algo_id, "tp")
        self.now += timedelta(minutes=5)
        self.assertEqual(self.trader.step().status, "CLOSED")
        self.assertEqual(self.store.closed_positions()[0].exit_reason, "TAKE_PROFIT")

    def test_ignored_items(self) -> None:
        for kwargs in ({"direction": "bearish"}, {"confirmed": False}, {"stale": True}, {"impact": 1}):
            self.publish(**kwargs)
        self.assertIsNone(self.trader.step())
        self.assertFalse(any(c[0] == "POST" for c in self.okx.calls))

    def test_old_news_and_backlog_are_ignored(self) -> None:
        self.publish()
        self.now += timedelta(minutes=10)
        self.assertIsNone(self.trader.step())
        self.assertIsNone(self.store.active_position())

    def test_one_position_at_a_time_and_news_not_queued(self) -> None:
        self.publish()
        self.trader.step()
        self.publish()
        self.assertEqual(self.trader.step().status, "HOLD")
        self.now += timedelta(minutes=31)
        self.trader.step()  # time exit
        self.assertIsNone(self.trader.step())  # the skipped item is not replayed
        self.assertEqual(len(self.store.closed_positions()), 1)

    def test_brake_blocks_news_buy(self) -> None:
        self.brake = "биржа взломана"
        self.publish()
        report = self.trader.step()
        self.assertIn("тормоз по новостям", " ".join(report.lines))
        self.assertIsNone(self.store.active_position())

    def test_halt_message_names_news_resume_command(self) -> None:
        self.okx.fail_algo = "51000"
        self.publish()
        self.assertEqual(self.trader.step().status, "HALTED")
        self.assertTrue(any("--news-trade-resume" in m for m in self.messages))
        self.publish()
        self.assertIsNone(self.trader.step())


if __name__ == "__main__":
    unittest.main()
