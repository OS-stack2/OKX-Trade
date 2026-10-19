from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from bot.news_radar import (Classification, NewsError, NewsItem, NewsRadar, NewsStore, active_brake,
                            paper_stats, parse_classification, parse_rss, parse_telegram)
from bot.strategy import Signal
from tests.test_demo_trader import DemoTraderTests


NOW = datetime(2026, 10, 19, 12, 0, 30, tzinfo=timezone.utc)


def tg_page(posts: list[tuple[int, str, datetime]]) -> str:
    blocks = []
    for number, text, at in posts:
        blocks.append(
            f'<div class="tgme_widget_message js-widget_message" data-post="WatcherGuru/{number}">'
            f'<div class="tgme_widget_message_text js-message_text" dir="auto">{text}</div>'
            f'<a class="tgme_widget_message_date"><time datetime="{at.isoformat()}" class="time">x</time></a></div>')
    return "<html>" + "".join(blocks) + "</html>"


RSS = """<?xml version="1.0"?><rss><channel>
<item><title><![CDATA[SEC approves &amp; lists]]></title><link>https://x/1</link><guid>g1</guid>
<description>&lt;p&gt;Body&lt;/p&gt;</description><pubDate>Mon, 19 Oct 2026 11:59:00 GMT</pubDate></item>
</channel></rss>"""


class FakeClassifier:
    def __init__(self) -> None:
        self.result = Classification(True, "bullish", 3, True, False, "США переходят на биткоин")
        self.calls: list[NewsItem] = []
        self.fail = False

    def classify(self, item: NewsItem) -> Classification:
        self.calls.append(item)
        if self.fail:
            raise NewsError("DeepSeek не ответил: TimeoutError")
        return self.result


class ParsingTests(unittest.TestCase):
    def test_telegram_post_text_time_and_entities(self) -> None:
        page = tg_page([(15559, "<b>JUST IN:</b> &quot;USHA&quot; adopts <i>bitcoin</i><br/>more", NOW)])
        [item] = parse_telegram("watcherguru", page)
        self.assertEqual(item.key, "tg:watcherguru/15559")
        self.assertEqual(item.text, 'JUST IN: "USHA" adopts bitcoin\nmore')
        self.assertEqual(item.published_at, NOW)
        self.assertEqual(item.url, "https://t.me/WatcherGuru/15559")

    def test_media_post_without_text_is_skipped(self) -> None:
        page = '<div data-post="WatcherGuru/1"><time datetime="2026-10-19T12:00:00+00:00"></time></div>'
        self.assertEqual(parse_telegram("watcherguru", page), [])

    def test_rss_item(self) -> None:
        [item] = parse_rss("sec", RSS, NOW)
        self.assertEqual(item.text, "SEC approves & lists\nBody")
        self.assertEqual(item.published_at, datetime(2026, 10, 19, 11, 59, tzinfo=timezone.utc))
        with self.assertRaises(NewsError):
            parse_rss("sec", "<rss", NOW)

    def test_classification_validation(self) -> None:
        good = parse_classification({"btc_relevant": True, "direction": "bearish", "impact": 2,
                                     "confirmed": True, "stale": False, "summary_ru": "x"})
        self.assertTrue(good.actionable)
        for bad in ({"direction": "up", "impact": 1}, {"direction": "bullish", "impact": 5},
                    {"direction": "bullish", "impact": "3"}, []):
            with self.assertRaises(NewsError):
                parse_classification(bad)
        rumour = parse_classification({"btc_relevant": True, "direction": "bullish", "impact": 3,
                                       "confirmed": False, "stale": False})
        self.assertFalse(rumour.actionable)


class RadarTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = NewsStore(Path(tmp.name) / "news.sqlite3")
        self.classifier = FakeClassifier()
        self.pages: dict[str, str] = {}
        self.now = NOW
        self.messages: list[str] = []
        self.candles: list[list[str]] = []
        self.radar = NewsRadar(store=self.store, classifier=self.classifier,
                               fetch_page=lambda url: self.pages.get(url, "<html></html>"),
                               price_now=lambda: D("80000"), minute_candles=lambda: self.candles,
                               notifier=self.messages.append, clock=lambda: self.now)

    def set_posts(self, posts) -> None:
        self.pages["https://t.me/s/watcherguru"] = tg_page(posts)

    def test_first_read_primes_backlog_without_classifying_or_alerting(self) -> None:
        self.set_posts([(1, "old big news", NOW - timedelta(minutes=1))])
        result = self.radar.poll(telegram=True, rss=False)
        self.assertEqual((result.new_items, self.classifier.calls, self.messages), (0, [], []))
        self.set_posts([(1, "old big news", NOW - timedelta(minutes=1)), (2, "USHA adopts BTC", NOW)])
        self.now += timedelta(seconds=30)
        result = self.radar.poll(telegram=True, rss=False)
        self.assertEqual(result.alerts, 1)
        self.assertEqual([i.key for i in self.classifier.calls], ["tg:watcherguru/2"])
        self.assertIn("🟢 рост, сила 3/3", self.messages[0])
        self.assertIn("через 30 с", self.messages[0])
        # Seen again on the next poll: no second alert.
        self.now += timedelta(seconds=30)
        self.assertEqual(self.radar.poll(telegram=True, rss=False).alerts, 0)

    def test_items_published_long_ago_are_not_classified(self) -> None:
        self.set_posts([])
        self.radar.poll(telegram=True, rss=False)
        self.set_posts([(5, "repost of last week", NOW - timedelta(hours=2))])
        self.assertEqual(self.radar.poll(telegram=True, rss=False).new_items, 0)
        self.assertEqual(self.classifier.calls, [])

    def test_confirmed_bearish_shock_brakes_and_rumour_does_not(self) -> None:
        self.set_posts([])
        self.radar.poll(telegram=True, rss=False)
        self.classifier.result = Classification(True, "bearish", 3, False, False, "слух о взломе")
        self.set_posts([(7, "rumour", NOW)])
        self.radar.poll(telegram=True, rss=False)
        self.assertIsNone(active_brake(NOW, self.store.path))
        self.classifier.result = Classification(True, "bearish", 3, True, False, "биржа взломана")
        self.set_posts([(7, "rumour", NOW), (8, "exchange hacked", NOW)])
        self.radar.poll(telegram=True, rss=False)
        self.assertIn("биржа взломана", active_brake(NOW, self.store.path))
        self.assertIsNone(active_brake(NOW + timedelta(hours=3), self.store.path))
        self.assertTrue(any("Тормоз" in m for m in self.messages))

    def test_classifier_failure_is_recorded_and_not_retried(self) -> None:
        self.set_posts([])
        self.radar.poll(telegram=True, rss=False)
        self.classifier.fail = True
        self.set_posts([(9, "x", NOW)])
        result = self.radar.poll(telegram=True, rss=False)
        self.assertIn("DeepSeek", result.errors[0])
        self.classifier.fail = False
        self.radar.poll(telegram=True, rss=False)
        self.assertEqual(len(self.classifier.calls), 1)

    def test_prices_and_paper_stats(self) -> None:
        self.set_posts([])
        self.radar.poll(telegram=True, rss=False)
        self.set_posts([(10, "USHA adopts BTC", NOW - timedelta(seconds=30))])
        self.radar.poll(telegram=True, rss=False)
        base = int(datetime(2026, 10, 19, 11, 59, tzinfo=timezone.utc).timestamp() * 1000)
        closes = {-0: "79600", 1: "80000", 2: "80400", 6: "80800", 31: "79900", 61: "80200"}
        self.candles = [[str(base + m * 60_000), "0", "0", "0", c, "0", "0", "0", "1"] for m, c in closes.items()]
        self.radar.update_prices()
        stats = paper_stats(self.store.classified())
        self.assertEqual(stats.alerts, 1)
        self.assertEqual(stats.median_delay_s, 30)
        self.assertEqual(stats.pre_move_pct.quantize(D("0.001")), D("0.503"))   # 79600 -> 80000
        n, avg, win = stats.horizons[1]
        self.assertEqual((n, avg.quantize(D("0.01")), win), (1, D("0.30"), 100))  # +0.5% - 0.2%
        n, avg, win = stats.horizons[30]
        self.assertEqual((avg.quantize(D("0.001")), win), (D("-0.325"), 0))      # -0.125% - 0.2%


class BrakeInTradersTests(DemoTraderTests):
    def trader(self):
        trader = super().trader()
        trader.brake = lambda _now: "биржа взломана"
        return trader

    def test_buy_is_blocked_by_news_brake(self) -> None:
        report = self.trader().run_once()
        self.assertEqual(report.status, "HOLD")
        self.assertIn("тормоз по новостям", " ".join(report.lines))
        self.assertFalse(any(c[0] == "POST" for c in self.okx.calls))

    def test_sell_signal_still_closes_under_brake(self) -> None:
        trader = DemoTraderTests.trader(self)
        trader.run_once()
        self.advance(Signal.SELL)
        self.assertEqual(self.trader().run_once().status, "CLOSED")

    # Inherited scenarios that need a BUY are not meaningful under a permanent brake.
    test_buy_is_sized_by_risk_and_protected_by_oco = None
    test_take_profit_closes_with_profit = None
    test_stop_loss_loss_is_bounded = None
    test_sell_signal_cancels_oco_and_sells_only_bot_btc = None
    test_failed_protection_sells_immediately_and_halts = None
    test_lost_buy_response_halts_then_resume_protects_the_fill = None
    test_rejected_buy_spends_nothing = None
    test_same_candle_is_acted_on_once = None
    test_daily_entry_limit = None
    test_externally_canceled_oco_halts = None
    test_abandon_after_manual_close = None
    test_owner_close_now = None


if __name__ == "__main__":
    unittest.main()
