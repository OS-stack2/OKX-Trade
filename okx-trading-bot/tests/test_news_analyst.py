from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
import json
import unittest

from bot.news_analyst import (BUY_NOW, SKIP, WAIT, AnalystDecision, MarketContext, NewsAnalyst,
                              apply_hard_rules, build_context, describe_ru, parse_decision)
from bot.news_radar import NewsError
from bot.news_trader import AnalystCases
from tests.test_news_trader import NewsTraderTests


NOW = datetime(2026, 10, 16, 12, 1, tzinfo=timezone.utc)


def ctx(**kwargs) -> MarketContext:
    return MarketContext(**{"price": 80000.0, "minutes_since_news": 2.0, "pre_move_pct": 0.2, **kwargs})


class DecisionTests(unittest.TestCase):
    def test_parse_reply_with_text_around_json(self) -> None:
        d = parse_decision('Думаю... {"decision": "WAIT", "confidence": 70, "flags": ["overheated", "bogus"],'
                           ' "reasons_ru": ["перегрев"], "recheck_minutes": 25}')
        self.assertEqual((d.decision, d.flags, d.recheck_minutes), (WAIT, ("overheated",), 30))
        self.assertEqual(parse_decision('{"decision":"BUY_NOW","confidence":90,"hold_minutes":999}').hold_minutes, 240)
        for bad in ("no json", '{"decision": "MAYBE"}', '{"decision": "SKIP", "confidence": "x"}'):
            with self.assertRaises(NewsError):
                parse_decision(bad)

    def test_hard_rules(self) -> None:
        buy = AnalystDecision(BUY_NOW, 80, (), ("сильная новость",))
        self.assertEqual(apply_hard_rules(buy, ctx()).decision, BUY_NOW)
        chased = apply_hard_rules(buy, ctx(pre_move_pct=1.8))
        self.assertEqual((chased.decision, chased.vetoed), (SKIP, True))
        self.assertIn("priced_in", chased.flags)
        self.assertEqual(apply_hard_rules(buy, ctx(minutes_since_news=75)).decision, SKIP)
        unsure = apply_hard_rules(AnalystDecision(BUY_NOW, 40, (), ()), ctx())
        self.assertEqual(unsure.decision, WAIT)
        # Staleness wins over low confidence.
        self.assertEqual(apply_hard_rules(AnalystDecision(BUY_NOW, 40, (), ()), ctx(minutes_since_news=90)).decision,
                         SKIP)

    def test_describe_ru(self) -> None:
        text = describe_ru(AnalystDecision(SKIP, 78, ("sell_the_news",), ("ожидаемое событие",)))
        self.assertIn("ПРОПУСТИТЬ (уверенность 78%)", text)
        self.assertIn("покупай на слухах", text)

    def test_analyst_request_uses_reasoner_and_context(self) -> None:
        seen = {}

        def transport(request, timeout):
            seen["body"] = json.loads(request.data)
            return {"choices": [{"message": {"content": '{"decision":"SKIP","confidence":80,"reasons_ru":["x"]}'}}]}

        decision = NewsAnalyst("k", transport=transport).evaluate(
            headline="USHA adopts BTC", summary="s", source="tg:x", impact=3, context=ctx(funding_pct=0.05),
            previous=("12:00 UTC: WAIT (55%)",))
        self.assertEqual(decision.decision, SKIP)
        self.assertEqual(seen["body"]["model"], "deepseek-reasoner")
        user = seen["body"]["messages"][1]["content"]
        self.assertIn("Perpetual funding per 8h: +0.0500%", user)
        self.assertIn("earlier decisions", user)


class FakePublic:
    def get_hourly_candles(self, limit):
        # newest first; price drifted from 79000 up to 80000
        return [[str(i), "0", str(80500 - i * 5), "0", str(80000 - i * 5), "0", "0", "0", "1"] for i in range(168)]

    def get_funding_rate(self):
        return [{"fundingRate": "0.0003"}]

    def get_open_interest_history(self):
        return [[str(i), str(1100 if i == 0 else 1000), "0"] for i in range(30)]

    def get_long_short_ratio(self):
        raise TimeoutError("down")

    def get_taker_volume(self):
        return [[str(i), "40", "60"] for i in range(10)]

    def get_daily_candles(self, limit):
        raise TimeoutError("down")


class ContextTests(unittest.TestCase):
    def test_context_with_partial_data(self) -> None:
        c = build_context(FakePublic(), price=80000.0, published_at=NOW - timedelta(minutes=3), now=NOW,
                          price_before_news=79800.0, recent_news=("10:00 UTC [bullish, impact 2] x",),
                          fear_greed=lambda: 72)
        self.assertAlmostEqual(c.change_pct["1h"], (80000 / 79995 - 1) * 100)
        self.assertAlmostEqual(c.funding_pct, 0.03)
        self.assertAlmostEqual(c.oi_change_24h_pct, 10.0)
        self.assertAlmostEqual(c.taker_buy_share_4h, 0.6)
        self.assertIsNone(c.long_short)
        self.assertIsNone(c.trend_share)
        text = c.describe()
        self.assertIn("Long/short account ratio: n/a", text)
        self.assertIn("Move since publication (toward the news): +0.25%", text)
        self.assertIn("с новости +0.25%", c.short_ru())
        self.assertIn("страх/жадность 72", c.short_ru())
        self.assertIn("Crypto Fear & Greed index: 72/100", text)


class ScriptedAnalyst:
    def __init__(self, *decisions: AnalystDecision) -> None:
        self.decisions = list(decisions)
        self.calls = 0

    def evaluate(self, **kwargs) -> AnalystDecision:
        self.calls += 1
        item = self.decisions.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class AnalystFlowTests(NewsTraderTests):
    """The news trader with the analyst between headline and order."""

    def setUp(self) -> None:
        super().setUp()
        self.analyst = ScriptedAnalyst()
        self.trader.analyst = self.analyst
        self.trader.context_builder = lambda case, now: ctx()
        self.trader.cases = AnalystCases(self.store.path)

    def cases(self):
        return self.trader.cases.all()

    def test_buy_now_opens_with_analyst_hold_time(self) -> None:
        self.analyst.decisions = [AnalystDecision(BUY_NOW, 85, (), ("сюрприз",), hold_minutes=90)]
        self.publish()
        self.assertEqual(self.trader.step().status, "OPENED")
        [case] = self.cases()
        self.assertEqual((case["state"], case["hold_minutes"]), ("BOUGHT", 90))
        self.assertIn("🧠 Аналитик: ПОКУПАТЬ", self.messages[0])
        self.now += timedelta(minutes=45)
        self.assertIsNone(self.trader.step())               # 30-min default no longer applies
        self.now += timedelta(minutes=46)
        self.assertEqual(self.trader.step().status, "CLOSED")

    def test_wait_then_buy_on_recheck(self) -> None:
        self.analyst.decisions = [AnalystDecision(WAIT, 55, ("overheated",), ("ждём отката",), recheck_minutes=15),
                                  AnalystDecision(BUY_NOW, 75, (), ("откат случился",))]
        self.publish()
        self.assertEqual(self.trader.step().status, "HOLD")
        self.assertIsNone(self.store.active_position())
        self.assertIn("Перепроверю", self.messages[0])
        self.now += timedelta(minutes=10)
        self.assertIsNone(self.trader.step())
        self.now += timedelta(minutes=6)
        self.assertEqual(self.trader.step().status, "OPENED")
        [case] = self.cases()
        self.assertEqual((case["state"], case["checks"], len(json.loads(case["decisions"]))), ("BOUGHT", 2, 2))

    def test_skip_and_expire_never_buy(self) -> None:
        self.analyst.decisions = [AnalystDecision(SKIP, 80, ("sell_the_news",), ("ожидали",))]
        self.publish()
        self.trader.step()
        self.analyst.decisions = [AnalystDecision(WAIT, 50, (), ("рано",))] * 3
        self.publish()
        for _ in range(3):
            self.trader.step()
            self.now += timedelta(minutes=16)
        self.assertEqual([c["state"] for c in self.cases()], ["SKIPPED", "EXPIRED"])
        self.assertFalse(any(c[0] == "POST" for c in self.okx.calls))
        self.assertIn("момент так и не наступил", self.messages[-1])

    def test_analyst_outage_retries_then_expires(self) -> None:
        self.analyst.decisions = [NewsError("Аналитик DeepSeek не ответил: TimeoutError")] * 3
        self.publish()
        self.assertEqual(self.trader.step().status, "ERROR")
        self.now += timedelta(minutes=2)
        self.trader.step()
        self.now += timedelta(minutes=2)
        self.trader.step()
        self.assertEqual(self.cases()[0]["state"], "EXPIRED")

    def test_open_position_blocks_new_case(self) -> None:
        self.analyst.decisions = [AnalystDecision(BUY_NOW, 90, (), ("да",))]
        self.publish()
        self.trader.step()
        self.publish()
        self.assertEqual(self.trader.step().status, "HOLD")
        self.assertEqual([c["state"] for c in self.cases()], ["BOUGHT", "BLOCKED"])
        self.assertEqual(self.analyst.calls, 1)

    # Inherited immediate-buy scenarios do not apply when the analyst decides.
    test_bullish_news_buys_20_percent_with_oco = None
    test_time_exit_after_hold = None
    test_oco_take_profit_before_time = None
    test_one_position_at_a_time_and_news_not_queued = None
    test_brake_blocks_news_buy = None
    test_halt_message_names_news_resume_command = None


if __name__ == "__main__":
    unittest.main()


class SummaryTests(unittest.TestCase):
    def test_analyst_vs_naive_counterfactual(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from bot.news_radar import Classification, NewsItem, NewsStore
        from bot.news_trader import NewsSignal, analyst_summary
        with TemporaryDirectory() as tmp:
            radar = NewsStore(Path(tmp) / "r.sqlite3")
            cases = AnalystCases(Path(tmp) / "c.sqlite3")
            for n, state in ((1, "SKIPPED"), (2, "BOUGHT")):
                key = f"tg:x/{n}"
                radar.add([NewsItem("tg:x", key, "u", NOW, "t")], NOW, primed=True)
                radar.save_classification(key, Classification(True, "bullish", 3, True, False, "s"))
                radar.set_prices(key, price_seen=D("80000"), p30=D("79200"), p60=D("80800"))
                cases.create(NewsSignal(n, NOW, "s", 3, "tg:x", "u", key, "t", NOW), NOW)
                [case] = [c for c in cases.all() if c["news_key"] == key]
                cases.update(case["id"], state=state, position_id=7 if state == "BOUGHT" else None)
            lines = analyst_summary(cases.all(), {7: D("3.5")}, radar.path)
        text = "\n".join(lines)
        self.assertIn("купил 1", text)
        self.assertIn("пропустил 1", text)
        self.assertIn("PnL +3.50", text)
        self.assertIn("через 30 мин: в среднем -1.200%", text)   # avoided a -1% move (+0.2% costs)
        self.assertIn("через 60 мин: в среднем +0.800%", text)
