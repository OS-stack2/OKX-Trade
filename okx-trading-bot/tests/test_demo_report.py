from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from bot.demo_report import build_report
from bot.demo_trader import TraderStore
from bot.trend_trader import TrendStore


class DemoReportTests(unittest.TestCase):
    def test_report_summarizes_both_bots(self) -> None:
        with TemporaryDirectory() as tmp:
            now = datetime(2026, 10, 23, tzinfo=timezone.utc)
            trader = TraderStore(Path(tmp) / "t.sqlite3")
            position = trader.create_opening(candle_closed_at=now - timedelta(days=1), client_id="DT" + "a" * 30,
                                             planned_btc=D("0.01"), now=now - timedelta(days=1))
            trader.update(position.id, state="CLOSED", net_btc=D("0.01"), closed_at=now.isoformat(),
                          pnl_usdt=D("12.5"), exit_reason="TAKE_PROFIT")
            trend = TrendStore(Path(tmp) / "r.sqlite3")
            trend.book(D("100"))
            with trend._db() as db:
                db.execute("INSERT INTO trades(at, day, side, btc, price, fee, fee_ccy, target, equity) "
                           "VALUES(?, '2026-10-16', 'buy', '0.001', '80000', '0', 'BTC', '1', '100')",
                           ((now - timedelta(days=6)).isoformat(),))
            trend.set(usdt=D("20"), btc=D("0.001"))
            text = build_report(trader=trader, trend=trend, trend_budget=D("100"), price=D("88000"),
                                now=now, final=True)
        self.assertIn("ИТОГ 2 МЕСЯЦЕВ", text)
        self.assertIn("PnL всего +12.50", text)
        self.assertIn("TAKE_PROFIT 1", text)
        self.assertIn("капитал 108.00 USDT (+8.0%)", text)
        self.assertIn("за неделю 1", text)


if __name__ == "__main__":
    unittest.main()


class CriteriaTests(unittest.TestCase):
    def test_final_report_applies_pre_registered_rules(self) -> None:
        from bot.criteria import describe, evaluate
        from bot.news_radar import Classification, NewsItem, NewsStore
        with TemporaryDirectory() as tmp:
            now = datetime(2026, 12, 16, 9, tzinfo=timezone.utc)
            news_trader = TraderStore(Path(tmp) / "n.sqlite3")
            for i in range(10):
                pos = news_trader.create_opening(candle_closed_at=now - timedelta(days=i), client_id=f"DT{i:030x}",
                                                 planned_btc=D("0.002"), now=now)
                news_trader.update(pos.id, state="CLOSED", net_btc=D("0.002"), entry_cost_usdt=D("200"),
                                   closed_at=now.isoformat(), pnl_usdt=D("1"), exit_reason="TIME")
            radar = NewsStore(Path(tmp) / "r.sqlite3")
            radar.add([NewsItem("tg:x", "tg:x/1", "u", now, "t")], now, primed=True)
            radar.save_classification("tg:x/1", Classification(True, "bullish", 3, True, False, "s"))
            radar.set_prices("tg:x/1", price_seen=D("80000"), p30=D("80080"))   # +0.1% - 0.2% = -0.1%
            trend = TrendStore(Path(tmp) / "t.sqlite3")
            trend.book(D("100"))
            ok = evaluate(news_trader=news_trader, radar_path=radar.path, trend=trend, suspicious=0)
            self.assertTrue(all(c.ok for c in ok), ok)
            text = "\n".join(describe(ok, final=True))
            self.assertIn("автоторговля на реальном счёте допустима", text)
            self.assertIn("аналитик +0.500% на сделку против -0.100%", text)
            trend.set(halts_total="1")
            failed = describe(evaluate(news_trader=news_trader, radar_path=radar.path, trend=trend,
                                       suspicious=2), final=True)
            text = "\n".join(failed)
            self.assertIn("❌ B1", text)
            self.assertIn("❌ B2: расхождений учёта (SUSPICIOUS с 19.10): 2", text)
            self.assertIn("тренд-бот: НЕ кандидат", text)
