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
