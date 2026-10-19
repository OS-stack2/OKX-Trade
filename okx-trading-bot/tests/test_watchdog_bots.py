"""Watchdog stays quiet only for activity fully explained by the Demo bot journals."""

from __future__ import annotations

from decimal import Decimal as D
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from bot.bot_ledger import BotHoldings, read_bot_holdings
from bot.demo_client import CurrencyBalance, DemoAlgoOrder
from bot.demo_trader import TraderStore
from bot.trend_trader import TrendStore
from bot.watchdog import Watchdog, WatchdogStore
from tests.test_watchdog import NOW, FakeAlerts, FakeDemo, FakePublic


class WatchdogBotLedgerTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.demo = FakeDemo()
        self.alerts = FakeAlerts()
        self.holdings = BotHoldings(D("0"), frozenset(), D("1"))
        self.runner = Watchdog(store=WatchdogStore(self.root / "watchdog.sqlite3"), demo_loader=lambda: self.demo,
                               public_factory=lambda _c: FakePublic(), alerter_loader=lambda: self.alerts,
                               clock=lambda: NOW, ledger_loader=lambda: self.holdings)

    def bot_buys(self, btc: str, algo: str | None = None) -> None:
        total = D("1") + D(btc)
        self.demo.btc = CurrencyBalance(total, D("1") if algo else total)
        self.demo.usdt = CurrencyBalance(D("4000"), D("4000"))
        if algo:
            self.demo.algos = (DemoAlgoOrder(algo, "oco", "live", D(btc), "sell"),)
        self.holdings = BotHoldings(D(btc), frozenset({algo} if algo else set()), D("1"))

    def test_bot_trade_with_its_oco_is_silent(self):
        self.runner.run_once()
        self.bot_buys("0.012", "A1")
        report = self.runner.run_once()
        self.assertEqual(report.status, "OBSERVED_UNVERIFIED", report.reasons)
        self.assertEqual(report.exit_code, 0)
        self.assertEqual(self.alerts.events, [])
        self.assertIn("объяснено", " ".join(report.reasons))

    def test_unexplained_btc_still_alerts(self):
        self.runner.run_once()
        self.bot_buys("0.012", "A1")
        self.demo.btc = CurrencyBalance(D("1.5"), D("1"))
        self.assertEqual(self.runner.run_once().status, "SUSPICIOUS")
        self.assertEqual(self.alerts.events, ["WATCHDOG_SUSPICIOUS"])

    def test_foreign_algo_still_alerts(self):
        self.runner.run_once()
        self.bot_buys("0.012", "A1")
        self.holdings = BotHoldings(D("0.012"), frozenset({"OTHER"}), D("1"))
        self.assertEqual(self.runner.run_once().status, "SUSPICIOUS")

    def test_bot_in_flight_is_not_explained(self):
        self.runner.run_once()
        self.bot_buys("0.012")
        self.holdings = None
        self.assertEqual(self.runner.run_once().status, "SUSPICIOUS")

    def test_ledger_reads_both_journals_read_only(self):
        env = self.root / ".env"
        env.write_text(f"OKX_TRADER_STATE_PATH={self.root / 't.sqlite3'}\n"
                       f"OKX_TREND_STATE_PATH={self.root / 'r.sqlite3'}\n", encoding="utf-8")
        self.assertIsNone(read_bot_holdings(env))
        trader = TraderStore(self.root / "t.sqlite3")
        position = trader.create_opening(candle_closed_at=NOW, client_id="DT" + "b" * 30,
                                         planned_btc=D("0.01"), now=NOW)
        self.assertIsNone(read_bot_holdings(env))  # order in flight
        trader.update(position.id, state="OPEN", net_btc=D("0.00999"), algo_id="A9")
        trend = TrendStore(self.root / "r.sqlite3")
        trend.book(D("100"))
        trend.set(btc=D("0.001"))
        holdings = read_bot_holdings(env)
        self.assertEqual(holdings.btc, D("0.01099"))
        self.assertEqual(holdings.algo_ids, frozenset({"A9"}))
        with env.open("a", encoding="utf-8") as handle:
            handle.write(f"OKX_NEWSTRADER_STATE_PATH={self.root / 'n.sqlite3'}\n")
        news = TraderStore(self.root / "n.sqlite3")
        opened = news.create_opening(candle_closed_at=NOW, client_id="DT" + "c" * 30,
                                     planned_btc=D("0.002"), now=NOW)
        news.update(opened.id, state="OPEN", net_btc=D("0.002"), algo_id="N1")
        holdings = read_bot_holdings(env)
        self.assertEqual(holdings.btc, D("0.01299"))
        self.assertEqual(holdings.algo_ids, frozenset({"A9", "N1"}))
        self.assertEqual(holdings.external_btc, D("1"))


if __name__ == "__main__":
    unittest.main()
