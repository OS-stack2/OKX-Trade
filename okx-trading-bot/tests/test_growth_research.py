"""Offline historical research uses disposable CSVs and synthetic candles only."""

from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
from io import StringIO
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest
import urllib.request
from unittest.mock import patch

import main
from bot.growth_research import (
    ResearchConfig, ResearchError, VARIANTS, _indicators, _signal_at,
    load_candles_csv, okx_taker_debit_rate, research, run_backtest, validate_candles,
)
from bot.market_data import Candle, SpotInstrument
from bot.risk_manager import RiskSettings


START = datetime(2024, 1, 1, tzinfo=timezone.utc)


def candles_for(closes):
    return tuple(Candle(START + timedelta(minutes=15 * i), D(str(price)), D(str(price)) * D("1.001"),
                        D(str(price)) * D("0.999"), D(str(price)), True)
                 for i, price in enumerate(closes))


def config(*, fee="0.001", slip="0.001", min_size="0.0001", max_position="5000"):
    instrument = SpotInstrument("BTC-USDT", D("0.01"), D("0.0001"), D(min_size),
                                "live", "BTC", "USDT", max_market_size=D("1000000"))
    return ResearchConfig(instrument, D(fee), D(slip), max_position_usdt=D(max_position))


class GrowthResearchTests(unittest.TestCase):
    def test_csv_requires_confirmed_ordered_utc_ohlc_and_no_missing_bars(self):
        candles = candles_for([100] * 110)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "history.csv"
            lines = ["opened_at,open,high,low,close,confirm"]
            for candle in candles:
                lines.append(f"{candle.opened_at.isoformat()},{candle.open},{candle.high},"
                             f"{candle.low},{candle.close},1")
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.assertEqual(load_candles_csv(path), candles)
            path.write_text("\n".join(lines[:-1] + [lines[-1][:-1] + "0"]) + "\n", encoding="utf-8")
            with self.assertRaises(ResearchError):
                load_candles_csv(path)
            path.write_text("\n".join(lines[:10] + lines[11:]) + "\n", encoding="utf-8")
            with self.assertRaises(ResearchError):
                load_candles_csv(path)
        bad = list(candles)
        bad[25] = replace(bad[25], high=D("1"))
        with self.assertRaises(ResearchError):
            validate_candles(tuple(bad))

    def test_ema_baseline_cross_matches_existing_strategy_and_uses_no_future(self):
        prefix = [100] * 21 + [110] + [110] * 48
        first = candles_for(prefix + [60] * 50)
        second = candles_for(prefix + [200] * 50)
        self.assertEqual(_signal_at(21, first, _indicators(first), "EMA_9_21"), "BUY")
        for index in range(21, 70):
            self.assertEqual(_signal_at(index, first, _indicators(first), "EMA_9_21"),
                             _signal_at(index, second, _indicators(second), "EMA_9_21"))

    def test_filter_variants_are_distinct_and_train_does_not_read_final(self):
        prices = [100] * 21 + [110] + [110] * 98
        candles = candles_for(prices)
        ind = _indicators(candles)
        self.assertEqual(_signal_at(21, candles, ind, "EMA_9_21"), "BUY")
        self.assertEqual(_signal_at(21, candles, ind, "EMA_TREND"), "HOLD")
        self.assertEqual(set(VARIANTS), {"EMA_9_21", "EMA_TREND", "EMA_VOLATILITY",
                                         "EMA_STRENGTH", "EMA_ALL_FILTERS"})
        report = research(candles, config())
        self.assertEqual(report.split_indices, (0, 72, 96, 120))
        self.assertEqual(len(report.training), 5)
        self.assertEqual(len(report.validation), 5)
        self.assertEqual(report.final_test.variant, report.selected_variant)
        self.assertIn("достаточных", report.recommendation.lower())
        self.assertTrue(report.warnings)

    def test_final_holdout_prices_cannot_change_variant_selected_earlier(self):
        prices = [100] * 21 + [110] * 20 + [90] * 20 + [100] * 59
        original = candles_for(prices)
        changed = candles_for(prices[:96] + [150] * 24)
        self.assertEqual(research(original, config()).selected_variant,
                         research(changed, config()).selected_variant)

    def test_next_open_execution_and_fees_slippage_reduce_returns(self):
        prices = [100] * 120
        prices[62] = 110
        candles = candles_for(prices)
        def forced(index, *_):
            return "BUY" if index == 60 else "SELL" if index == 61 else "HOLD"
        with patch("bot.growth_research._signal_at", side_effect=forced):
            free = run_backtest(candles, 0, 120, "EMA_9_21", config(fee="0", slip="0"))
            costly = run_backtest(candles, 0, 120, "EMA_9_21", config(fee="0.01", slip="0.01"))
        self.assertEqual(free.entry_indices, (61,))
        self.assertEqual(free.trades, 1)
        self.assertGreater(free.return_pct, costly.return_pct)

    def test_minimum_lot_and_position_cap(self):
        candles = candles_for([100] * 120)
        with patch("bot.growth_research._signal_at", side_effect=lambda i, *_: "BUY" if i == 60 else "HOLD"):
            tiny = run_backtest(candles, 0, 120, "EMA_9_21", config(min_size="100"))
            capped = run_backtest(candles, 0, 120, "EMA_9_21", config(max_position="100"))
        self.assertEqual(tiny.entries, 0)
        self.assertEqual(tiny.skipped_min_size, 1)
        self.assertLessEqual(capped.max_entry_notional_usdt, D("100"))
        self.assertEqual(capped.entries, 1)
        restricted = replace(config(), instrument=replace(config().instrument, max_market_size=D("0.5")))
        with patch("bot.growth_research._signal_at", side_effect=lambda i, *_: "BUY" if i == 60 else "HOLD"):
            max_size = run_backtest(candles, 0, 120, "EMA_9_21", restricted)
        self.assertLessEqual(max_size.max_entry_notional_usdt, D("0.5"))

    def test_spot_market_buy_cap_is_quote_notional_and_buy_hold_obeys_it(self):
        candles = candles_for([100] * 119 + [200])
        restricted = replace(config(fee="0", slip="0"),
                             instrument=replace(config().instrument, max_market_size=D("20")))
        with patch("bot.growth_research._signal_at", side_effect=lambda i, *_: "BUY" if i == 60 else "HOLD"):
            result = run_backtest(candles, 0, 120, "EMA_9_21", restricted)
        self.assertEqual(result.entry_quantities, (D("0.2"),))
        self.assertEqual(result.max_entry_notional_usdt, D("20"))
        self.assertLessEqual(result.buy_hold_return_pct, D("0.4"))
        # Different units: this is a valid 0.01 USDT cap, despite minSz=0.1 BTC.
        mixed_units = replace(config(), instrument=replace(config().instrument,
                              min_size=D("0.1"), max_market_size=D("0.01")))
        mixed_units.validate()

    def test_negative_okx_taker_rate_is_positive_model_cost_and_rebates_block(self):
        self.assertEqual(okx_taker_debit_rate(D("-0.001")), D("0.001"))
        self.assertEqual(okx_taker_debit_rate(D("0")), D("0"))
        for value in (D("0.0001"), D("NaN"), D("-1")):
            with self.subTest(value=value), self.assertRaises(ResearchError):
                okx_taker_debit_rate(value)
        candles = candles_for([100] * 120)
        with patch("bot.growth_research._signal_at", side_effect=lambda i, *_: "BUY" if i == 60 else "HOLD"):
            free = run_backtest(candles, 0, 120, "EMA_9_21", config(fee="0", slip="0"))
            charged = run_backtest(candles, 0, 120, "EMA_9_21",
                                   config(fee=str(okx_taker_debit_rate(D("-0.001"))), slip="0"))
        self.assertLess(charged.return_pct, free.return_pct)

    def test_research_rejects_changed_risk_settings_and_multiple_positions(self):
        with self.assertRaises(ResearchError):
            replace(config(), risk=RiskSettings(risk_per_trade=D("0.01"))).validate()
        candles = candles_for([100] * 120)
        with patch("bot.growth_research._signal_at",
                   side_effect=lambda i, *_: "BUY" if i in (21, 22, 23) else "SELL" if i == 24 else "HOLD"):
            result = run_backtest(candles, 0, 120, "EMA_9_21", config(fee="0", slip="0"))
        self.assertEqual(result.entries, 1)

    def test_same_bar_stop_and_take_choose_stop(self):
        candles = list(candles_for([100] * 120))
        candles[22] = replace(candles[22], high=D("103"), low=D("98"))
        with patch("bot.growth_research._signal_at", side_effect=lambda i, *_: "BUY" if i == 21 else "HOLD"):
            result = run_backtest(tuple(candles), 0, 120, "EMA_9_21", config(fee="0", slip="0"))
        self.assertEqual(result.trades, 1)
        self.assertLess(result.return_pct, 0)
        self.assertGreaterEqual(result.return_pct, D("-0.5"))

    def test_profit_reinvestment_is_bounded_by_hard_position_cap(self):
        prices = [100] * 120
        prices[23] = 110
        prices[33] = 110
        candles = candles_for(prices)
        def forced(index, *_):
            return "BUY" if index in (21, 31) else "SELL" if index in (22, 32) else "HOLD"
        with patch("bot.growth_research._signal_at", side_effect=forced):
            result = run_backtest(candles, 0, 120, "EMA_9_21", config(fee="0", slip="0", max_position="3000"))
        self.assertEqual(result.entries, 2)
        self.assertGreater(result.entry_quantities[1], result.entry_quantities[0])
        self.assertLessEqual(result.max_entry_notional_usdt, D("3000"))

    def test_utc_daily_entry_cap_and_loss_lock(self):
        candles = candles_for([100] * 120)
        def alternating(index, *_):
            return "BUY" if index % 3 == 0 else "SELL" if index % 3 == 1 else "HOLD"
        with patch("bot.growth_research._signal_at", side_effect=alternating):
            result = run_backtest(candles, 0, 96, "EMA_9_21", config(fee="0", slip="0"))
        self.assertEqual(result.entries, 5)
        self.assertGreater(result.blocked_daily_limit, 0)
        self.assertGreater(result.blocked_entry_limit, 0)
        self.assertEqual(result.trades, 5)

    def test_gap_loss_trips_daily_loss_gate_for_next_buy(self):
        candles = list(candles_for([100] * 120))
        candles[23] = replace(candles[23], open=D("80"), low=D("79"), high=D("101"))
        with patch("bot.growth_research._signal_at",
                   side_effect=lambda i, *_: "BUY" if i in (21, 24, 27) else "HOLD"):
            result = run_backtest(tuple(candles), 0, 96, "EMA_9_21", config(fee="0", slip="0"))
        self.assertEqual(result.entries, 1)
        self.assertGreater(result.blocked_loss_limit, 0)
        self.assertLess(result.return_pct, D("-2"))

    def test_gap_and_stress_scenarios_can_exceed_planned_risk(self):
        prices = [100] * 150
        candles = list(candles_for(prices))
        candles[62] = replace(candles[62], open=D("80"), low=D("79"), close=D("80"), high=D("101"))
        with patch("bot.growth_research._signal_at", side_effect=lambda i, *_: "BUY" if i == 60 else "HOLD"):
            result = run_backtest(tuple(candles), 0, 150, "EMA_9_21", config(fee="0", slip="0"))
            report = research(candles_for([100] * 150), config())
        self.assertLess(result.return_pct, D("-0.5"))
        self.assertEqual(result.trades, 1)
        self.assertEqual(len(report.stresses), 4)
        self.assertTrue(any("последовательных стопов" in item.name for item in report.stresses))

    def test_fee_slippage_gap_and_loss_streak_stresses_are_reported(self):
        prices = [100] * 300
        prices[247] = 110
        candles = candles_for(prices)
        with patch("bot.growth_research._signal_at",
                   side_effect=lambda i, *_: "BUY" if i == 245 else "SELL" if i == 246 else "HOLD"):
            report = research(candles, config(fee="0.001", slip="0.001"))
        self.assertEqual(report.final_test.entries, 1)
        self.assertLess(report.stresses[0].return_pct, report.final_test.return_pct)
        self.assertLess(report.stresses[1].return_pct, report.final_test.return_pct)
        self.assertLess(report.stresses[2].return_pct, report.final_test.return_pct)
        self.assertLessEqual(int(report.stresses[3].note.split(": ")[1].split(";")[0]), 5)

    def test_cli_uses_csv_only_and_does_not_call_okx_or_sqlite(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "candles.csv"
            lines = ["opened_at,open,high,low,close,confirm"]
            for c in candles_for([100] * 120):
                lines.append(f"{c.opened_at.isoformat()},{c.open},{c.high},{c.low},{c.close},1")
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            before = path.read_bytes()
            output = StringIO()
            with (patch.object(main.DemoClient, "from_env_file", side_effect=AssertionError("OKX called")),
                  patch.object(main, "DecisionJournal", side_effect=AssertionError("SQLite called")),
                  patch.object(sqlite3, "connect", side_effect=AssertionError("SQLite opened")),
                  patch.object(urllib.request, "urlopen", side_effect=AssertionError("HTTP called")),
                  redirect_stdout(output)):
                code = main.main(["--growth-research", str(path), "--research-min-sz", "0.0001",
                                  "--research-lot-sz", "0.0001", "--research-tick-sz", "0.01",
                                  "--research-max-mkt-sz", "100", "--research-taker-fee", "0.001",
                                  "--research-slippage", "0.001"])
            self.assertEqual(code, 0)
            self.assertEqual(path.read_bytes(), before)
            self.assertIn("FINAL_TEST UTC:", output.getvalue())
            self.assertIn("ПРЕДУПРЕЖДЕНИЕ", output.getvalue())
            self.assertIn("торговля отключена", output.getvalue())

    def test_cli_accepts_signed_okx_debit_without_touching_sqlite(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "candles.csv"
            path.write_text("opened_at,open,high,low,close,confirm\n" + "".join(
                f"{c.opened_at.isoformat()},{c.open},{c.high},{c.low},{c.close},1\n"
                for c in candles_for([100] * 120)), encoding="utf-8")
            output = StringIO()
            with patch.object(sqlite3, "connect", side_effect=AssertionError("SQLite opened")), redirect_stdout(output):
                result = main.main(["--growth-research", str(path), "--research-min-sz", "0.00001",
                                    "--research-lot-sz", "0.00000001", "--research-tick-sz", "0.1",
                                    "--research-max-mkt-sz", "1000000",
                                    "--research-okx-taker-rate=-0.001", "--research-slippage", "0.001"])
            self.assertEqual(result, 0)
            self.assertIn("taker 0.001", output.getvalue())
            self.assertIn("maxMktSz 1000000 USDT", output.getvalue())


if __name__ == "__main__":
    unittest.main()
