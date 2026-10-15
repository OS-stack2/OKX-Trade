from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from io import StringIO
import unittest
from unittest.mock import patch

import main
from bot.market_data import Candle, MarketSnapshot, SpotInstrument
from bot.strategy import Signal, StrategyError, calculate_ema, evaluate_signal


START = datetime(2024, 1, 1, tzinfo=timezone.utc)


def candles_for(prices: list[int]) -> list[Candle]:
    return [
        Candle(
            opened_at=START + timedelta(minutes=15 * index),
            open=Decimal(price),
            high=Decimal(price),
            low=Decimal(price),
            close=Decimal(price),
            confirmed=True,
        )
        for index, price in enumerate(prices)
    ]


class StrategyTests(unittest.TestCase):
    def test_ema_uses_initial_sma_then_standard_multiplier(self) -> None:
        prices = [Decimal(value) for value in range(1, 23)]
        ema9 = calculate_ema(prices, 9)
        ema21 = calculate_ema(prices, 21)
        self.assertEqual(ema9[0], Decimal(5))
        self.assertEqual(ema9[-2:], (Decimal(17), Decimal(18)))
        self.assertEqual(ema21, (Decimal(11), Decimal(12)))

    def test_buy_on_crossover_from_below(self) -> None:
        result = evaluate_signal(candles_for([100] * 20 + [90, 120]))
        self.assertEqual(result.signal, Signal.BUY)
        self.assertLess(result.previous_ema9, result.previous_ema21)
        self.assertGreater(result.ema9, result.ema21)
        self.assertEqual(result.candle_closed_at, START + timedelta(minutes=15 * 22))
        self.assertIn("снизу вверх", result.reason)

    def test_sell_on_crossover_from_above(self) -> None:
        result = evaluate_signal(candles_for([100] * 20 + [110, 80]))
        self.assertEqual(result.signal, Signal.SELL)
        self.assertGreater(result.previous_ema9, result.previous_ema21)
        self.assertLess(result.ema9, result.ema21)
        self.assertIn("сверху вниз", result.reason)

    def test_equal_previous_emas_can_cross_and_no_cross_means_hold(self) -> None:
        self.assertEqual(evaluate_signal(candles_for([100] * 21 + [120])).signal, Signal.BUY)
        self.assertEqual(evaluate_signal(candles_for([100] * 21 + [80])).signal, Signal.SELL)
        hold = evaluate_signal(candles_for([100] * 22))
        self.assertEqual(hold.signal, Signal.HOLD)
        self.assertIn("пересечения", hold.reason)

    def test_invalid_or_insufficient_data_raises_instead_of_hold(self) -> None:
        good = candles_for([100] * 22)
        unconfirmed = good.copy()
        unconfirmed[-1] = replace(unconfirmed[-1], confirmed=False)
        missing = good.copy()
        missing.pop(10)
        duplicate = good.copy()
        duplicate[-1] = duplicate[-2]
        reversed_candles = list(reversed(good))
        bad_price = good.copy()
        bad_price[-1] = replace(bad_price[-1], close=Decimal("NaN"))
        naive_time = good.copy()
        naive_time[-1] = replace(naive_time[-1], opened_at=datetime(2024, 1, 1, 5, 15))
        for name, candles in [
            ("too_few", good[:21]),
            ("unconfirmed", unconfirmed),
            ("gap", missing),
            ("duplicate", duplicate),
            ("reverse_order", reversed_candles),
            ("bad_price", bad_price),
            ("naive_time", naive_time),
        ]:
            with self.subTest(case=name), self.assertRaises(StrategyError):
                evaluate_signal(candles)
        for prices in ([Decimal(0)] * 9, [Decimal("Infinity")] * 9, [Decimal(1)] * 8):
            with self.subTest(prices=prices), self.assertRaises(StrategyError):
                calculate_ema(prices, 9)

    def test_cli_signal_is_read_only_and_explains_decision(self) -> None:
        candles = tuple(candles_for([100] * 20 + [90, 120]))
        instrument = SpotInstrument("BTC-USDT", Decimal("0.1"), Decimal("0.00001"), Decimal("0.00001"))
        snapshot = MarketSnapshot(instrument, Decimal("120"), candles)
        output = StringIO()
        with patch.object(main, "get_market_snapshot", return_value=snapshot), redirect_stdout(output):
            exit_code = main.main(["--signal"])
        self.assertEqual(exit_code, 0)
        self.assertIn("EMA9:", output.getvalue())
        self.assertIn("EMA21:", output.getvalue())
        self.assertIn("Сигнал: BUY", output.getvalue())
        self.assertIn("Время закрытия свечи: 2024-01-01 05:30:00 UTC", output.getvalue())
        self.assertIn("Причина: EMA9 пересекла EMA21 снизу вверх.", output.getvalue())


if __name__ == "__main__":
    unittest.main()
