"""One-shot, read-only BTC-USDT public market check."""

from __future__ import annotations

import argparse
import sys

from bot.market_data import get_market_snapshot
from bot.okx_client import OkxClient, OkxError
from bot.strategy import StrategyError, evaluate_signal
from config import Config, ConfigError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only OKX BTC-USDT market data and EMA signal")
    commands = parser.add_mutually_exclusive_group(required=True)
    commands.add_argument("--check", action="store_true", help="Fetch public market data once")
    commands.add_argument("--signal", action="store_true", help="Calculate the EMA 9/21 signal once")
    args = parser.parse_args(argv)

    try:
        config = Config.from_environment()
        snapshot = get_market_snapshot(OkxClient(config))
        result = evaluate_signal(snapshot.candles) if args.signal else None
    except (ConfigError, OkxError, StrategyError) as exc:
        print(f"Ошибка {('сигнала' if args.signal else 'проверки OKX')}: {exc}", file=sys.stderr)
        return 1

    if args.check:
        print(f"Инструмент: {snapshot.instrument.instrument_id}")
        print(f"Текущая цена: {snapshot.last_price} USDT")
        print(
            "Время последней закрытой свечи: "
            f"{snapshot.latest_closed_candle.closed_at:%Y-%m-%d %H:%M:%S} UTC"
        )
        print(f"Количество закрытых свечей: {len(snapshot.candles)}")
    else:
        assert result is not None
        print(f"Инструмент: {snapshot.instrument.instrument_id}")
        print(f"EMA9: {result.ema9:.8f}")
        print(f"EMA21: {result.ema21:.8f}")
        print(f"Сигнал: {result.signal.value}")
        print(f"Время закрытия свечи: {result.candle_closed_at:%Y-%m-%d %H:%M:%S} UTC")
        print(f"Причина: {result.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
