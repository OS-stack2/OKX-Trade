"""One-shot, read-only BTC-USDT public market check."""

from __future__ import annotations

import argparse
import sys

from bot.market_data import get_market_snapshot
from bot.okx_client import OkxClient, OkxError
from config import Config, ConfigError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only OKX BTC-USDT market check")
    parser.add_argument("--check", action="store_true", help="Fetch public market data once")
    args = parser.parse_args(argv)
    if not args.check:
        parser.error("Use --check")

    try:
        config = Config.from_environment()
        snapshot = get_market_snapshot(OkxClient(config))
    except (ConfigError, OkxError) as exc:
        print(f"Ошибка проверки OKX: {exc}", file=sys.stderr)
        return 1

    print(f"Инструмент: {snapshot.instrument.instrument_id}")
    print(f"Текущая цена: {snapshot.last_price} USDT")
    print(
        "Время последней закрытой свечи: "
        f"{snapshot.latest_closed_candle.closed_at:%Y-%m-%d %H:%M:%S} UTC"
    )
    print(f"Количество закрытых свечей: {len(snapshot.candles)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
