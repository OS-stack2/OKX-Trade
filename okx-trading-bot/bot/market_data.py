"""Validate and normalize public BTC-USDT market data."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

from bot.okx_client import OkxClient, OkxResponseError


@dataclass(frozen=True)
class Candle:
    opened_at: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    confirmed: bool

    @property
    def closed_at(self) -> datetime:
        return self.opened_at + timedelta(minutes=15)


@dataclass(frozen=True)
class SpotInstrument:
    instrument_id: str
    tick_size: Decimal
    lot_size: Decimal
    min_size: Decimal


@dataclass(frozen=True)
class MarketSnapshot:
    instrument: SpotInstrument
    last_price: Decimal
    candles: tuple[Candle, ...]

    @property
    def latest_closed_candle(self) -> Candle:
        return self.candles[-1]


def _positive_decimal(value: Any, field: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise OkxResponseError(f"Invalid {field} in OKX response") from exc
    if not result.is_finite() or result <= 0:
        raise OkxResponseError(f"Invalid {field} in OKX response")
    return result


def _one_record(data: list[Any], field: str) -> dict[str, Any]:
    if len(data) != 1 or not isinstance(data[0], dict):
        raise OkxResponseError(f"Invalid {field} record in OKX response")
    return data[0]


def get_market_snapshot(client: OkxClient) -> MarketSnapshot:
    instrument_data = _one_record(client.get_spot_instrument(), "instrument")
    if instrument_data.get("instId") != "BTC-USDT" or instrument_data.get("instType") != "SPOT":
        raise OkxResponseError("Unexpected instrument returned by OKX")
    instrument = SpotInstrument(
        instrument_id="BTC-USDT",
        tick_size=_positive_decimal(instrument_data.get("tickSz"), "tickSz"),
        lot_size=_positive_decimal(instrument_data.get("lotSz"), "lotSz"),
        min_size=_positive_decimal(instrument_data.get("minSz"), "minSz"),
    )

    ticker_data = _one_record(client.get_ticker(), "ticker")
    if ticker_data.get("instId") != "BTC-USDT":
        raise OkxResponseError("Unexpected ticker returned by OKX")
    last_price = _positive_decimal(ticker_data.get("last"), "last price")

    closed: list[Candle] = []
    for row in client.get_candles(limit=101):
        if not isinstance(row, list) or len(row) < 9:
            raise OkxResponseError("Invalid candle record in OKX response")
        if row[8] == "0":
            continue
        if row[8] != "1":
            raise OkxResponseError("Invalid candle confirmation flag")
        try:
            opened_at = datetime.fromtimestamp(int(row[0]) / 1000, tz=timezone.utc)
        except (TypeError, ValueError, OverflowError, OSError) as exc:
            raise OkxResponseError("Invalid candle timestamp") from exc
        candle = Candle(
            opened_at=opened_at,
            open=_positive_decimal(row[1], "candle open"),
            high=_positive_decimal(row[2], "candle high"),
            low=_positive_decimal(row[3], "candle low"),
            close=_positive_decimal(row[4], "candle close"),
            confirmed=True,
        )
        if candle.high < max(candle.open, candle.close, candle.low) or candle.low > min(
            candle.open, candle.close, candle.high
        ):
            raise OkxResponseError("Invalid candle OHLC range")
        closed.append(candle)

    closed.sort(key=lambda item: item.opened_at, reverse=True)
    if len(closed) < 100:
        raise OkxResponseError(f"Only {len(closed)} closed candles returned; need 100")
    selected = closed[:100]
    if len({item.opened_at for item in selected}) != 100:
        raise OkxResponseError("Duplicate candle timestamps returned by OKX")
    return MarketSnapshot(instrument, last_price, tuple(reversed(selected)))
