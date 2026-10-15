"""Read-only EMA 9/21 crossover signal for confirmed 15-minute candles."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from enum import StrEnum

from bot.market_data import Candle


SHORT_PERIOD = 9
LONG_PERIOD = 21
CANDLE_INTERVAL = timedelta(minutes=15)


class StrategyError(ValueError):
    """The candle sequence cannot produce a trustworthy EMA signal."""


class Signal(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass(frozen=True)
class StrategyResult:
    signal: Signal
    ema9: Decimal
    ema21: Decimal
    previous_ema9: Decimal
    previous_ema21: Decimal
    candle_closed_at: datetime
    reason: str


def calculate_ema(prices: Sequence[Decimal], period: int) -> tuple[Decimal, ...]:
    """Seed with the first period's SMA, then apply the standard EMA multiplier."""
    if type(period) is not int or period < 1:
        raise StrategyError("EMA period must be a positive integer")
    if not isinstance(prices, Sequence) or isinstance(prices, (str, bytes)):
        raise StrategyError("Closing prices must be a sequence")
    if len(prices) < period:
        raise StrategyError(f"At least {period} closing prices are required for EMA{period}")

    validated: list[Decimal] = []
    for index, price in enumerate(prices):
        try:
            value = Decimal(str(price))
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise StrategyError(f"Invalid closing price at index {index}") from exc
        if not value.is_finite() or value <= 0:
            raise StrategyError(f"Invalid closing price at index {index}")
        validated.append(value)

    ema = sum(validated[:period], Decimal(0)) / Decimal(period)
    result = [ema]
    multiplier = Decimal(2) / Decimal(period + 1)
    for price in validated[period:]:
        ema += multiplier * (price - ema)
        result.append(ema)
    return tuple(result)


def evaluate_signal(candles: Sequence[Candle]) -> StrategyResult:
    """Compare EMAs at the two most recent confirmed, ordered candles."""
    if not isinstance(candles, Sequence) or isinstance(candles, (str, bytes)):
        raise StrategyError("Candles must be a sequence")
    if len(candles) < LONG_PERIOD + 1:
        raise StrategyError("At least 22 confirmed candles are required for a crossover")

    closes: list[Decimal] = []
    previous_opened_at: datetime | None = None
    for index, candle in enumerate(candles):
        if not isinstance(candle, Candle) or candle.confirmed is not True:
            raise StrategyError(f"Candle at index {index} is not confirmed")
        opened_at = candle.opened_at
        if not isinstance(opened_at, datetime) or opened_at.tzinfo is None or opened_at.utcoffset() is None:
            raise StrategyError(f"Candle at index {index} has no timezone-aware timestamp")
        opened_at = opened_at.astimezone(timezone.utc)
        if opened_at.minute % 15 or opened_at.second or opened_at.microsecond:
            raise StrategyError(f"Candle at index {index} is not aligned to 15 minutes")
        if previous_opened_at is not None and opened_at - previous_opened_at != CANDLE_INTERVAL:
            raise StrategyError("Candles must be ordered oldest to newest without gaps or duplicates")
        previous_opened_at = opened_at
        closes.append(candle.close)

    ema9 = calculate_ema(closes, SHORT_PERIOD)
    ema21 = calculate_ema(closes, LONG_PERIOD)
    previous_ema9, current_ema9 = ema9[-2:]
    previous_ema21, current_ema21 = ema21[-2:]

    if previous_ema9 <= previous_ema21 and current_ema9 > current_ema21:
        signal, reason = Signal.BUY, "EMA9 пересекла EMA21 снизу вверх."
    elif previous_ema9 >= previous_ema21 and current_ema9 < current_ema21:
        signal, reason = Signal.SELL, "EMA9 пересекла EMA21 сверху вниз."
    else:
        signal, reason = Signal.HOLD, "На двух последних закрытых свечах пересечения EMA9 и EMA21 нет."

    assert previous_opened_at is not None
    return StrategyResult(
        signal=signal,
        ema9=current_ema9,
        ema21=current_ema21,
        previous_ema9=previous_ema9,
        previous_ema21=previous_ema21,
        candle_closed_at=previous_opened_at + CANDLE_INTERVAL,
        reason=reason,
    )
