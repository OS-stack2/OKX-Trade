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
    state: str | None = None
    base_currency: str | None = None
    quote_currency: str | None = None
    max_limit_size: Decimal | None = None
    max_market_size: Decimal | None = None


@dataclass(frozen=True)
class MarketSnapshot:
    instrument: SpotInstrument
    last_price: Decimal
    candles: tuple[Candle, ...]
    ticker_as_of_utc: datetime | None = None

    @property
    def latest_closed_candle(self) -> Candle:
        return self.candles[-1]


@dataclass(frozen=True)
class PublicQuoteEvidence:
    """Independent public GET results; a failed endpoint cannot hide the other."""

    instrument: SpotInstrument | None
    last_price: Decimal | None
    ticker_as_of_utc: datetime | None
    issues: tuple[str, ...]


@dataclass(frozen=True)
class PublicQuoteAssessment:
    """Only independently validated evidence may be used for Demo accounting."""

    instrument: SpotInstrument | None
    price: Decimal | None
    ticker_as_of_utc: datetime | None
    checks: tuple[tuple[str, str], ...]
    reasons: tuple[str, ...]


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


def read_spot_instrument(client: OkxClient) -> SpotInstrument:
    """Read only the public BTC-USDT SPOT instrument, without ticker/candles."""
    instrument_data = _one_record(client.get_spot_instrument(), "instrument")
    if instrument_data.get("instId") != "BTC-USDT" or instrument_data.get("instType") != "SPOT":
        raise OkxResponseError("Unexpected instrument returned by OKX")
    return SpotInstrument(
        instrument_id="BTC-USDT",
        tick_size=_positive_decimal(instrument_data.get("tickSz"), "tickSz"),
        lot_size=_positive_decimal(instrument_data.get("lotSz"), "lotSz"),
        min_size=_positive_decimal(instrument_data.get("minSz"), "minSz"),
        state=instrument_data.get("state"),
        base_currency=instrument_data.get("baseCcy"),
        quote_currency=instrument_data.get("quoteCcy"),
        max_limit_size=(_positive_decimal(instrument_data["maxLmtSz"], "maxLmtSz")
                        if instrument_data.get("maxLmtSz") not in (None, "") else None),
        max_market_size=(_positive_decimal(instrument_data["maxMktSz"], "maxMktSz")
                         if instrument_data.get("maxMktSz") not in (None, "") else None),
    )


def get_public_quote_evidence(client: OkxClient) -> PublicQuoteEvidence:
    """Read BTC-USDT instrument and ticker independently; never request candles."""
    instrument = None
    price = None
    ticker_time = None
    issues: list[str] = []
    try:
        instrument = read_spot_instrument(client)
    except OkxResponseError as exc:
        field = next((name for name in ("minSz", "lotSz", "tickSz", "maxLmtSz", "maxMktSz")
                      if str(exc) == f"Invalid {name} in OKX response"), None)
        issues.append((f"LIMITS: поле {field} отсутствует или некорректно" if field
                       else "INSTRUMENT: публичный ответ инструмента повреждён или отклонён"))
    except Exception:
        # HTTP/API exception text is deliberately not echoed into a diagnostic.
        issues.append("INSTRUMENT: публичный GET инструмента недоступен или ответ повреждён")
    try:
        ticker = _one_record(client.get_ticker(), "ticker")
    except Exception:
        issues.append("PRICE: публичный GET тикера недоступен или ответ повреждён")
    else:
        if ticker.get("instId") != "BTC-USDT":
            issues.append("PRICE: ticker.instId не равен BTC-USDT")
        else:
            try:
                price = _positive_decimal(ticker.get("last"), "last price")
            except OkxResponseError:
                issues.append("PRICE: ticker.last отсутствует или некорректен")
            raw_time = ticker.get("ts")
            if not isinstance(raw_time, str) or not raw_time.isascii() or not raw_time.isdigit():
                issues.append("QUOTE_TIME: ticker.ts отсутствует или некорректен")
            else:
                try:
                    ticker_time = datetime.fromtimestamp(int(raw_time) / 1000, tz=timezone.utc)
                except (OverflowError, OSError, ValueError):
                    issues.append("QUOTE_TIME: ticker.ts вне диапазона UTC")
    return PublicQuoteEvidence(instrument, price, ticker_time, tuple(issues))


def assess_public_quote(
    evidence: PublicQuoteEvidence | MarketSnapshot, now: datetime,
    *, max_ticker_age: timedelta = timedelta(seconds=60),
) -> PublicQuoteAssessment:
    """Separate price, freshness, identity and limit checks; fail closed."""
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() != timedelta(0):
        raise ValueError("Current UTC time is invalid")
    if not isinstance(evidence, (PublicQuoteEvidence, MarketSnapshot)):
        raise ValueError("Public quote evidence is unavailable")
    reasons = list(evidence.issues) if isinstance(evidence, PublicQuoteEvidence) else []
    checks: list[tuple[str, str]] = []
    raw_price = evidence.last_price
    if not isinstance(raw_price, Decimal) or not raw_price.is_finite() or raw_price <= 0:
        checks.append(("PRICE", "BLOCKED"))
        if not any(reason.startswith("PRICE:") for reason in reasons):
            reasons.append("PRICE: положительная конечная цена ticker.last не подтверждена")
    else:
        checks.append(("PRICE", "OK"))
    at = evidence.ticker_as_of_utc
    if not isinstance(at, datetime) or at.tzinfo is None or at.utcoffset() != timedelta(0):
        checks.append(("QUOTE_TIME", "BLOCKED"))
        if not any(reason.startswith("QUOTE_TIME:") for reason in reasons):
            reasons.append("QUOTE_TIME: время тикера неизвестно или не UTC")
    elif at > now:
        checks.append(("QUOTE_TIME", "BLOCKED"))
        reasons.append(f"QUOTE_TIME: ticker.ts {at.isoformat()} находится в будущем относительно UTC сервера")
    elif now - at > max_ticker_age:
        checks.append(("QUOTE_TIME", "BLOCKED"))
        reasons.append(f"QUOTE_TIME: котировка старше {int(max_ticker_age.total_seconds())} секунд "
                       f"(возраст {int((now - at).total_seconds())} с; ticker.ts {at.isoformat()})")
    else:
        checks.append(("QUOTE_TIME", "OK"))
    instrument = evidence.instrument
    instrument_faults = []
    if not isinstance(instrument, SpotInstrument):
        instrument_faults.append("запись инструмента отсутствует")
    else:
        for name, actual, required in (("instId", instrument.instrument_id, "BTC-USDT"),
                                       ("state", instrument.state, "live"),
                                       ("baseCcy", instrument.base_currency, "BTC"),
                                       ("quoteCcy", instrument.quote_currency, "USDT")):
            if actual != required:
                instrument_faults.append(f"{name} не равен {required}")
    if instrument_faults:
        checks.append(("INSTRUMENT", "BLOCKED"))
        if not any(reason.startswith("INSTRUMENT:") for reason in reasons):
            reasons.append("INSTRUMENT: " + ", ".join(instrument_faults))
    else:
        checks.append(("INSTRUMENT", "OK"))
    bad_limits = []
    if isinstance(instrument, SpotInstrument):
        for name, value in (("minSz", instrument.min_size), ("lotSz", instrument.lot_size),
                            ("tickSz", instrument.tick_size), ("maxLmtSz", instrument.max_limit_size),
                            ("maxMktSz", instrument.max_market_size)):
            if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
                bad_limits.append(name)
        if ("minSz" not in bad_limits and "maxLmtSz" not in bad_limits
                and instrument.max_limit_size < instrument.min_size):
            bad_limits.append("maxLmtSz<minSz")
    else:
        bad_limits.append("instrument GET")
    if bad_limits:
        checks.append(("LIMITS", "BLOCKED"))
        reasons.append("LIMITS: ограничения отсутствуют или противоречивы; поля " + ", ".join(bad_limits))
    else:
        checks.append(("LIMITS", "OK"))
    statuses = dict(checks)
    confirmed_instrument = (instrument if statuses["INSTRUMENT"] == statuses["LIMITS"] == "OK" else None)
    confirmed_price = (raw_price if statuses["PRICE"] == statuses["QUOTE_TIME"] == "OK" else None)
    return PublicQuoteAssessment(confirmed_instrument, confirmed_price, at,
                                 tuple(checks), tuple(dict.fromkeys(reasons)))


def get_market_snapshot(client: OkxClient) -> MarketSnapshot:
    instrument = read_spot_instrument(client)

    ticker_data = _one_record(client.get_ticker(), "ticker")
    if ticker_data.get("instId") != "BTC-USDT":
        raise OkxResponseError("Unexpected ticker returned by OKX")
    last_price = _positive_decimal(ticker_data.get("last"), "last price")
    ticker_as_of_utc = None
    if "ts" in ticker_data:
        try:
            ticker_as_of_utc = datetime.fromtimestamp(int(ticker_data["ts"]) / 1000, tz=timezone.utc)
        except (TypeError, ValueError, OverflowError, OSError) as exc:
            raise OkxResponseError("Invalid ticker timestamp") from exc

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
    return MarketSnapshot(instrument, last_price, tuple(reversed(selected)), ticker_as_of_utc)
