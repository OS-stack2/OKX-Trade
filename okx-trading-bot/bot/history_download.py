"""Validated public OKX history download; never reads Demo keys or trading SQLite."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import os
from pathlib import Path
import tempfile
import time
from collections.abc import Callable

from bot.growth_research import MIN_BARS, validate_candles
from bot.market_data import Candle
from bot.okx_client import OkxClient


BAR_MS = 900_000
PAGE_SIZE = 100
RESEARCH_BARS = 17_280  # About 180 days of 15-minute candles.
MAX_BARS = 35_040  # Bound the number of API pages and memory use to about one year.
CSV_HEADER = ("opened_at", "open", "high", "low", "close", "confirm")


class HistoryError(ValueError):
    """The historical API series or destination cannot be trusted."""


@dataclass(frozen=True)
class HistoryReport:
    candles: tuple[Candle, ...]
    requested_bars: int
    sufficient: bool
    reason: str

    @property
    def split_counts(self) -> tuple[int, int, int]:
        count = len(self.candles)
        train_end, validation_end = count * 3 // 5, count * 4 // 5
        return train_end, validation_end - train_end, count - validation_end


def _number(value: object, name: str, *, positive: bool) -> Decimal:
    if not isinstance(value, str):
        raise HistoryError(f"OKX candle {name} is malformed")
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise HistoryError(f"OKX candle {name} is malformed") from exc
    if not number.is_finite() or number < 0 or (positive and number == 0):
        raise HistoryError(f"OKX candle {name} is malformed")
    return number


def _parse_row(row: object, now: datetime) -> Candle | None:
    if not isinstance(row, list) or len(row) != 9:
        raise HistoryError("OKX candle row must have nine fields")
    raw_time = row[0]
    if not isinstance(raw_time, str) or not raw_time.isascii() or not raw_time.isdigit():
        raise HistoryError("OKX candle timestamp is malformed")
    millis = int(raw_time)
    if millis <= 0 or millis % BAR_MS:
        raise HistoryError("OKX candle timestamp is not aligned to 15m UTC")
    try:
        opened_at = datetime.fromtimestamp(millis / 1000, tz=timezone.utc)
    except (OverflowError, OSError, ValueError) as exc:
        raise HistoryError("OKX candle timestamp is out of range") from exc
    open_, high, low, close = (_number(row[index], f"OHLC[{index}]", positive=True)
                               for index in range(1, 5))
    for index in range(5, 8):
        _number(row[index], f"volume[{index}]", positive=False)
    if high < max(open_, close) or low > min(open_, close) or low > high:
        raise HistoryError("OKX candle OHLC values contradict each other")
    if row[8] not in ("0", "1"):
        raise HistoryError("OKX candle confirmation flag is malformed")
    if row[8] == "0":
        if opened_at > now or opened_at + timedelta(minutes=15) <= now:
            raise HistoryError("OKX candle is unconfirmed after its close time")
        return None
    if opened_at + timedelta(minutes=15) > now:
        raise HistoryError("OKX claims a future candle is closed")
    return Candle(opened_at, open_, high, low, close, True)


def download_history(
    client: OkxClient,
    *,
    bars: int = RESEARCH_BARS,
    now: datetime | None = None,
    sleeper: Callable[[float], None] = time.sleep,
) -> HistoryReport:
    """Read newest-to-oldest pages, rejecting any overlap, gap or stalled cursor."""
    if not isinstance(bars, int) or isinstance(bars, bool) or not MIN_BARS <= bars <= MAX_BARS:
        raise HistoryError(f"Requested bars must be between {MIN_BARS} and {MAX_BARS}")
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() != timedelta(0):
        raise HistoryError("Current time must be UTC")
    newest_first: list[Candle] = []
    seen: set[datetime] = set()
    cursor: int | None = None
    previous: datetime | None = None
    open_seen = False
    exhausted = False
    # One page may contain a still-open candle; an extra empty page detects exhaustion.
    max_pages = (bars + PAGE_SIZE - 1) // PAGE_SIZE + 2
    for page_number in range(max_pages):
        if page_number:
            sleeper(0.11)  # Below OKX's documented 20 requests / 2 seconds.
        rows = client.get_history_candles(after=cursor, limit=PAGE_SIZE)
        if not rows:
            exhausted = True
            break
        if len(rows) > PAGE_SIZE:
            raise HistoryError("OKX returned more candles than the page limit")
        page_oldest: int | None = None
        for row in rows:
            candle = _parse_row(row, now)
            raw_ms = int(row[0])
            if page_oldest is not None and raw_ms >= page_oldest:
                raise HistoryError("OKX page is not strictly newest-to-oldest")
            if cursor is not None and raw_ms >= cursor:
                raise HistoryError("OKX pages overlap or pagination did not advance")
            page_oldest = raw_ms
            if candle is None:
                if page_number != 0 or newest_first or open_seen:
                    raise HistoryError("Unclosed OKX candle appeared inside historical series")
                open_seen = True
                continue
            if candle.opened_at in seen:
                raise HistoryError("Duplicate OKX historical candle")
            if previous is not None and previous - candle.opened_at != timedelta(minutes=15):
                raise HistoryError("Gap or overlap in OKX 15m candle history")
            seen.add(candle.opened_at)
            newest_first.append(candle)
            previous = candle.opened_at
        if page_oldest is None or (cursor is not None and page_oldest >= cursor):
            raise HistoryError("OKX history cursor did not advance")
        cursor = page_oldest
        if len(newest_first) >= bars:
            break
    else:
        raise HistoryError("OKX history exceeded the bounded page count")
    if len(newest_first) < MIN_BARS:
        raise HistoryError(f"Only {len(newest_first)} validated closed candles; need at least {MIN_BARS}")
    candles = tuple(reversed(newest_first[:bars]))
    validate_candles(candles)
    count = len(candles)
    train, validation, final = count * 3 // 5, count * 4 // 5 - count * 3 // 5, count - count * 4 // 5
    sufficient = count >= RESEARCH_BARS and train >= 8640 and validation >= 2880 and final >= 2880
    if sufficient:
        reason = "At least six months of contiguous candles for TRAIN/VALIDATION/FINAL_TEST"
    else:
        why = "API history exhausted" if exhausted else "requested window too short"
        reason = f"Insufficient research history ({why}): {count}/{RESEARCH_BARS} bars"
    return HistoryReport(candles, bars, sufficient, reason)


def save_history_csv(report: HistoryReport, destination: Path, *, overwrite: bool = False) -> None:
    """Write a complete private temp file, then publish without replacing by default."""
    if destination.suffix.lower() != ".csv":
        raise HistoryError("Destination must be a CSV path")
    if destination.exists() and not overwrite:
        raise HistoryError("History CSV already exists; explicit --overwrite-history required")
    if not destination.parent.is_dir():
        raise HistoryError("History destination directory does not exist")
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=".okx-history-", suffix=".tmp", dir=destination.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(CSV_HEADER)
            for candle in report.candles:
                writer.writerow((candle.opened_at.isoformat().replace("+00:00", "Z"), candle.open,
                                 candle.high, candle.low, candle.close, "1"))
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)  # Fails atomically if another process created it.
    except FileExistsError as exc:
        raise HistoryError("History CSV already exists; explicit --overwrite-history required") from exc
    except OSError as exc:
        raise HistoryError("Could not save history CSV safely") from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
