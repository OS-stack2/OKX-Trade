"""Daily trend-following BTC allocator for OKX Demo, sized like the real account.

Once per UTC day the bot computes the share of momentum lookbacks (20/30/40/50/60
days) on which the last daily close is above the close N days earlier, and
holds that share of its own book in BTC (0%, 20%, ... 100%), the rest in USDT.
The book is virtual (OKX_TREND_BUDGET_USDT, default 100) and tracked in its own
SQLite. The bot only trades the BTC/USDT it accounts for, never other balances.

There is no stop-loss: the backtested strategy exits only on the daily
re-evaluation, so an intraday crash is fully felt. That is the trade-off for
fewer whipsaws compared with tight stops.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import os
from pathlib import Path
import sqlite3
import time
from typing import Iterator
from uuid import uuid4

from bot.demo_client import DemoApiCodeError, DemoError
from bot.demo_trader import DemoTradeClient, OrderStatus, OrderUncertain, TraderError, _floor
from bot.market_data import SpotInstrument
from bot.okx_client import OkxClient, OkxResponseError
from config import ConfigError, _read_env_file


LOOKBACKS = (20, 30, 40, 50, 60)
DEFAULT_BUDGET_USDT = Decimal("100")
DEFAULT_STATE_PATH = Path(__file__).resolve().parents[1] / "data" / "trend_trader.sqlite3"
RESUME_PHRASE = "RESUME OKX DEMO TREND"
LIMIT_WAIT_SECONDS = 180          # how long a maker order may rest per attempt
LIMIT_POLL_SECONDS = 5
MAX_LIMIT_ATTEMPTS = 3            # per day; then fall back to a market order
MIN_ORDER_USDT = Decimal("1")     # OKX rejects smaller orders (51020), seen live on Demo
ZERO = Decimal(0)


@dataclass(frozen=True)
class DailyClose:
    day: date
    close: Decimal


def read_daily_closes(client: OkxClient, count: int = max(LOOKBACKS) + 5) -> tuple[DailyClose, ...]:
    """Confirmed UTC daily closes, oldest first, without gaps."""
    closes: list[DailyClose] = []
    for row in client.get_daily_candles(limit=count + 1):
        if not isinstance(row, list) or len(row) < 9 or row[8] not in ("0", "1"):
            raise OkxResponseError("Invalid daily candle record")
        if row[8] == "0":
            continue
        try:
            opened = datetime.fromtimestamp(int(row[0]) / 1000, tz=timezone.utc)
            close = Decimal(row[4])
        except (TypeError, ValueError, InvalidOperation, OverflowError, OSError) as exc:
            raise OkxResponseError("Invalid daily candle") from exc
        if opened.hour or opened.minute or not close.is_finite() or close <= 0:
            raise OkxResponseError("Daily candle is not UTC-aligned or has a bad close")
        closes.append(DailyClose(opened.date(), close))
    closes.sort(key=lambda item: item.day)
    for older, newer in zip(closes, closes[1:]):
        if newer.day - older.day != timedelta(days=1):
            raise OkxResponseError("Daily candles have gaps or duplicates")
    if len(closes) < max(LOOKBACKS) + 1:
        raise OkxResponseError("Not enough confirmed daily candles")
    return tuple(closes)


def target_fraction(closes: Sequence[DailyClose], lookbacks: Sequence[int] = LOOKBACKS) -> Decimal:
    last = closes[-1].close
    rising = sum(1 for n in lookbacks if last > closes[-1 - n].close)
    return Decimal(rising) / Decimal(len(lookbacks))


@dataclass(frozen=True)
class Book:
    usdt: Decimal
    btc: Decimal
    last_day: str | None
    pending_client_id: str | None
    halted: str | None


class TrendStore:
    def __init__(self, path: Path = DEFAULT_STATE_PATH) -> None:
        self.path = path

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=5, isolation_level=None)) as db:
            db.row_factory = sqlite3.Row
            db.execute("BEGIN IMMEDIATE")
            try:
                db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                db.execute("""CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT NOT NULL, day TEXT NOT NULL,
                    side TEXT NOT NULL, btc TEXT NOT NULL, price TEXT NOT NULL, fee TEXT NOT NULL,
                    fee_ccy TEXT NOT NULL, target TEXT NOT NULL, equity TEXT NOT NULL)""")
                yield db
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def book(self, budget: Decimal) -> Book:
        with self._db() as db:
            meta = {row["key"]: row["value"] for row in db.execute("SELECT key, value FROM meta")}
            if "usdt" not in meta:
                db.execute("INSERT INTO meta VALUES('usdt', ?), ('btc', '0')", (format(budget, "f"),))
                meta.update(usdt=format(budget, "f"), btc="0")
        return Book(Decimal(meta["usdt"]), Decimal(meta["btc"]), meta.get("last_day"),
                    meta.get("pending"), meta.get("halted"))

    def meta_value(self, key: str) -> str | None:
        with self._db() as db:
            row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else row["value"]

    def set(self, **values: str | Decimal | None) -> None:
        with self._db() as db:
            for key, value in values.items():
                if value is None:
                    db.execute("DELETE FROM meta WHERE key=?", (key,))
                else:
                    text = format(value, "f") if isinstance(value, Decimal) else value
                    db.execute("INSERT INTO meta(key, value) VALUES(?, ?) "
                               "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, text))

    def apply_fill(self, *, book: Book, order: OrderStatus, day: str, target: Decimal,
                   now: datetime) -> Book:
        fee = -order.fee  # positive when charged
        notional = order.filled_btc * order.avg_price
        if order.side == "buy":
            btc = book.btc + order.filled_btc - (fee if order.fee_currency == "BTC" else ZERO)
            usdt = book.usdt - notional - (fee if order.fee_currency == "USDT" else ZERO)
        else:
            btc = book.btc - order.filled_btc - (fee if order.fee_currency == "BTC" else ZERO)
            usdt = book.usdt + notional - (fee if order.fee_currency == "USDT" else ZERO)
        equity = usdt + btc * order.avg_price
        with self._db() as db:
            if order.filled_btc > 0:
                db.execute("INSERT INTO trades(at, day, side, btc, price, fee, fee_ccy, target, equity) "
                           "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                           (now.isoformat(), day, order.side, format(order.filled_btc, "f"),
                            format(order.avg_price, "f"), format(fee, "f"), order.fee_currency,
                            format(target, "f"), format(equity, "f")))
            for key, value in (("usdt", usdt), ("btc", btc)):
                db.execute("UPDATE meta SET value=? WHERE key=?", (format(value, "f"), key))
            db.execute("DELETE FROM meta WHERE key='pending'")
        return Book(usdt, btc, book.last_day, None, book.halted)

    def trades(self) -> list[sqlite3.Row]:
        with self._db() as db:
            return db.execute("SELECT * FROM trades ORDER BY id").fetchall()


@dataclass(frozen=True)
class TrendSettings:
    budget_usdt: Decimal = DEFAULT_BUDGET_USDT

    @classmethod
    def from_env_file(cls, path: Path | None = None) -> TrendSettings:
        values = _read_env_file(path or Path(__file__).resolve().parents[1] / ".env")
        if values.get("OKX_TREND_ENABLED") != "1":
            raise ConfigError("Трендовый Demo-бот выключен: установите OKX_TREND_ENABLED=1 в .env")
        try:
            budget = Decimal(values.get("OKX_TREND_BUDGET_USDT", str(DEFAULT_BUDGET_USDT)))
        except InvalidOperation as exc:
            raise ConfigError("Некорректный OKX_TREND_BUDGET_USDT") from exc
        if not budget.is_finite() or not 10 <= budget <= 100_000:
            raise ConfigError("OKX_TREND_BUDGET_USDT должен быть от 10 до 100000")
        return cls(budget)


def trend_state_path_from_env(path: Path | None = None) -> Path:
    values = _read_env_file(path or Path(__file__).resolve().parents[1] / ".env")
    raw = values.get("OKX_TREND_STATE_PATH")
    return Path(raw) if raw else DEFAULT_STATE_PATH


@dataclass(frozen=True)
class TrendReport:
    status: str           # HALTED, HOLD, REBALANCED, SKIPPED, ERROR
    lines: tuple[str, ...]

    @property
    def exit_code(self) -> int:
        return 1 if self.status in ("HALTED", "ERROR") else 0


class TrendTrader:
    def __init__(self, *, client: DemoTradeClient, store: TrendStore,
                 closes_provider: Callable[[], Sequence[DailyClose]],
                 instrument_provider: Callable[[], SpotInstrument],
                 price_provider: Callable[[], Decimal], settings: TrendSettings = TrendSettings(),
                 notifier: Callable[[str], object] | None = None,
                 clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                 sleeper: Callable[[float], None] = time.sleep,
                 brake: Callable[[datetime], str | None] | None = None,
                 quote_provider: Callable[[], tuple[Decimal, Decimal]] | None = None) -> None:
        self.client = client
        self.store = store
        self.closes_provider = closes_provider
        self.instrument_provider = instrument_provider
        self.price_provider = price_provider
        self.settings = settings
        self.notifier = notifier
        self.clock = clock
        self.sleep = sleeper
        self.brake = brake
        self.quote_provider = quote_provider

    def _notify(self, text: str) -> None:
        if self.notifier is not None:
            try:
                self.notifier("OKX Demo тренд-бот BTC: " + text)
            except Exception:
                pass

    def _wait(self, client_id: str) -> OrderStatus | None:
        status = None
        for _ in range(10):
            status = self.client.get_order(client_id)
            if status is not None and status.terminal:
                return status
            self.sleep(0.5)
        return status

    def _halt(self, reason: str, lines: list[str]) -> TrendReport:
        count = int(self.store.meta_value("halts_total") or "0") + 1
        self.store.set(halted=f"{self.clock().isoformat()} {reason}", halts_total=str(count))
        self._notify(f"АВАРИЙНАЯ ОСТАНОВКА. {reason}. Нужна ручная проверка и --trend-resume.")
        lines.append(f"HALTED: {reason}")
        return TrendReport("HALTED", tuple(lines))

    def run_once(self) -> TrendReport:
        lines: list[str] = []
        book = self.store.book(self.settings.budget_usdt)
        if book.halted:
            return TrendReport("HALTED", (f"HALTED с {book.halted}",))
        try:
            if book.pending_client_id:
                order = self.client.get_order(book.pending_client_id)
                if order is None:
                    self.store.set(pending=None)
                    lines.append("Незавершённый ордер не дошёл до OKX; запись снята")
                else:
                    if not order.terminal:
                        # A resting maker order from an interrupted run: take it off the book.
                        self.client.cancel_order(book.pending_client_id)
                        order = self._wait(book.pending_client_id)
                        if order is None or not order.terminal:
                            raise TraderError("Ордер ребалансировки не удалось снять")
                    book = self.store.apply_fill(book=book, order=order, day=book.last_day or "",
                                                 target=Decimal(-1), now=self.clock())
                    lines.append("Учтено исполнение ордера, прерванного сбоем")
                book = self.store.book(self.settings.budget_usdt)
            closes = self.closes_provider()
            day = closes[-1].day
            if self.clock().date() - day > timedelta(days=1):
                return TrendReport("SKIPPED", (*lines, "Последняя дневная свеча устарела"))
            if book.last_day == day.isoformat():
                return TrendReport("SKIPPED", (*lines, f"День {day} уже обработан"))
            target = target_fraction(closes)
            price = self.price_provider()
            equity = book.usdt + book.btc * price
            current = book.btc * price / equity if equity > 0 else ZERO
            lines.append(f"День {day}: цель {target * 100:.0f}% в BTC, сейчас {current * 100:.0f}%; "
                         f"капитал {equity:.2f} USDT, цена {price}")
            status = "HOLD"
            if abs(target - current) >= Decimal("0.1"):
                status = self._rebalance(book, target, price, equity, day.isoformat(), lines)
                if status == "HALTED":
                    return TrendReport(status, tuple(lines))
                if status == "BRAKED":
                    return TrendReport("HOLD", tuple(lines))
                if status == "PARTIAL":
                    # Not marked as processed: the next hourly run continues with a fresh target.
                    return TrendReport("REBALANCED" if any("КУПЛЕНО" in x or "ПРОДАНО" in x for x in lines)
                                       else "HOLD", tuple(lines))
            self.store.set(last_day=day.isoformat())
            return TrendReport(status, tuple(lines))
        except TraderError as exc:
            return self._halt(str(exc), lines)
        except OrderUncertain as exc:
            return self._halt(f"Неопределённый результат ордера: {exc}", lines)
        except (DemoError, ConfigError, OkxResponseError, ValueError, OSError) as exc:
            lines.append(f"Цикл пропущен: {exc}")
            return TrendReport("ERROR", tuple(lines))

    def _rebalance(self, book: Book, target: Decimal, price: Decimal, equity: Decimal,
                   day: str, lines: list[str]) -> str:
        instrument = self.instrument_provider()
        delta_usdt = target * equity - book.btc * price
        if delta_usdt > 0:
            reason = self.brake(self.clock()) if self.brake is not None else None
            if reason:
                # Not marked as processed: the next hourly run retries after the brake.
                lines.append(f"Докупка отложена: тормоз по новостям — {reason}")
                return "BRAKED"
            side = "buy"
            size = _floor(min(delta_usdt, book.usdt * Decimal("0.995")) / price, instrument.lot_size)
        else:
            side = "sell"
            size = _floor(min(-delta_usdt / price, book.btc), instrument.lot_size)
            if target == 0:
                size = _floor(book.btc, instrument.lot_size)
        if size < instrument.min_size or size * price < MIN_ORDER_USDT:
            lines.append(f"Ребалансировка пропущена: {size} BTC меньше минимума OKX "
                         f"({instrument.min_size} BTC и {MIN_ORDER_USDT} USDT)")
            return "HOLD"
        if side == "sell":
            btc, _ = self.client.get_balances()
            if btc.available < size:
                raise TraderError("На счёте меньше BTC, чем в книге бота")
        attempts = self._limit_attempts(day)
        limit_price = None
        if self.quote_provider is not None and attempts < MAX_LIMIT_ATTEMPTS:
            bid, ask = self.quote_provider()
            limit_price = bid if side == "buy" else ask
            if side == "buy":
                size = _floor(min(delta_usdt, book.usdt * Decimal("0.995")) / limit_price, instrument.lot_size)
            if size < instrument.min_size:
                lines.append(f"Ребалансировка пропущена: {size} BTC меньше минимума {instrument.min_size}")
                return "HOLD"
        client_id = "DT" + uuid4().hex[:30]
        self.store.set(pending=client_id)
        try:
            if limit_price is None:
                self.client.place_market_order(client_order_id=client_id, side=side, size_btc=size)
            else:
                self.store.set(limit_attempts=f"{day}:{attempts + 1}")
                self.client.place_post_only_order(client_order_id=client_id, side=side,
                                                  size_btc=size, price=limit_price)
        except DemoApiCodeError as exc:
            self.store.set(pending=None)
            lines.append(f"Ордер отклонён OKX (код {exc.code}); книга не изменена")
            return "PARTIAL" if limit_price is not None else "HOLD"
        if limit_price is None:
            order = self._wait(client_id)
        else:
            lines.append(f"Лимитный {side} {size} BTC по {limit_price} (мейкер, попытка {attempts + 1} "
                         f"из {MAX_LIMIT_ATTEMPTS})")
            order = self._wait_resting(client_id)
        if order is None or not order.terminal:
            raise TraderError("Ордер ребалансировки не подтвердился вовремя")
        complete = limit_price is None or order.filled_btc >= size
        if order.filled_btc == 0:
            self.store.set(pending=None)
            lines.append("Лимитный ордер не исполнился за отведённое время и снят; повтор через час")
            return "PARTIAL"
        book = self.store.apply_fill(book=book, order=order, day=day, target=target, now=self.clock())
        new_equity = book.usdt + book.btc * order.avg_price
        verb = "КУПЛЕНО" if side == "buy" else "ПРОДАНО"
        text = (f"{verb} {order.filled_btc} BTC по {order.avg_price}. Доля BTC теперь "
                f"{book.btc * order.avg_price / new_equity * 100:.0f}% (цель {target * 100:.0f}%). "
                f"Капитал {new_equity:.2f} USDT; в BTC {new_equity / order.avg_price:.8f}.")
        lines.append(text)
        self._notify(text)
        return "REBALANCED" if complete else "PARTIAL"

    def _limit_attempts(self, day: str) -> int:
        raw = self.store.meta_value("limit_attempts") or ""
        stored_day, _, count = raw.partition(":")
        return int(count) if stored_day == day and count.isdigit() else 0

    def _wait_resting(self, client_id: str) -> OrderStatus | None:
        """Let a maker order rest, then cancel what is left and return the final state."""
        status = None
        for _ in range(int(LIMIT_WAIT_SECONDS / LIMIT_POLL_SECONDS)):
            status = self.client.get_order(client_id)
            if status is not None and status.terminal:
                return status
            self.sleep(LIMIT_POLL_SECONDS)
        self.client.cancel_order(client_id)
        return self._wait(client_id)


def resume(store: TrendStore, phrase: str) -> bool:
    if phrase != RESUME_PHRASE:
        return False
    store.set(halted=None)
    return True
