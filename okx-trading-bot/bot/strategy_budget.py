"""Immutable owner allocation of existing virtual USDT; no exchange writes."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
import sqlite3

from bot.journal import DEFAULT_JOURNAL_PATH


STRATEGY_BUDGET_USDT = Decimal("5000")
CONFIRMATION_PHRASE = "CONFIRM DEMO BTC-USDT 5000 USDT"
SOURCE_DESCRIPTION = "Owner allocation from existing OKX Demo USDT; no bot deposit or transfer"


class BudgetError(ValueError):
    """The budget record is missing, corrupt, repeated or unsafe to save."""


@dataclass(frozen=True)
class BudgetBaseline:
    confirmed_at_utc: datetime
    strategy_capital_usdt: Decimal
    demo_btc_total: Decimal
    demo_btc_available: Decimal
    demo_usdt_total: Decimal
    demo_usdt_available: Decimal
    btc_price_usdt: Decimal
    portfolio_btc_usdt_value: Decimal
    maker_fee_rate: Decimal
    taker_fee_rate: Decimal
    journal_decision_count: int
    source_description: str = SOURCE_DESCRIPTION


def _utc_iso(value: datetime) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise BudgetError("Для бюджета требуется время UTC")
    return value.isoformat().replace("+00:00", "Z")


def _money(value: Decimal, *, positive: bool = False) -> str:
    if not isinstance(value, Decimal) or not value.is_finite() or value < 0 or (positive and value == 0):
        raise BudgetError("Некорректная сумма бюджета или снимка Demo")
    return str(value)


class StrategyBudgetStore:
    """Single immutable row in the existing, Git-ignored decision journal."""

    _CREATE = """
        CREATE TABLE IF NOT EXISTS strategy_budget (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            confirmed_at_utc TEXT NOT NULL,
            strategy_capital_usdt TEXT NOT NULL,
            demo_btc_total TEXT NOT NULL,
            demo_btc_available TEXT NOT NULL,
            demo_usdt_total TEXT NOT NULL,
            demo_usdt_available TEXT NOT NULL,
            btc_price_usdt TEXT NOT NULL,
            portfolio_btc_usdt_value TEXT NOT NULL,
            maker_fee_rate TEXT NOT NULL,
            taker_fee_rate TEXT NOT NULL,
            journal_decision_count INTEGER NOT NULL,
            source_description TEXT NOT NULL
        )
    """

    def __init__(self, path: Path = DEFAULT_JOURNAL_PATH) -> None:
        self.path = Path(path)

    def read(self) -> BudgetBaseline | None:
        if not self.path.is_file():
            raise BudgetError("SQLite-журнал решений отсутствует")
        try:
            with closing(sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=3)) as db:
                db.execute("PRAGMA query_only=ON")
                exists = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='strategy_budget'").fetchone()
                if not exists:
                    return None
                rows = db.execute("SELECT id,confirmed_at_utc,strategy_capital_usdt,demo_btc_total,"
                                  "demo_btc_available,demo_usdt_total,demo_usdt_available,btc_price_usdt,"
                                  "portfolio_btc_usdt_value,maker_fee_rate,taker_fee_rate,"
                                  "journal_decision_count,source_description FROM strategy_budget").fetchall()
            if len(rows) != 1:
                raise BudgetError("Запись бюджета отсутствует или дублируется")
            row = rows[0]
            if type(row[0]) is not int or row[0] != 1:
                raise BudgetError("Идентификатор стартового бюджета повреждён")
            confirmed_at = datetime.fromisoformat(row[1].replace("Z", "+00:00"))
            values = tuple(Decimal(value) for value in row[2:11])
            record = BudgetBaseline(confirmed_at, *values, row[11], row[12])
            self._validate(record)
            return record
        except (sqlite3.Error, OSError, TypeError, ValueError, InvalidOperation, IndexError) as exc:
            raise BudgetError("Запись стартового бюджета повреждена или недоступна") from exc

    @staticmethod
    def _validate(record: BudgetBaseline) -> None:
        _utc_iso(record.confirmed_at_utc)
        if record.strategy_capital_usdt != STRATEGY_BUDGET_USDT or record.source_description != SOURCE_DESCRIPTION:
            raise BudgetError("Неверная сумма или происхождение бюджета")
        for value in (record.strategy_capital_usdt, record.demo_btc_total, record.demo_btc_available,
                      record.demo_usdt_total, record.demo_usdt_available, record.btc_price_usdt,
                      record.portfolio_btc_usdt_value):
            _money(value)
        if (record.demo_btc_total != Decimal("1") or record.demo_btc_available != Decimal("1")
                or record.demo_usdt_available != record.demo_usdt_total
                or record.demo_usdt_available < STRATEGY_BUDGET_USDT
                or record.btc_price_usdt <= 0
                or record.portfolio_btc_usdt_value != record.demo_usdt_total + record.demo_btc_total * record.btc_price_usdt):
            raise BudgetError("Снимок Demo противоречит бюджету")
        for rate in (record.maker_fee_rate, record.taker_fee_rate):
            if not isinstance(rate, Decimal) or not rate.is_finite() or not -1 < rate < 1:
                raise BudgetError("Некорректная комиссия в записи бюджета")
        if type(record.journal_decision_count) is not int or record.journal_decision_count < 0:
            raise BudgetError("Некорректная ссылка бюджета на журнал")

    def confirm_once(self, record: BudgetBaseline) -> None:
        self._validate(record)
        if not self.path.is_file():
            raise BudgetError("SQLite-журнал решений отсутствует")
        try:
            with closing(sqlite3.connect(self.path, timeout=3)) as db:
                db.execute("PRAGMA synchronous=FULL")
                db.execute("BEGIN IMMEDIATE")
                count, executed = db.execute("SELECT COUNT(*), SUM(executed) FROM decisions").fetchone()
                if count != record.journal_decision_count or executed not in (None, 0):
                    raise BudgetError("Журнал изменился во время подтверждения; повторите просмотр")
                db.execute(self._CREATE)
                if db.execute("SELECT 1 FROM strategy_budget LIMIT 1").fetchone():
                    raise BudgetError("Стартовый бюджет уже подтверждён; повторная инициализация запрещена")
                db.execute("""INSERT INTO strategy_budget VALUES (1,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                    _utc_iso(record.confirmed_at_utc), str(record.strategy_capital_usdt),
                    str(record.demo_btc_total), str(record.demo_btc_available),
                    str(record.demo_usdt_total), str(record.demo_usdt_available),
                    str(record.btc_price_usdt), str(record.portfolio_btc_usdt_value),
                    str(record.maker_fee_rate), str(record.taker_fee_rate),
                    record.journal_decision_count, record.source_description,
                ))
                db.commit()
        except (sqlite3.Error, OSError) as exc:
            raise BudgetError("Не удалось безопасно сохранить стартовый бюджет") from exc
