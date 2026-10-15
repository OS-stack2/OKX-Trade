"""Durable execution preparation for OKX Demo BTC-USDT SPOT.

There is deliberately no HTTP client, order payload builder, or send method here.
The transition methods only record locally observed or simulated facts.
"""

from __future__ import annotations

from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_DOWN
from enum import StrEnum
from pathlib import Path
import re
import sqlite3
from typing import Iterator
from uuid import uuid4

from bot.journal import DEFAULT_JOURNAL_PATH, DecisionJournal, JournalError
from bot.reconcile import ReconcileReport, ReconcileStatus
from bot.risk_manager import (
    AccountSnapshot, ExecutionCosts, InstrumentLimits, RiskDecision, RiskManager, RiskSettings,
    RiskStateError, RiskStatus,
)
from bot.strategy_budget import BudgetError, STRATEGY_BUDGET_USDT, StrategyBudgetStore


class ExecutionError(ValueError):
    """Unsafe or contradictory local execution evidence."""


class ExecutionState(StrEnum):
    PLANNED = "PLANNED"
    VALIDATED = "VALIDATED"
    SUBMITTING = "SUBMITTING"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"
    HALTED = "HALTED"


BLOCKING_STATES = frozenset(state for state in ExecutionState if state != ExecutionState.REJECTED)
RECOVERY_PHRASE = "RECOVER DEMO AFTER ORDER RECONCILIATION"
_ORDER_ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_CLIENT_ID = re.compile(r"TB[a-f0-9]{30}\Z")


@dataclass(frozen=True)
class ExecutionIntent:
    client_order_id: str
    decision_id: str
    candle_closed_at: datetime
    state: ExecutionState
    quantity: Decimal
    entry_price: Decimal
    stop_price: Decimal
    take_profit_price: Decimal
    estimated_risk_usdt: Decimal
    min_size: Decimal
    lot_size: Decimal
    tick_size: Decimal
    created_at_utc: datetime
    updated_at_utc: datetime
    exchange_order_id: str | None = None
    filled_quantity: Decimal = Decimal(0)
    fee_btc: Decimal = Decimal(0)  # Signed OKX fee/rebate.
    fee_usdt: Decimal = Decimal(0)
    average_fill_price: Decimal | None = None
    reason_code: str = "PLANNED"

    @property
    def owned_btc(self) -> Decimal:
        return self.filled_quantity + self.fee_btc


@dataclass(frozen=True)
class ExecutionDiagnostic:
    blocked: bool
    reason: str
    halted: bool
    intents: tuple[ExecutionIntent, ...]


@dataclass(frozen=True)
class RecoveryProof:
    """Per-client-ID exchange reconciliation, supplied by a future read-only reader."""

    client_order_id: str | None
    checked_at_utc: datetime
    no_exchange_order: bool
    no_exchange_fills: bool
    balances_reconciled: bool
    source: str


def _utc(value: datetime, label: str) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ExecutionError(f"{label}: требуется UTC")
    return value.isoformat().replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        _utc(parsed, "Время исполнения")
        return parsed
    except (TypeError, AttributeError, ValueError) as exc:
        raise ExecutionError("Повреждено время в журнале исполнения") from exc


def _decimal(value: object, label: str, *, positive: bool = False) -> Decimal:
    try:
        number = value if isinstance(value, Decimal) else Decimal(value)
    except (TypeError, ValueError, InvalidOperation) as exc:
        raise ExecutionError(f"{label}: некорректное число") from exc
    if not number.is_finite() or (positive and number <= 0):
        raise ExecutionError(f"{label}: некорректное число")
    return number


def _floor_step(value: Decimal, step: Decimal) -> Decimal:
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def _ceil_step(value: Decimal, step: Decimal) -> Decimal:
    return (value / step).to_integral_value(rounding=ROUND_CEILING) * step


class ExecutionEngine:
    """SQLite-backed state machine with no ability to place or cancel orders."""

    def __init__(
        self, *, journal: DecisionJournal | None = None, risk_manager: RiskManager | None = None,
        budget_store: StrategyBudgetStore | None = None,
        clock=lambda: datetime.now(timezone.utc),
    ) -> None:
        self.journal = journal or DecisionJournal()
        self.risk_manager = risk_manager or RiskManager()
        self.budget_store = budget_store or StrategyBudgetStore(self.journal.path)
        self.path = Path(self.journal.path)
        self.clock = clock

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """Commit or roll back before closing SQLite, including on exceptions."""
        with closing(sqlite3.connect(self.path, timeout=3)) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA synchronous=FULL")
            try:
                db.execute("BEGIN IMMEDIATE")
                yield db
                db.commit()
            except BaseException:
                db.rollback()
                raise

    @staticmethod
    def _schema(db: sqlite3.Connection) -> None:
        db.execute("PRAGMA foreign_keys=ON")
        names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        required = {"execution_control", "execution_intents", "execution_fills", "execution_events"}
        present = names & required
        if present and present != required:
            raise ExecutionError("Схема исполнения неполна; восстановление автоматически запрещено")
        if present and db.execute("SELECT halted FROM execution_control WHERE id=1").fetchone() is None:
            raise ExecutionError("Запись аварийной остановки отсутствует; восстановление автоматически запрещено")
        db.execute("""CREATE TABLE IF NOT EXISTS execution_control (
            id INTEGER PRIMARY KEY CHECK (id=1), halted INTEGER NOT NULL CHECK (halted IN (0,1)),
            reason_code TEXT NOT NULL, updated_at_utc TEXT NOT NULL)""")
        db.execute("""CREATE TABLE IF NOT EXISTS execution_intents (
            client_order_id TEXT PRIMARY KEY, decision_id TEXT NOT NULL UNIQUE,
            candle_closed_at TEXT NOT NULL UNIQUE, mode TEXT NOT NULL CHECK (mode='DEMO'),
            inst_id TEXT NOT NULL CHECK (inst_id='BTC-USDT'),
            side TEXT NOT NULL CHECK (side='BUY'), state TEXT NOT NULL,
            quantity TEXT NOT NULL, entry_price TEXT NOT NULL, stop_price TEXT NOT NULL,
            take_profit_price TEXT NOT NULL, estimated_risk_usdt TEXT NOT NULL,
            min_size TEXT NOT NULL, lot_size TEXT NOT NULL, tick_size TEXT NOT NULL,
            created_at_utc TEXT NOT NULL, updated_at_utc TEXT NOT NULL,
            exchange_order_id TEXT UNIQUE, filled_quantity TEXT NOT NULL DEFAULT '0',
            fee_btc TEXT NOT NULL DEFAULT '0', fee_usdt TEXT NOT NULL DEFAULT '0',
            average_fill_price TEXT, reason_code TEXT NOT NULL)""")
        db.execute("""CREATE TABLE IF NOT EXISTS execution_fills (
            trade_id TEXT PRIMARY KEY, client_order_id TEXT NOT NULL,
            quantity TEXT NOT NULL, price TEXT NOT NULL, fee TEXT NOT NULL,
            fee_ccy TEXT NOT NULL CHECK (fee_ccy IN ('BTC','USDT')),
            observed_at_utc TEXT NOT NULL,
            FOREIGN KEY(client_order_id) REFERENCES execution_intents(client_order_id))""")
        db.execute("""CREATE TABLE IF NOT EXISTS execution_events (
            id INTEGER PRIMARY KEY, client_order_id TEXT, from_state TEXT, to_state TEXT NOT NULL,
            reason_code TEXT NOT NULL, at_utc TEXT NOT NULL)""")
        if not present:
            db.execute("INSERT INTO execution_control VALUES (1,0,'CLEAR',?)",
                       (_utc(datetime.now(timezone.utc), "Время"),))

    @staticmethod
    def _intent(row: sqlite3.Row) -> ExecutionIntent:
        try:
            if (not _CLIENT_ID.fullmatch(row["client_order_id"])
                    or row["mode"] != "DEMO" or row["inst_id"] != "BTC-USDT" or row["side"] != "BUY"):
                raise ExecutionError("Журнал содержит неразрешённый инструмент, режим или идентификатор")
            intent = ExecutionIntent(
                client_order_id=row["client_order_id"], decision_id=row["decision_id"],
                candle_closed_at=_parse_utc(row["candle_closed_at"]),
                state=ExecutionState(row["state"]),
                quantity=_decimal(row["quantity"], "Размер", positive=True),
                entry_price=_decimal(row["entry_price"], "Вход", positive=True),
                stop_price=_decimal(row["stop_price"], "Стоп", positive=True),
                take_profit_price=_decimal(row["take_profit_price"], "Тейк", positive=True),
                estimated_risk_usdt=_decimal(row["estimated_risk_usdt"], "Риск", positive=True),
                min_size=_decimal(row["min_size"], "Минимальный размер", positive=True),
                lot_size=_decimal(row["lot_size"], "Шаг", positive=True),
                tick_size=_decimal(row["tick_size"], "Ценовой шаг", positive=True),
                created_at_utc=_parse_utc(row["created_at_utc"]),
                updated_at_utc=_parse_utc(row["updated_at_utc"]),
                exchange_order_id=row["exchange_order_id"],
                filled_quantity=_decimal(row["filled_quantity"], "Исполненный размер"),
                fee_btc=_decimal(row["fee_btc"], "Комиссия BTC"),
                fee_usdt=_decimal(row["fee_usdt"], "Комиссия USDT"),
                average_fill_price=(_decimal(row["average_fill_price"], "Средняя цена", positive=True)
                                    if row["average_fill_price"] is not None else None),
                reason_code=row["reason_code"],
            )
            if (intent.filled_quantity < 0 or intent.filled_quantity > intent.quantity
                    or intent.owned_btc < 0 or intent.stop_price >= intent.entry_price
                    or intent.take_profit_price <= intent.entry_price
                    or not isinstance(intent.decision_id, str) or not intent.decision_id
                    or not isinstance(intent.reason_code, str) or not intent.reason_code):
                raise ExecutionError("Журнал исполнения содержит противоречивые величины")
            return intent
        except (KeyError, TypeError, ValueError, InvalidOperation) as exc:
            raise ExecutionError("Журнал исполнения повреждён или противоречив") from exc

    @staticmethod
    def _get(db: sqlite3.Connection, client_order_id: str) -> ExecutionIntent:
        row = db.execute("SELECT * FROM execution_intents WHERE client_order_id=?", (client_order_id,)).fetchone()
        if row is None:
            raise ExecutionError("Неизвестный clientOrderId; действие заблокировано")
        return ExecutionEngine._intent(row)

    def diagnose(self) -> ExecutionDiagnostic:
        """Offline and read-only, including when no intent table exists yet."""
        if not self.path.is_file():
            return ExecutionDiagnostic(True, "SQLite-журнал решений отсутствует", False, ())
        try:
            self.journal.audit_read_only()
            with closing(sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=3)) as db:
                db.row_factory = sqlite3.Row
                db.execute("PRAGMA query_only=ON")
                tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                present = {"execution_control", "execution_intents", "execution_fills", "execution_events"} & tables
                if not present:
                    return ExecutionDiagnostic(False, "Намерений нет; это только офлайн-диагностика", False, ())
                if len(present) != 4:
                    raise ExecutionError("Схема исполнения неполна")
                control = db.execute("SELECT halted,reason_code FROM execution_control WHERE id=1").fetchone()
                if control is None or control["halted"] not in (0, 1):
                    raise ExecutionError("Аварийное состояние неизвестно")
                intents = tuple(self._intent(row) for row in db.execute(
                    "SELECT * FROM execution_intents ORDER BY created_at_utc,client_order_id"))
                halted = bool(control["halted"])
                blocked = halted or any(intent.state in BLOCKING_STATES for intent in intents)
                reason = ("Аварийная остановка сохранена" if halted else
                          "Есть незавершённое или подтверждённое исполнение; новый вход заблокирован" if blocked else
                          "Незавершённых намерений нет; это только офлайн-диагностика")
                return ExecutionDiagnostic(blocked, reason, halted, intents)
        except (sqlite3.Error, OSError, JournalError, ExecutionError, TypeError, ValueError):
            return ExecutionDiagnostic(True, "Журнал исполнения недоступен или повреждён", True, ())

    def _risk_preview(
        self, report: ReconcileReport, instrument: InstrumentLimits,
        costs: ExecutionCosts, entry_price: Decimal, now: datetime,
    ):
        if not isinstance(report, ReconcileReport):
            raise ExecutionError("Сверка Demo отсутствует")
        for field in ("strategy_capital_usdt", "strategy_equity_usdt", "external_btc",
                      "btc_total", "btc_available", "bot_btc", "usdt_total", "usdt_available",
                      "confirmed_realized_pnl_usdt", "confirmed_unrealized_pnl_usdt",
                      "daily_loss_usdt", "daily_loss_limit_usdt", "daily_risk_remaining_usdt",
                      "per_trade_risk_limit_usdt", "taker_fee_rate"):
            value = getattr(report, field)
            if not isinstance(value, Decimal) or not value.is_finite():
                raise ExecutionError("Сверка Demo содержит неизвестные или некорректные величины")
        if (report.status != ReconcileStatus.READY
                or not isinstance(report.checked_at_utc, datetime)
                or report.checked_at_utc.tzinfo is None
                or report.checked_at_utc.utcoffset() != timedelta(0)
                or report.checked_at_utc > now
                or now - report.checked_at_utc > timedelta(seconds=60)
                or report.strategy_capital_usdt != STRATEGY_BUDGET_USDT
                or report.strategy_equity_usdt != STRATEGY_BUDGET_USDT
                or report.confirmed_realized_pnl_usdt != 0
                or report.confirmed_unrealized_pnl_usdt != 0
                or report.daily_loss_usdt != 0
                or report.daily_loss_limit_usdt != Decimal("100")
                or report.daily_risk_remaining_usdt != Decimal("100")
                or report.per_trade_risk_limit_usdt != Decimal("25")
                or not isinstance(report.external_btc, Decimal) or report.external_btc <= 0
                or report.btc_total != report.btc_available
                or report.btc_total != report.external_btc
                or report.bot_btc != 0 or report.open_order_count != 0
                or report.fill_count != 0 or not isinstance(report.usdt_available, Decimal)
                or report.usdt_available < STRATEGY_BUDGET_USDT
                or report.usdt_total != report.usdt_available):
            raise ExecutionError("Свежая полная сверка Demo-стратегии отсутствует")
        if (not isinstance(costs, ExecutionCosts) or not isinstance(costs.entry_fee_rate, Decimal)
                or not isinstance(costs.exit_fee_rate, Decimal)
                or not costs.entry_fee_rate.is_finite() or not costs.exit_fee_rate.is_finite()
                or costs.entry_fee_rate < max(Decimal(0), -report.taker_fee_rate)
                or costs.exit_fee_rate < max(Decimal(0), -report.taker_fee_rate)):
            raise ExecutionError("Оценка комиссий ниже подтверждённой ставки OKX")
        if (not isinstance(instrument, InstrumentLimits) or instrument.instrument_id != "BTC-USDT"
                or instrument.state != "live" or report.instrument is None
                or instrument.min_size != report.instrument.min_size
                or instrument.lot_size != report.instrument.lot_size
                or instrument.tick_size != report.instrument.tick_size):
            raise ExecutionError("Ограничения BTC-USDT не согласованы со сверкой")
        try:
            baseline = self.budget_store.read()
            state = self.risk_manager.inspect_daily_state(now)
        except (BudgetError, RiskStateError, OSError) as exc:
            raise ExecutionError("Бюджет или дневное состояние Risk Manager не подтверждены") from exc
        if (baseline is None or baseline.strategy_capital_usdt != STRATEGY_BUDGET_USDT
                or baseline.demo_btc_total != report.external_btc
                or baseline.demo_usdt_total != report.usdt_total
                or state.start_capital_usdt != STRATEGY_BUDGET_USDT or state.pending_entry
                or state.loss_locked or self.risk_manager.settings != RiskSettings()):
            raise ExecutionError("Состояние риска не допускает нового плана")
        account = AccountSnapshot(
            as_of_utc=now, day_utc=now.date(), equity_usdt=STRATEGY_BUDGET_USDT,
            day_start_capital_usdt=STRATEGY_BUDGET_USDT,
            available_usdt=STRATEGY_BUDGET_USDT,
            realized_pnl_today_usdt=Decimal(0), unrealized_pnl_usdt=Decimal(0),
            open_positions=0, pending_orders=0, entries_today=state.entries_used,
        )
        decision = self.risk_manager.preview_buy(
            account=account, instrument=instrument, costs=costs,
            entry_price=entry_price, now_utc=now,
        )
        if decision.status != RiskStatus.ALLOW:
            raise ExecutionError("Risk Manager не подтвердил гипотетический BUY")
        return decision

    @staticmethod
    def _validate_protection(decision, instrument: InstrumentLimits) -> None:
        quantity, entry, stop, take = (decision.quantity, decision.entry_price,
                                       decision.stop_price, decision.take_profit_price)
        if any(not isinstance(value, Decimal) or not value.is_finite() or value <= 0
               for value in (quantity, entry, stop, take)):
            raise ExecutionError("Защитные уровни или размер неизвестны")
        minimum = _ceil_step(instrument.min_size, instrument.lot_size)
        if (quantity < minimum or quantity > instrument.max_size
                or quantity != _floor_step(quantity, instrument.lot_size)
                or any(price != _floor_step(price, instrument.tick_size) for price in (entry, stop, take))
                or stop != _floor_step(entry * Decimal("0.99"), instrument.tick_size)
                or take != _ceil_step(entry * Decimal("1.02"), instrument.tick_size)):
            raise ExecutionError("Размер, стоп 1% или тейк 2% нарушают ограничения OKX")

    def _decision_row(self, db: sqlite3.Connection, decision_id: str, now: datetime):
        row = db.execute("SELECT * FROM decisions WHERE id=?", (decision_id,)).fetchone()
        if row is None or row["status"] != "ALLOW" or row["signal"] != "BUY" or row["executed"] != 0 or row["reference_id"] is not None:
            raise ExecutionError("Нужен исходный BUY/ALLOW из локального журнала dry-run")
        created = _parse_utc(row["created_at_utc"])
        candle = _parse_utc(row["candle_closed_at"])
        if (created > now or now - created > timedelta(seconds=60)
                or candle > now or now - candle > timedelta(minutes=30)):
            raise ExecutionError("Решение или закрытая свеча устарели")
        return row, candle

    def plan_buy(
        self, *, decision_id: str, reconciliation: ReconcileReport,
        instrument: InstrumentLimits, costs: ExecutionCosts, entry_price: Decimal,
        max_size_btc_verified: bool, slippage_verified: bool,
    ) -> ExecutionIntent:
        """Persist one hypothetical BUY; no risk reservation and no network call."""
        if max_size_btc_verified is not True or slippage_verified is not True:
            raise ExecutionError("Лимит BTC и ожидаемое проскальзывание не подтверждены")
        now = self.clock()
        _utc(now, "Время плана")
        decision = self._risk_preview(reconciliation, instrument, costs, entry_price, now)
        self._validate_protection(decision, instrument)
        if not isinstance(decision_id, str) or not decision_id:
            raise ExecutionError("Идентификатор решения отсутствует")
        if not self.path.is_file():
            raise ExecutionError("SQLite-журнал решений отсутствует")
        try:
            self.journal.audit_read_only()
            with self._transaction() as db:
                self._schema(db)
                control = db.execute("SELECT halted FROM execution_control WHERE id=1").fetchone()
                if control is None or control["halted"] != 0:
                    raise ExecutionError("Аварийная остановка блокирует новые намерения")
                if db.execute("SELECT 1 FROM execution_intents WHERE state != 'REJECTED' LIMIT 1").fetchone():
                    raise ExecutionError("Есть незавершённое или исполненное намерение; новый вход запрещён")
                row, candle = self._decision_row(db, decision_id, now)
                for field, expected in (
                    ("quantity", decision.quantity), ("entry_price", decision.entry_price),
                    ("stop_price", decision.stop_price), ("take_profit_price", decision.take_profit_price),
                    ("estimated_risk_usdt", decision.estimated_risk_usdt),
                ):
                    if _decimal(row[field], field) != expected:
                        raise ExecutionError("План противоречит подтверждённому расчёту Risk Manager")
                # Re-read risk state under the SQLite write lock; a later stage
                # must revalidate and reserve it again before any submission.
                state = self.risk_manager.inspect_daily_state(now)
                if state.pending_entry or state.loss_locked or state.start_capital_usdt != STRATEGY_BUDGET_USDT:
                    raise ExecutionError("Состояние риска изменилось во время планирования")
                client_id = "TB" + uuid4().hex[:30]
                db.execute("""INSERT INTO execution_intents (
                    client_order_id,decision_id,candle_closed_at,mode,inst_id,side,state,
                    quantity,entry_price,stop_price,take_profit_price,estimated_risk_usdt,
                    min_size,lot_size,tick_size,created_at_utc,updated_at_utc,reason_code
                ) VALUES (?,?,?,'DEMO','BTC-USDT','BUY','PLANNED',?,?,?,?,?,?,?,?,?,?,'PLANNED')""", (
                    client_id, decision_id, _utc(candle, "Свеча"), str(decision.quantity),
                    str(decision.entry_price), str(decision.stop_price),
                    str(decision.take_profit_price), str(decision.estimated_risk_usdt),
                    str(instrument.min_size), str(instrument.lot_size), str(instrument.tick_size),
                    _utc(now, "Время"), _utc(now, "Время"),
                ))
                db.execute("INSERT INTO execution_events (client_order_id,from_state,to_state,reason_code,at_utc) VALUES (?,NULL,'PLANNED','PLANNED',?)",
                           (client_id, _utc(now, "Время")))
                intent = self._get(db, client_id)
                return intent
        except (sqlite3.Error, OSError, JournalError, RiskStateError, InvalidOperation, TypeError, ValueError) as exc:
            if isinstance(exc, ExecutionError):
                raise
            raise ExecutionError("Намерение не удалось безопасно сохранить; отправка запрещена") from exc

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        if not self.path.is_file():
            raise ExecutionError("SQLite-журнал решений отсутствует")
        try:
            self.journal.audit_read_only()
            with self._transaction() as db:
                self._schema(db)
                yield db
        except (sqlite3.Error, OSError, JournalError) as exc:
            raise ExecutionError("Состояние исполнения не удалось безопасно сохранить") from exc

    def _transition(self, db: sqlite3.Connection, intent: ExecutionIntent,
                    target: ExecutionState, reason: str, *, order_id: str | None = None) -> ExecutionIntent:
        if not re.fullmatch(r"[A-Z_]{3,40}", reason):
            raise ExecutionError("Некорректная причина перехода")
        now = _utc(self.clock(), "Время перехода")
        if order_id is not None and not _ORDER_ID.fullmatch(order_id):
            raise ExecutionError("Недопустимый идентификатор биржевого ордера")
        db.execute("""UPDATE execution_intents SET state=?,reason_code=?,updated_at_utc=?,
                   exchange_order_id=COALESCE(?,exchange_order_id) WHERE client_order_id=? AND state=?""",
                   (target.value, reason, now, order_id, intent.client_order_id, intent.state.value))
        if db.execute("SELECT changes()").fetchone()[0] != 1:
            raise ExecutionError("Состояние изменилось; необходима сверка")
        db.execute("INSERT INTO execution_events (client_order_id,from_state,to_state,reason_code,at_utc) VALUES (?,?,?,?,?)",
                   (intent.client_order_id, intent.state.value, target.value, reason, now))
        return self._get(db, intent.client_order_id)

    @staticmethod
    def _clear(db: sqlite3.Connection) -> None:
        row = db.execute("SELECT halted FROM execution_control WHERE id=1").fetchone()
        if row is None or row[0] != 0:
            raise ExecutionError("Аварийная остановка: требуется отдельная сверка")

    def validate_intent(self, client_order_id: str, *, reconciliation: ReconcileReport,
                        instrument: InstrumentLimits, costs: ExecutionCosts,
                        max_size_btc_verified: bool, slippage_verified: bool) -> ExecutionIntent:
        """Recheck the current risk decision; never reserves funds or sends an order."""
        if max_size_btc_verified is not True or slippage_verified is not True:
            raise ExecutionError("Ограничения количества и проскальзывание не подтверждены")
        now = self.clock()
        with self._write() as db:
            self._clear(db)
            intent = self._get(db, client_order_id)
            if intent.state != ExecutionState.PLANNED:
                raise ExecutionError("Повторная валидация недопустима; нужна сверка")
            if (intent.created_at_utc > now or now - intent.created_at_utc > timedelta(seconds=60)
                    or intent.candle_closed_at > now
                    or now - intent.candle_closed_at > timedelta(minutes=30)):
                raise ExecutionError("План или закрытая свеча устарели")
            decision = self._risk_preview(reconciliation, instrument, costs, intent.entry_price, now)
            self._validate_protection(decision, instrument)
            if (decision.quantity != intent.quantity or decision.entry_price != intent.entry_price
                    or decision.stop_price != intent.stop_price or decision.take_profit_price != intent.take_profit_price
                    or decision.estimated_risk_usdt != intent.estimated_risk_usdt
                    or instrument.min_size != intent.min_size or instrument.lot_size != intent.lot_size
                    or instrument.tick_size != intent.tick_size):
                raise ExecutionError("Повторная проверка Risk Manager расходится с сохранённым планом")
            return self._transition(db, intent, ExecutionState.VALIDATED, "RISK_REVALIDATED")

    def mark_submitting(self, client_order_id: str, *, risk_confirmation: RiskDecision) -> ExecutionIntent:
        """Persist ambiguous boundary before a hypothetical future transport call.

        There is no transport in this module. The durable Risk Manager reservation
        must already exist; an unreserved plan cannot cross this boundary.
        """
        now = self.clock()
        with self._write() as db:
            self._clear(db)
            intent = self._get(db, client_order_id)
            if intent.state != ExecutionState.VALIDATED:
                raise ExecutionError("Отправка без подтверждённого состояния запрещена")
            if (intent.updated_at_utc > now or now - intent.updated_at_utc > timedelta(seconds=60)
                    or intent.candle_closed_at > now
                    or now - intent.candle_closed_at > timedelta(minutes=30)):
                raise ExecutionError("Валидация или закрытая свеча устарели")
            if (not isinstance(risk_confirmation, RiskDecision)
                    or risk_confirmation.status != RiskStatus.ALLOW
                    or risk_confirmation.quantity != intent.quantity
                    or risk_confirmation.entry_price != intent.entry_price
                    or risk_confirmation.stop_price != intent.stop_price
                    or risk_confirmation.take_profit_price != intent.take_profit_price
                    or risk_confirmation.estimated_risk_usdt != intent.estimated_risk_usdt):
                raise ExecutionError("Risk Manager не подтвердил именно это намерение")
            try:
                state = self.risk_manager.inspect_daily_state(now)
            except RiskStateError as exc:
                raise ExecutionError("Резерв Risk Manager не подтверждён") from exc
            if not state.pending_entry or state.start_capital_usdt != STRATEGY_BUDGET_USDT or state.loss_locked:
                raise ExecutionError("Резерв Risk Manager не подтверждён")
            return self._transition(db, intent, ExecutionState.SUBMITTING, "PRE_SEND_BOUNDARY")

    def record_ack(self, client_order_id: str, exchange_order_id: str) -> ExecutionIntent:
        with self._write() as db:
            self._clear(db)
            intent = self._get(db, client_order_id)
            if intent.state != ExecutionState.SUBMITTING:
                raise ExecutionError("ACK допустим только после SUBMITTING")
            return self._transition(db, intent, ExecutionState.ACKNOWLEDGED, "EXCHANGE_ACK", order_id=exchange_order_id)

    def record_uncertain(self, client_order_id: str, *, timeout: bool) -> ExecutionIntent:
        """Timeout/ambiguous response is UNKNOWN; never retry the order."""
        with self._write() as db:
            self._clear(db)
            intent = self._get(db, client_order_id)
            if intent.state not in (ExecutionState.SUBMITTING, ExecutionState.ACKNOWLEDGED,
                                    ExecutionState.PARTIALLY_FILLED):
                raise ExecutionError("Неоднозначный ответ требует известного ожидающего намерения")
            return self._transition(db, intent, ExecutionState.UNKNOWN,
                                    "NETWORK_TIMEOUT" if timeout else "AMBIGUOUS_RESPONSE")

    def record_rejection(self, client_order_id: str, *, proof: RecoveryProof | None = None) -> ExecutionIntent:
        """Discard local plan, or close uncertain send only with per-ID read-only proof."""
        with self._write() as db:
            self._clear(db)
            intent = self._get(db, client_order_id)
            if intent.state not in (ExecutionState.PLANNED, ExecutionState.VALIDATED,
                                    ExecutionState.SUBMITTING, ExecutionState.UNKNOWN) or intent.filled_quantity:
                raise ExecutionError("Отказ противоречит наблюдаемому исполнению")
            if intent.state in (ExecutionState.SUBMITTING, ExecutionState.UNKNOWN):
                self._check_proof(proof, client_order_id, self.clock())
                reason = "RECONCILED_NO_FILL"
            else:
                reason = "LOCAL_PLAN_REJECTED"
            return self._transition(db, intent, ExecutionState.REJECTED, reason)

    def record_fill(self, client_order_id: str, *, trade_id: str, quantity: Decimal,
                    price: Decimal, fee: Decimal, fee_ccy: str,
                    exchange_order_id: str) -> ExecutionIntent:
        """Record externally observed fill; fees use signed OKX convention."""
        quantity = _decimal(quantity, "Объём исполнения", positive=True)
        price = _decimal(price, "Цена исполнения", positive=True)
        fee = _decimal(fee, "Комиссия")
        if (not isinstance(trade_id, str) or not _ORDER_ID.fullmatch(trade_id)
                or not isinstance(exchange_order_id, str) or not _ORDER_ID.fullmatch(exchange_order_id)
                or fee_ccy not in ("BTC", "USDT") or fee > 0):
            raise ExecutionError("Исполнение или комиссия не подтверждены")
        with self._write() as db:
            intent = self._get(db, client_order_id)
            control = db.execute("SELECT halted FROM execution_control WHERE id=1").fetchone()
            if control is None or (control[0] == 1 and intent.state != ExecutionState.HALTED):
                raise ExecutionError("Аварийная остановка требует сверки")
            if intent.state not in (ExecutionState.SUBMITTING, ExecutionState.ACKNOWLEDGED,
                                    ExecutionState.PARTIALLY_FILLED, ExecutionState.UNKNOWN,
                                    ExecutionState.HALTED) or (intent.exchange_order_id is not None
                                    and intent.exchange_order_id != exchange_order_id):
                raise ExecutionError("Исполнение расходится с сохранённым намерением")
            previous = db.execute("SELECT client_order_id,quantity,price,fee,fee_ccy FROM execution_fills WHERE trade_id=?",
                                  (trade_id,)).fetchone()
            if previous is not None:
                if tuple(previous) == (client_order_id, str(quantity), str(price), str(fee), fee_ccy):
                    return intent
                self._halt(db, "CONFLICTING_TRADE_ID", client_order_id)
                return self._get(db, client_order_id)
            new_filled = intent.filled_quantity + quantity
            new_fee_btc = intent.fee_btc + (fee if fee_ccy == "BTC" else Decimal(0))
            new_fee_usdt = intent.fee_usdt + (fee if fee_ccy == "USDT" else Decimal(0))
            if new_filled > intent.quantity or new_filled + new_fee_btc <= 0:
                self._halt(db, "INVALID_FILL_TOTAL", client_order_id)
                return self._get(db, client_order_id)
            db.execute("INSERT INTO execution_fills VALUES (?,?,?,?,?,?,?)", (trade_id, client_order_id,
                       str(quantity), str(price), str(fee), fee_ccy, _utc(self.clock(), "Исполнение")))
            previous_notional = ((intent.average_fill_price or Decimal(0)) * intent.filled_quantity)
            average = (previous_notional + price * quantity) / new_filled
            db.execute("""UPDATE execution_intents SET filled_quantity=?,fee_btc=?,fee_usdt=?,
                       average_fill_price=?,exchange_order_id=? WHERE client_order_id=?""",
                       (str(new_filled), str(new_fee_btc), str(new_fee_usdt), str(average),
                        exchange_order_id, client_order_id))
            # Protection cannot be promised for a sub-minimum/step-incompatible net fill.
            net = new_filled + new_fee_btc
            if net < intent.min_size or net != _floor_step(net, intent.lot_size):
                self._halt(db, "UNPROTECTABLE_PARTIAL_FILL", client_order_id)
                return self._get(db, client_order_id)
            if intent.state == ExecutionState.HALTED:
                db.execute("INSERT INTO execution_events (client_order_id,from_state,to_state,reason_code,at_utc) VALUES (?,'HALTED','HALTED','FILL_OBSERVED_HALTED',?)",
                           (client_order_id, _utc(self.clock(), "Исполнение")))
                return self._get(db, client_order_id)
            target = ExecutionState.FILLED if new_filled == intent.quantity else ExecutionState.PARTIALLY_FILLED
            return self._transition(db, intent, target, "FILL_OBSERVED", order_id=exchange_order_id)

    def _halt(self, db: sqlite3.Connection, reason: str, client_order_id: str | None = None) -> None:
        now = _utc(self.clock(), "Аварийная остановка")
        db.execute("UPDATE execution_control SET halted=1,reason_code=?,updated_at_utc=? WHERE id=1",
                   (reason, now))
        targets = ([self._get(db, client_order_id)] if client_order_id is not None else
                   [self._intent(row) for row in db.execute("SELECT * FROM execution_intents WHERE state != 'REJECTED'")])
        for intent in targets:
            if intent.state != ExecutionState.HALTED:
                self._transition(db, intent, ExecutionState.HALTED, reason)
        if not targets:
            db.execute("INSERT INTO execution_events (client_order_id,from_state,to_state,reason_code,at_utc) VALUES (NULL,NULL,'HALTED',?,?)",
                       (reason, now))

    def emergency_halt(self, reason_code: str = "MANUAL_HALT") -> None:
        if not re.fullmatch(r"[A-Z_]{3,40}", reason_code):
            raise ExecutionError("Недопустимая причина остановки")
        with self._write() as db:
            self._halt(db, reason_code)

    @staticmethod
    def _check_proof(proof: RecoveryProof | None, client_order_id: str | None,
                     now: datetime) -> None:
        if (not isinstance(proof, RecoveryProof) or proof.client_order_id != client_order_id
                or proof.source != "OKX_DEMO_READ_ONLY" or proof.no_exchange_order is not True
                or proof.no_exchange_fills is not True or proof.balances_reconciled is not True
                or not isinstance(proof.checked_at_utc, datetime)
                or proof.checked_at_utc.tzinfo is None or proof.checked_at_utc.utcoffset() != timedelta(0)
                or proof.checked_at_utc > now or now - proof.checked_at_utc > timedelta(seconds=60)):
            raise ExecutionError("Требуется свежая сверка конкретного Demo clientOrderId")

    def manual_recover(self, *, phrase: str, proof: RecoveryProof) -> None:
        """Local unlock only after per-ID read-only Demo proof; no CLI shortcut."""
        now = self.clock()
        if phrase != RECOVERY_PHRASE or not isinstance(proof, RecoveryProof):
            raise ExecutionError("Ручное восстановление требует свежей сверки конкретного Demo-ордера")
        self._check_proof(proof, proof.client_order_id, now)
        with self._write() as db:
            row = db.execute("SELECT halted FROM execution_control WHERE id=1").fetchone()
            if row is None or row[0] != 1:
                raise ExecutionError("Аварийная остановка отсутствует")
            intents = tuple(self._intent(row) for row in db.execute("SELECT * FROM execution_intents WHERE state != 'REJECTED'"))
            if (len(intents) > 1 or (intents and (proof.client_order_id != intents[0].client_order_id
                    or intents[0].filled_quantity != 0)) or (not intents and proof.client_order_id is not None)):
                raise ExecutionError("Сверка не покрывает все намерения или подтверждённые исполнения")
            if intents:
                self._transition(db, intents[0], ExecutionState.REJECTED, "MANUAL_RECONCILED_NO_FILL")
            db.execute("UPDATE execution_control SET halted=0,reason_code='MANUAL_RECONCILED',updated_at_utc=? WHERE id=1",
                       (_utc(now, "Восстановление"),))
