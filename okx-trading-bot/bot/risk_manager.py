"""Offline, fail-closed SPOT buy sizing with durable UTC daily limits.

This module never calls an exchange and never creates an order. Its account,
instrument, and fee inputs must come from a trusted, complete source.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from decimal import Decimal, DecimalException, ROUND_CEILING, ROUND_DOWN
from enum import StrEnum
import json
import os
from pathlib import Path
import tempfile
from typing import Iterator


ZERO = Decimal("0")
ONE = Decimal("1")
STATE_VERSION = 1
DEFAULT_STATE_PATH = Path(__file__).resolve().parents[1] / "data" / "risk_state.json"


class RiskStatus(StrEnum):
    ALLOW = "ALLOW"
    DENY = "DENY"


class RiskStateError(ValueError):
    """The durable daily state is missing, corrupt, or unsafe to update."""


@dataclass(frozen=True)
class RiskSettings:
    risk_per_trade: Decimal = Decimal("0.005")
    stop_loss_pct: Decimal = Decimal("0.01")
    take_profit_pct: Decimal = Decimal("0.02")
    daily_loss_limit_pct: Decimal = Decimal("0.02")
    max_open_positions: int = 1
    max_new_entries_per_day: int = 5
    max_snapshot_age_seconds: int = 60


@dataclass(frozen=True)
class AccountSnapshot:
    as_of_utc: datetime
    day_utc: date
    equity_usdt: Decimal
    day_start_capital_usdt: Decimal
    available_usdt: Decimal
    realized_pnl_today_usdt: Decimal
    unrealized_pnl_usdt: Decimal
    open_positions: int
    pending_orders: int
    entries_today: int


@dataclass(frozen=True)
class InstrumentLimits:
    instrument_id: str
    state: str
    min_size: Decimal
    lot_size: Decimal
    tick_size: Decimal
    max_size: Decimal


@dataclass(frozen=True)
class ExecutionCosts:
    entry_fee_rate: Decimal
    exit_fee_rate: Decimal
    entry_slippage_rate: Decimal
    stop_slippage_rate: Decimal


@dataclass(frozen=True)
class RiskDecision:
    status: RiskStatus
    reason: str
    quantity: Decimal = ZERO
    entry_price: Decimal | None = None
    stop_price: Decimal | None = None
    take_profit_price: Decimal | None = None
    estimated_risk_usdt: Decimal = ZERO
    estimated_cost_usdt: Decimal = ZERO


@dataclass(frozen=True)
class _DailyState:
    day_utc: date
    start_capital_usdt: Decimal
    entries_used: int
    pending_entry: bool
    loss_locked: bool


@dataclass(frozen=True)
class DailyRiskSnapshot:
    """Read-only view of a previously, explicitly initialized UTC day."""

    day_utc: date
    start_capital_usdt: Decimal
    entries_used: int
    pending_entry: bool
    loss_locked: bool


def _decimal(value: object, name: str, *, positive: bool = False) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise RiskStateError(f"{name}: требуется конечное значение Decimal")
    if positive and value <= ZERO:
        raise RiskStateError(f"{name}: значение должно быть больше нуля")
    return value


def _rate(value: object, name: str, *, positive: bool = False) -> Decimal:
    number = _decimal(value, name)
    if number < ZERO or number >= ONE or (positive and number == ZERO):
        raise RiskStateError(f"{name}: ставка должна быть от 0 до 1")
    return number


def _count(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise RiskStateError(f"{name}: требуется неотрицательное целое число")
    return value


def _utc_time(value: object, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise RiskStateError(f"{name}: требуется время с часовым поясом UTC")
    return value


def _floor_step(value: Decimal, step: Decimal) -> Decimal:
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def _ceil_step(value: Decimal, step: Decimal) -> Decimal:
    return (value / step).to_integral_value(rounding=ROUND_CEILING) * step


class RiskManager:
    """Persists risk locks and entry reservations without performing trades."""

    def __init__(self, state_path: Path = DEFAULT_STATE_PATH, settings: RiskSettings | None = None) -> None:
        self.state_path = Path(state_path)
        self.settings = settings if settings is not None else RiskSettings()

    def _validate_settings(self) -> None:
        if not isinstance(self.settings, RiskSettings):
            raise RiskStateError("Настройки риска отсутствуют")
        for name in ("risk_per_trade", "stop_loss_pct", "take_profit_pct", "daily_loss_limit_pct"):
            _rate(getattr(self.settings, name), name, positive=True)
        if self.settings.max_open_positions != 1 or type(self.settings.max_open_positions) is not int:
            raise RiskStateError("Разрешена максимум одна открытая позиция, усреднение запрещено")
        if type(self.settings.max_new_entries_per_day) is not int or self.settings.max_new_entries_per_day < 1:
            raise RiskStateError("Некорректен максимум входов за сутки")
        if type(self.settings.max_snapshot_age_seconds) is not int or self.settings.max_snapshot_age_seconds < 1:
            raise RiskStateError("Некорректен срок актуальности снимка счёта")

    def validate_settings(self) -> None:
        """Check risk settings without reading or mutating daily state."""
        self._validate_settings()

    def inspect_daily_state(self, now_utc: datetime) -> DailyRiskSnapshot:
        """Read existing limits without initializing a day or changing a lock."""
        self._validate_settings()
        now = _utc_time(now_utc, "Текущее время")
        lock_path = Path(f"{self.state_path}.lock")
        if lock_path.exists():
            raise RiskStateError("Состояние риска занято или требует восстановления")
        state = self._load()
        if lock_path.exists() or state.day_utc != now.date():
            raise RiskStateError("Нет подтверждённого состояния риска для текущих суток UTC")
        return DailyRiskSnapshot(state.day_utc, state.start_capital_usdt,
                                 state.entries_used, state.pending_entry, state.loss_locked)

    def _validate_account(self, account: AccountSnapshot, now_utc: datetime) -> None:
        if not isinstance(account, AccountSnapshot):
            raise RiskStateError("Состояние счёта неизвестно")
        observed = _utc_time(account.as_of_utc, "Время снимка счёта")
        if observed > now_utc or now_utc - observed > timedelta(seconds=self.settings.max_snapshot_age_seconds):
            raise RiskStateError("Снимок счёта отсутствует или устарел")
        if type(account.day_utc) is not date or account.day_utc != now_utc.date() or observed.date() != account.day_utc:
            raise RiskStateError("Снимок счёта не относится к текущим суткам UTC")
        equity = _decimal(account.equity_usdt, "Капитал", positive=True)
        _decimal(account.day_start_capital_usdt, "Капитал на начало суток", positive=True)
        available = _decimal(account.available_usdt, "Доступный USDT")
        if available < ZERO or available > equity:
            raise RiskStateError("Доступный USDT противоречит капиталу счёта")
        _decimal(account.realized_pnl_today_usdt, "Реализованный PnL")
        _decimal(account.unrealized_pnl_usdt, "Нереализованный PnL")
        for name in ("open_positions", "pending_orders", "entries_today"):
            _count(getattr(account, name), name)

    @staticmethod
    def _validate_instrument(instrument: InstrumentLimits) -> None:
        if not isinstance(instrument, InstrumentLimits) or instrument.instrument_id != "BTC-USDT":
            raise RiskStateError("Неизвестен SPOT-инструмент BTC-USDT")
        if instrument.state != "live":
            raise RiskStateError("Инструмент OKX недоступен для торговли")
        for name in ("min_size", "lot_size", "tick_size", "max_size"):
            _decimal(getattr(instrument, name), name, positive=True)
        if instrument.max_size < instrument.min_size:
            raise RiskStateError("Максимальный размер инструмента меньше минимального")

    @staticmethod
    def _validate_costs(costs: ExecutionCosts) -> None:
        if not isinstance(costs, ExecutionCosts):
            raise RiskStateError("Комиссии или проскальзывание неизвестны")
        for name in ("entry_fee_rate", "exit_fee_rate", "entry_slippage_rate", "stop_slippage_rate"):
            _rate(getattr(costs, name), name)

    @staticmethod
    def _daily_loss(account: AccountSnapshot) -> Decimal:
        # Gains do not offset losses from the other component.
        return max(ZERO, -account.realized_pnl_today_usdt) + max(ZERO, -account.unrealized_pnl_usdt)

    @contextmanager
    def _exclusive(self) -> Iterator[None]:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = Path(f"{self.state_path}.lock")
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise RiskStateError("Состояние риска занято или требует восстановления") from exc
        os.close(fd)
        keep_lock = False
        try:
            yield
        except OSError:
            # A failed state write must leave a durable fail-closed marker.
            keep_lock = True
            raise
        finally:
            if not keep_lock:
                lock_path.unlink(missing_ok=True)

    def _load(self) -> _DailyState:
        if not self.state_path.is_file():
            raise RiskStateError("Дневное состояние риска отсутствует; нужна явная инициализация")
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or set(payload) != {
                "version", "day_utc", "start_capital_usdt", "entries_used", "pending_entry", "loss_locked"
            }:
                raise ValueError("bad state fields")
            if type(payload["version"]) is not int or payload["version"] != STATE_VERSION:
                raise ValueError("bad state version")
            if not isinstance(payload["day_utc"], str) or not isinstance(payload["start_capital_usdt"], str):
                raise ValueError("bad state types")
            day = date.fromisoformat(payload["day_utc"])
            capital = Decimal(payload["start_capital_usdt"])
            _decimal(capital, "Капитал на начало суток", positive=True)
            entries = _count(payload["entries_used"], "entries_used")
            if type(payload["pending_entry"]) is not bool or type(payload["loss_locked"]) is not bool:
                raise ValueError("bad state flags")
            return _DailyState(day, capital, entries, payload["pending_entry"], payload["loss_locked"])
        except (ValueError, TypeError, KeyError, DecimalException) as exc:
            raise RiskStateError("Дневное состояние риска повреждено") from exc

    def _save(self, state: _DailyState) -> None:
        payload = {
            "version": STATE_VERSION,
            "day_utc": state.day_utc.isoformat(),
            "start_capital_usdt": str(state.start_capital_usdt),
            "entries_used": state.entries_used,
            "pending_entry": state.pending_entry,
            "loss_locked": state.loss_locked,
        }
        fd, temporary = tempfile.mkstemp(prefix=".risk-state-", dir=self.state_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.state_path)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def initialize_day(self, account: AccountSnapshot, now_utc: datetime) -> None:
        """Explicitly establish a UTC day from a trusted complete account snapshot.

        Calling this again on the same day cannot clear reservations or a loss lock.
        """
        self._validate_settings()
        now = _utc_time(now_utc, "Текущее время")
        self._validate_account(account, now)
        with self._exclusive():
            if self.state_path.exists():
                old = self._load()
                if old.day_utc >= account.day_utc:
                    raise RiskStateError("Дневное состояние уже создано; сброс в те же сутки запрещён")
            locked = self._daily_loss(account) >= account.day_start_capital_usdt * self.settings.daily_loss_limit_pct
            self._save(_DailyState(account.day_utc, account.day_start_capital_usdt, account.entries_today, False, locked))

    def _prepared_state(self, account: AccountSnapshot, now: datetime, state: _DailyState) -> _DailyState:
        if state.day_utc != now.date():
            raise RiskStateError("Нет состояния для текущих суток UTC")
        if state.start_capital_usdt != account.day_start_capital_usdt:
            raise RiskStateError("Капитал на начало суток противоречит сохранённому значению")
        if not state.pending_entry and account.entries_today < state.entries_used:
            raise RiskStateError("История входов противоречит сохранённому счётчику")
        if state.pending_entry and account.entries_today > state.entries_used:
            raise RiskStateError("При зарезервированном входе обнаружены другие входы; нужна сверка")
        return replace(state, entries_used=max(state.entries_used, account.entries_today))

    def _evaluate_buy(
        self,
        account: AccountSnapshot,
        instrument: InstrumentLimits | None,
        costs: ExecutionCosts | None,
        entry_price: Decimal | None,
        state: _DailyState,
        *,
        preview: bool,
    ) -> tuple[RiskDecision, _DailyState]:
        daily_limit = state.start_capital_usdt * self.settings.daily_loss_limit_pct
        daily_loss = self._daily_loss(account)
        if state.loss_locked:
            return RiskDecision(RiskStatus.DENY, "Дневной лимит убытка уже заблокировал новые входы"), state
        if daily_loss >= daily_limit:
            return RiskDecision(RiskStatus.DENY, "Достигнут дневной лимит реализованного и нереализованного убытка"), replace(state, loss_locked=True)
        if account.open_positions >= self.settings.max_open_positions:
            return RiskDecision(RiskStatus.DENY, "Есть открытая позиция; усреднение и новый вход запрещены"), state
        if account.pending_orders:
            return RiskDecision(RiskStatus.DENY, "Есть незавершённый ордер; состояние входа не определено"), state
        if state.pending_entry:
            return RiskDecision(RiskStatus.DENY, "Предыдущий вход уже зарезервирован"), state
        if state.entries_used >= self.settings.max_new_entries_per_day:
            return RiskDecision(RiskStatus.DENY, "Достигнут суточный лимит новых входов"), state

        self._validate_instrument(instrument)
        self._validate_costs(costs)
        price = _decimal(entry_price, "Цена входа", positive=True)
        assert instrument is not None and costs is not None
        entry = _ceil_step(price, instrument.tick_size)
        stop = _floor_step(entry * (ONE - self.settings.stop_loss_pct), instrument.tick_size)
        take_profit = _ceil_step(entry * (ONE + self.settings.take_profit_pct), instrument.tick_size)
        if stop <= ZERO or stop >= entry or take_profit <= entry:
            raise RiskStateError("Ценовой шаг делает стоп или тейк некорректным")

        entry_cost_per_unit = entry * (ONE + costs.entry_slippage_rate) * (ONE + costs.entry_fee_rate)
        stop_proceeds_per_unit = stop * (ONE - costs.stop_slippage_rate) * (ONE - costs.exit_fee_rate)
        loss_per_unit = entry_cost_per_unit - stop_proceeds_per_unit
        if loss_per_unit <= ZERO:
            raise RiskStateError("Невозможно оценить убыток на единицу позиции")
        risk_budget = min(account.equity_usdt * self.settings.risk_per_trade, daily_limit - daily_loss)
        minimum = _ceil_step(instrument.min_size, instrument.lot_size)
        maximum = _floor_step(instrument.max_size, instrument.lot_size)
        if minimum > maximum:
            raise RiskStateError("Минимальный лот выше максимального размера OKX")
        if minimum * loss_per_unit > risk_budget:
            return RiskDecision(RiskStatus.DENY, "Минимальный лот превышает допустимый риск", entry_price=entry, stop_price=stop, take_profit_price=take_profit), state
        if minimum * entry_cost_per_unit > account.available_usdt:
            return RiskDecision(RiskStatus.DENY, "Недостаточно доступного USDT даже для минимального лота", entry_price=entry, stop_price=stop, take_profit_price=take_profit), state

        quantity = min(
            _floor_step(risk_budget / loss_per_unit, instrument.lot_size),
            _floor_step(account.available_usdt / entry_cost_per_unit, instrument.lot_size),
            maximum,
        )
        if quantity < minimum:
            raise RiskStateError("Размер после округления ниже минимального лота")
        estimated_risk = quantity * loss_per_unit
        estimated_cost = quantity * entry_cost_per_unit
        if estimated_risk > risk_budget or estimated_cost > account.available_usdt or quantity > maximum:
            raise RiskStateError("Округлённый размер нарушает лимит риска, баланса или инструмента")
        decision = RiskDecision(
            status=RiskStatus.ALLOW,
            reason=("Риск и лимиты соблюдены; гипотетический вход без резервирования"
                    if preview else "Риск и лимиты соблюдены; один вход зарезервирован без отправки ордера"),
            quantity=quantity,
            entry_price=entry,
            stop_price=stop,
            take_profit_price=take_profit,
            estimated_risk_usdt=estimated_risk,
            estimated_cost_usdt=estimated_cost,
        )
        return decision, replace(state, entries_used=state.entries_used + 1, pending_entry=True)

    def assess_buy(
        self,
        *,
        account: AccountSnapshot | None = None,
        instrument: InstrumentLimits | None = None,
        costs: ExecutionCosts | None = None,
        entry_price: Decimal | None = None,
        now_utc: datetime | None = None,
    ) -> RiskDecision:
        """Return ALLOW only after a durable, single-use entry reservation."""
        try:
            self._validate_settings()
            now = _utc_time(now_utc, "Текущее время")
            self._validate_account(account, now)
            assert account is not None
            with self._exclusive():
                state = self._load()
                prepared = self._prepared_state(account, now, state)
                if prepared != state:
                    state = prepared
                    self._save(state)
                decision, updated = self._evaluate_buy(account, instrument, costs, entry_price, state, preview=False)
                if updated != state:
                    self._save(updated)
                return decision
        except RiskStateError as exc:
            return RiskDecision(RiskStatus.DENY, f"Проверка риска отклонена: {exc}")
        except (OSError, DecimalException, OverflowError, TypeError, ValueError):
            return RiskDecision(RiskStatus.DENY, "Проверка риска отклонена: некорректные данные или ошибка чтения состояния")

    def preview_buy(
        self,
        *,
        account: AccountSnapshot | None = None,
        instrument: InstrumentLimits | None = None,
        costs: ExecutionCosts | None = None,
        entry_price: Decimal | None = None,
        now_utc: datetime | None = None,
    ) -> RiskDecision:
        """Evaluate a hypothetical BUY without locks, reservations or state writes."""
        try:
            self._validate_settings()
            now = _utc_time(now_utc, "Текущее время")
            self._validate_account(account, now)
            assert account is not None
            lock_path = Path(f"{self.state_path}.lock")
            if lock_path.exists():
                raise RiskStateError("Состояние риска занято или требует восстановления")
            state = self._load()
            if lock_path.exists():
                raise RiskStateError("Состояние риска занято или требует восстановления")
            prepared = self._prepared_state(account, now, state)
            decision, _ = self._evaluate_buy(account, instrument, costs, entry_price, prepared, preview=True)
            return decision
        except RiskStateError as exc:
            return RiskDecision(RiskStatus.DENY, f"Проверка риска отклонена: {exc}")
        except (OSError, DecimalException, OverflowError, TypeError, ValueError):
            return RiskDecision(RiskStatus.DENY, "Проверка риска отклонена: некорректные данные или ошибка чтения состояния")

    def reconcile_confirmed_entry(self, account: AccountSnapshot, now_utc: datetime) -> bool:
        """Clear a reservation only after trusted account history confirms its entry.

        A canceled or uncertain entry remains reserved for the UTC day.
        """
        try:
            self._validate_settings()
            now = _utc_time(now_utc, "Текущее время")
            self._validate_account(account, now)
            with self._exclusive():
                state = self._load()
                if (
                    state.day_utc != now.date()
                    or state.start_capital_usdt != account.day_start_capital_usdt
                    or not state.pending_entry
                    or account.entries_today != state.entries_used
                    or account.pending_orders
                ):
                    return False
                locked = state.loss_locked or self._daily_loss(account) >= state.start_capital_usdt * self.settings.daily_loss_limit_pct
                self._save(replace(state, entries_used=max(state.entries_used, account.entries_today), pending_entry=False, loss_locked=locked))
                return True
        except (RiskStateError, OSError, DecimalException, OverflowError, TypeError, ValueError):
            return False
