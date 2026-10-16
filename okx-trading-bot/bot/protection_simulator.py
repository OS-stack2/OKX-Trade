"""Offline, temporary-SQLite lifecycle simulation. No exchange or working DB IO.

The model deliberately remains BLOCK even when its synthetic OCO evidence is
complete: it cannot establish real OKX Demo partial-fill atomicity or region
support. All prices, fills and Algo IDs here are invented scenario inputs.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_DOWN
from pathlib import Path
import json
import re
import sqlite3
import tempfile

from bot.journal import DEFAULT_JOURNAL_PATH


class SimulationError(ValueError):
    """Invalid or contradictory simulated evidence; never authorizes an order."""


ZERO = Decimal(0)
ONE = Decimal(1)
_ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_KINDS = frozenset({"BUY_ACK", "BUY_FILL", "BUY_CANCEL", "BUY_REJECT", "BUY_TIMEOUT",
                    "PROTECTION_ACK", "PROTECTION_REJECT", "PROTECTION_FILL",
                    "DISCONNECT", "RESTART", "PRICE"})


def _number(value: object, name: str, *, positive: bool = False) -> Decimal:
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise SimulationError(f"Некорректное число: {name}") from exc
    if not result.is_finite() or (positive and result <= 0):
        raise SimulationError(f"Некорректное число: {name}")
    return result


def _floor(value: Decimal, step: Decimal) -> Decimal:
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def _ceil(value: Decimal, step: Decimal) -> Decimal:
    return (value / step).to_integral_value(rounding=ROUND_CEILING) * step


@dataclass(frozen=True)
class SimulationConfig:
    planned_btc: Decimal = Decimal("0.02")
    entry_usdt: Decimal = Decimal("80000")
    min_size_btc: Decimal = Decimal("0.00001")
    lot_size_btc: Decimal = Decimal("0.00000001")
    tick_usdt: Decimal = Decimal("0.1")
    max_limit_size_btc: Decimal = Decimal("1")  # illustrative, not an OKX account fact
    exit_fee_rate: Decimal = Decimal("0.001")
    exit_slippage_rate: Decimal = Decimal("0.001")
    external_btc: Decimal = Decimal("1")

    def __post_init__(self) -> None:
        for name in ("planned_btc", "entry_usdt", "min_size_btc", "lot_size_btc",
                     "tick_usdt", "max_limit_size_btc", "external_btc"):
            value = getattr(self, name)
            if not isinstance(value, Decimal):
                raise SimulationError(f"Параметр учебного плана должен быть Decimal: {name}")
            _number(value, name, positive=True)
        for name in ("exit_fee_rate", "exit_slippage_rate"):
            value = getattr(self, name)
            if not isinstance(value, Decimal):
                raise SimulationError(f"Параметр учебного плана должен быть Decimal: {name}")
            rate = _number(value, name)
            if not ZERO <= rate < ONE:
                raise SimulationError(f"Некорректная оценка: {name}")
        if (self.planned_btc < self.min_size_btc
                or self.planned_btc > self.max_limit_size_btc
                or self.planned_btc != _floor(self.planned_btc, self.lot_size_btc)
                or self.entry_usdt != _floor(self.entry_usdt, self.tick_usdt)
                or self.exit_fee_rate + self.exit_slippage_rate >= ONE):
            raise SimulationError("Размер, шаг, цена или издержки учебного плана противоречивы")

    @property
    def stop_usdt(self) -> Decimal:
        return _floor(self.entry_usdt * Decimal("0.99"), self.tick_usdt)

    @property
    def take_usdt(self) -> Decimal:
        return _ceil(self.entry_usdt * Decimal("1.02"), self.tick_usdt)

    def fingerprint(self) -> str:
        return json.dumps({key: str(value) for key, value in vars(self).items()}, sort_keys=True)


@dataclass(frozen=True)
class SimEvent:
    event_id: str
    second: int
    kind: str
    quantity_btc: Decimal | None = None
    stop_size_btc: Decimal | None = None
    take_size_btc: Decimal | None = None
    price_usdt: Decimal | None = None
    stop_price_usdt: Decimal | None = None
    take_price_usdt: Decimal | None = None
    fee_btc: Decimal = ZERO  # signed buy fee, negative when charged in BTC
    fee_usdt: Decimal = ZERO  # signed buy/sell fee, negative when charged
    order_id: str | None = None
    stop_id: str | None = None
    take_id: str | None = None
    group_id: str | None = None
    exclusive_oco: bool = False

    def payload(self) -> str:
        if (not isinstance(self.event_id, str) or _ID.fullmatch(self.event_id) is None
                or type(self.second) is not int or not 0 <= self.second <= 86400
                or self.kind not in _KINDS or type(self.exclusive_oco) is not bool):
            raise SimulationError("Некорректный ID, тип или время учебного события")
        values = {}
        for name in ("quantity_btc", "stop_size_btc", "take_size_btc",
                     "price_usdt", "stop_price_usdt", "take_price_usdt",
                     "fee_btc", "fee_usdt"):
            value = getattr(self, name)
            values[name] = None if value is None else str(_number(value, name))
        for name in ("order_id", "stop_id", "take_id", "group_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or _ID.fullmatch(value) is None):
                raise SimulationError(f"Некорректный синтетический ID: {name}")
            values[name] = value
        values["exclusive_oco"] = self.exclusive_oco
        return json.dumps(values, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class SimulationResult:
    buy_state: str
    decision: str
    halted: bool
    reasons: tuple[str, ...]
    bot_owned_btc: Decimal
    bot_remaining_btc: Decimal
    external_btc: Decimal
    simulated_reserved_btc: Decimal
    uncovered_btc: Decimal
    unprotected_seconds: int | None
    btc_seconds_unprotected: Decimal | None
    stop_usdt: Decimal
    take_usdt: Decimal
    price_losses_usdt: tuple[tuple[Decimal, Decimal | None], ...]
    event_count: int
    simulated_realized_pnl_usdt: Decimal | None = None


@dataclass
class _State:
    buy_state: str = "PLANNED"
    gross_bought: Decimal = ZERO
    fee_btc: Decimal = ZERO
    buy_cost_usdt: Decimal = ZERO
    sold_btc: Decimal = ZERO
    sale_proceeds_usdt: Decimal = ZERO
    reserved_btc: Decimal = ZERO
    stop_id: str | None = None
    take_id: str | None = None
    group_id: str | None = None
    triggered_id: str | None = None
    halted: bool = False
    exposure_unknown: bool = False
    reasons: list[str] = field(default_factory=list)
    last_second: int = 0
    unprotected_seconds: int = 0
    btc_seconds_unprotected: Decimal = ZERO
    price_losses: list[tuple[Decimal, Decimal | None]] = field(default_factory=list)
    seen_fill_ids: set[str] = field(default_factory=set)

    @property
    def owned(self) -> Decimal:
        return self.gross_bought + self.fee_btc

    @property
    def remaining(self) -> Decimal:
        return self.owned - self.sold_btc

    @property
    def uncovered(self) -> Decimal:
        return max(ZERO, self.remaining - self.reserved_btc)

    def halt(self, reason: str) -> None:
        self.halted = True
        if reason not in self.reasons:
            self.reasons.append(reason)


def _loss_at(state: _State, price: Decimal, config: SimulationConfig) -> Decimal | None:
    if state.exposure_unknown:
        return None
    conditional_exit_value = (state.remaining * price
                              * (ONE - config.exit_fee_rate - config.exit_slippage_rate))
    return max(ZERO, state.buy_cost_usdt - state.sale_proceeds_usdt - conditional_exit_value)


def _apply(state: _State, event: SimEvent, config: SimulationConfig) -> None:
    if event.second < state.last_second:
        raise SimulationError("Время учебных событий идёт назад")
    delta = event.second - state.last_second
    if state.uncovered > 0:
        state.unprotected_seconds += delta
        state.btc_seconds_unprotected += state.uncovered * delta
    state.last_second = event.second

    kind = event.kind
    if kind == "BUY_ACK":
        if state.buy_state != "PLANNED":
            state.halt("Повторный ACK BUY: новый вход заблокирован")
        else:
            state.buy_state = "ACKNOWLEDGED"
    elif kind == "BUY_FILL":
        qty = _number(event.quantity_btc, "BUY fill", positive=True)
        price = _number(event.price_usdt, "BUY price", positive=True)
        fee_btc = _number(event.fee_btc, "BUY BTC fee")
        fee_usdt = _number(event.fee_usdt, "BUY USDT fee")
        if (state.buy_state not in ("ACKNOWLEDGED", "PARTIAL_ACTIVE", "UNKNOWN")
                or not event.order_id or event.order_id in state.seen_fill_ids
                or fee_btc > 0 or fee_usdt > 0
                or qty + fee_btc <= 0
                or state.gross_bought + qty > config.planned_btc):
            state.halt("BUY fill повторён или противоречив; объём неизвестен")
            state.exposure_unknown = True
        else:
            state.seen_fill_ids.add(event.order_id)
            state.gross_bought += qty
            state.fee_btc += fee_btc
            state.buy_cost_usdt += qty * price - fee_usdt
            state.buy_state = ("FILLED" if state.gross_bought == config.planned_btc
                               else "PARTIAL_ACTIVE")
            if state.uncovered > 0:
                state.halt("Фактически купленный BTC не имеет подтверждённого SL")
    elif kind == "BUY_CANCEL":
        if state.buy_state not in ("ACKNOWLEDGED", "PARTIAL_ACTIVE"):
            state.halt("Отмена остатка BUY противоречит известному состоянию")
            state.exposure_unknown = True
        else:
            state.buy_state = "PARTIAL_CANCELED" if state.gross_bought > 0 else "CANCELED"
            if state.remaining > 0 and state.uncovered > 0:
                state.halt("Отмена остатка BUY не защищает уже купленный BTC")
    elif kind == "BUY_REJECT":
        if state.buy_state not in ("PLANNED", "ACKNOWLEDGED") or state.gross_bought:
            state.halt("Отказ BUY противоречит наблюдаемым исполнениям")
            state.exposure_unknown = True
        else:
            state.buy_state = "REJECTED"
    elif kind == "BUY_TIMEOUT":
        state.buy_state = "UNKNOWN"
        state.exposure_unknown = True
        state.halt("Таймаут BUY: объём и возможный убыток неизвестны; повторять нельзя")
    elif kind == "PROTECTION_ACK":
        qty = _number(event.quantity_btc, "SL/TP size", positive=True)
        stop_size = _number(event.stop_size_btc, "SL size", positive=True)
        take_size = _number(event.take_size_btc, "TP size", positive=True)
        expected = _floor(state.remaining, config.lot_size_btc)
        if (state.remaining <= 0 or state.reserved_btc > 0 or state.stop_id is not None
                or not event.stop_id or not event.take_id or not event.group_id
                or event.stop_id == event.take_id or not event.exclusive_oco
                or qty != expected or qty < config.min_size_btc
                or stop_size != qty or take_size != qty
                or qty > state.remaining
                or event.stop_price_usdt != config.stop_usdt
                or event.take_price_usdt != config.take_usdt):
            state.reserved_btc = ZERO
            state.halt("SL/TP не подтверждены: дубликат, размер, OCO или ID противоречат позиции")
        else:
            state.stop_id, state.take_id, state.group_id = event.stop_id, event.take_id, event.group_id
            state.reserved_btc = qty  # one OCO group reserves BTC once, not twice
            if state.uncovered > 0:
                state.halt("Округлённый SL меньше чистого BTC: остаток без защиты")
    elif kind == "PROTECTION_REJECT":
        state.reserved_btc = ZERO
        state.halt("Биржа отказала в SL/TP: BTC без защиты")
    elif kind == "PROTECTION_FILL":
        qty = _number(event.quantity_btc, "protective fill", positive=True)
        price = _number(event.price_usdt, "protective price", positive=True)
        fee = _number(event.fee_usdt, "protective USDT fee")
        if (not event.order_id or event.order_id not in (state.stop_id, state.take_id)
                or state.reserved_btc <= 0 or qty > state.reserved_btc
                or qty > state.remaining or fee > 0
                or (state.triggered_id is not None and event.order_id != state.triggered_id)):
            state.reserved_btc = ZERO
            state.halt("Защитное исполнение противоречиво: возможна двойная продажа")
            state.exposure_unknown = True
        else:
            state.triggered_id = event.order_id
            state.sold_btc += qty
            state.sale_proceeds_usdt += qty * price + fee
            state.reserved_btc = ZERO  # remainder needs separately proven protection
            if state.remaining > 0:
                state.halt("Частичный защитный fill оставил BTC без подтверждённого SL")
    elif kind == "DISCONNECT":
        state.exposure_unknown = True
        state.reserved_btc = ZERO
        state.halt("Обрыв соединения: статус BUY/SL/TP неизвестен")
    elif kind == "RESTART":
        # Replaying durable events cannot clear a prior halt or uncertainty.
        pass
    elif kind == "PRICE":
        price = _number(event.price_usdt, "scenario price", positive=True)
        state.price_losses.append((price, _loss_at(state, price, config)))
    else:
        raise SimulationError("Неизвестное учебное событие")


class SimulationLedger:
    """Append/replay a sandbox SQLite file; all connections close per operation."""

    def __init__(self, path: Path, config: SimulationConfig = SimulationConfig()) -> None:
        self.path = Path(path).resolve()
        self.config = config
        temp_root = Path(tempfile.gettempdir()).resolve()
        if (not self.path.is_relative_to(temp_root)
                or self.path == DEFAULT_JOURNAL_PATH.resolve()
                or self.path.suffix != ".sqlite3"):
            raise SimulationError("Симуляция допускает SQLite только во временной директории")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with closing(sqlite3.connect(self.path, timeout=3)) as db:
                names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if names and names != {"simulation_meta", "simulation_events"}:
                    raise SimulationError("Чужую SQLite-базу симулятор не изменяет")
                if not names:
                    db.execute("CREATE TABLE simulation_meta (version INTEGER NOT NULL, config TEXT NOT NULL)")
                    db.execute("""CREATE TABLE simulation_events (
                        sequence INTEGER PRIMARY KEY, event_id TEXT NOT NULL UNIQUE,
                        second INTEGER NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL)""")
                    db.execute("INSERT INTO simulation_meta VALUES (1,?)", (config.fingerprint(),))
                    db.commit()
                elif db.execute("SELECT version, config FROM simulation_meta").fetchall() != [(1, config.fingerprint())]:
                    raise SimulationError("Параметры или версия временной симуляции изменились")
        except sqlite3.Error as exc:
            raise SimulationError("Временная SQLite симуляции недоступна") from exc

    def append(self, event: SimEvent) -> SimulationResult:
        payload = event.payload()
        try:
            with closing(sqlite3.connect(self.path, timeout=3)) as db:
                try:
                    db.execute("BEGIN IMMEDIATE")
                    existing = db.execute("SELECT second,kind,payload FROM simulation_events WHERE event_id=?",
                                          (event.event_id,)).fetchone()
                    if existing is None:
                        db.execute("INSERT INTO simulation_events (event_id,second,kind,payload) VALUES (?,?,?,?)",
                                   (event.event_id, event.second, event.kind, payload))
                    elif existing != (event.second, event.kind, payload):
                        raise SimulationError("Повторный ID события с другим содержимым")
                    db.commit()
                except BaseException:
                    db.rollback()
                    raise
        except sqlite3.Error as exc:
            raise SimulationError("Событие не записано во временную SQLite") from exc
        return self.replay()

    def replay(self) -> SimulationResult:
        try:
            with closing(sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=3)) as db:
                if db.execute("SELECT version,config FROM simulation_meta").fetchall() != [(1, self.config.fingerprint())]:
                    raise SimulationError("Параметры временной симуляции не совпадают; BLOCK")
                rows = db.execute("SELECT event_id,second,kind,payload FROM simulation_events ORDER BY sequence").fetchall()
        except sqlite3.Error as exc:
            raise SimulationError("Временную SQLite нельзя восстановить") from exc
        state = _State()
        for event_id, second, kind, payload in rows:
            try:
                values = json.loads(payload)
                event = SimEvent(event_id, second, kind,
                    quantity_btc=(None if values["quantity_btc"] is None else Decimal(values["quantity_btc"])),
                    stop_size_btc=(None if values["stop_size_btc"] is None else Decimal(values["stop_size_btc"])),
                    take_size_btc=(None if values["take_size_btc"] is None else Decimal(values["take_size_btc"])),
                    price_usdt=(None if values["price_usdt"] is None else Decimal(values["price_usdt"])),
                    stop_price_usdt=(None if values["stop_price_usdt"] is None else Decimal(values["stop_price_usdt"])),
                    take_price_usdt=(None if values["take_price_usdt"] is None else Decimal(values["take_price_usdt"])),
                    fee_btc=Decimal(values["fee_btc"]), fee_usdt=Decimal(values["fee_usdt"]),
                    order_id=values["order_id"], stop_id=values["stop_id"],
                    take_id=values["take_id"], group_id=values["group_id"],
                    exclusive_oco=values["exclusive_oco"])
                if event.payload() != payload:
                    raise SimulationError("Учебное событие повреждено")
                _apply(state, event, self.config)
            except (KeyError, TypeError, ValueError, InvalidOperation, ArithmeticError) as exc:
                raise SimulationError("Временный журнал симуляции повреждён; BLOCK") from exc
        # Only a fully closed, internally consistent synthetic position has a
        # realized result. Partial/ambiguous exits cannot invent a final PnL.
        realized = (state.sale_proceeds_usdt - state.buy_cost_usdt
                    if state.gross_bought > 0 and state.remaining == 0
                    and not state.exposure_unknown else None)
        return SimulationResult(state.buy_state, "BLOCK", state.halted,
            tuple(state.reasons), state.owned, state.remaining, self.config.external_btc,
            state.reserved_btc, state.uncovered,
            None if state.exposure_unknown else state.unprotected_seconds,
            None if state.exposure_unknown else state.btc_seconds_unprotected,
            self.config.stop_usdt, self.config.take_usdt,
            tuple(state.price_losses), len(rows), realized)


def parse_price_path(raw: str) -> tuple[Decimal, ...]:
    if not isinstance(raw, str):
        raise SimulationError("Укажите цены через запятую")
    parts = raw.split(",")
    if not 1 <= len(parts) <= 20:
        raise SimulationError("Укажите от 1 до 20 цен через запятую")
    return tuple(_number(part.strip(), "цена сценария", positive=True) for part in parts)


def run_sample_scenarios(prices: tuple[Decimal, ...]) -> tuple[tuple[str, SimulationResult], ...]:
    """Run fixed hypothetical lifecycle cases in disposable SQLite databases."""
    if not isinstance(prices, tuple) or not 1 <= len(prices) <= 20:
        raise SimulationError("Ценовой путь неизвестен")
    for price in prices:
        _number(price, "цена сценария", positive=True)
    c = SimulationConfig()
    full = c.planned_btc
    half = full / 2
    fee_full = -full * Decimal("0.001")
    fee_half = -half * Decimal("0.001")
    net_full = full + fee_full
    net_half = half + fee_half

    def e(tag: str, second: int, kind: str, **kwargs) -> SimEvent:
        return SimEvent(tag, second, kind, **kwargs)

    scenarios: tuple[tuple[str, tuple[SimEvent, ...]], ...] = (
        ("Полный BUY, мгновенная учебная OCO-защита", (
            e("ack", 0, "BUY_ACK"),
            e("fill", 1, "BUY_FILL", quantity_btc=full, price_usdt=c.entry_usdt,
              fee_btc=fee_full, order_id="fill-1"),
            e("oco", 1, "PROTECTION_ACK", quantity_btc=net_full,
              stop_size_btc=net_full, take_size_btc=net_full,
              stop_price_usdt=c.stop_usdt, take_price_usdt=c.take_usdt,
              stop_id="stop-1", take_id="take-1", group_id="oco-1", exclusive_oco=True))),
        ("Полный учебный цикл BUY, защита, закрытие TP и PnL", (
            e("ack", 0, "BUY_ACK"),
            e("fill", 1, "BUY_FILL", quantity_btc=full, price_usdt=c.entry_usdt,
              fee_btc=fee_full, order_id="fill-1"),
            e("oco", 1, "PROTECTION_ACK", quantity_btc=net_full,
              stop_size_btc=net_full, take_size_btc=net_full,
              stop_price_usdt=c.stop_usdt, take_price_usdt=c.take_usdt,
              stop_id="stop-1", take_id="take-1", group_id="oco-1", exclusive_oco=True),
            e("exit", 30, "PROTECTION_FILL", quantity_btc=net_full,
              price_usdt=c.take_usdt,
              fee_usdt=-(net_full * c.take_usdt * c.exit_fee_rate), order_id="take-1"))),
        ("Частичный BUY: SL задержан, затем новая часть без SL", (
            e("ack", 0, "BUY_ACK"),
            e("fill-a", 1, "BUY_FILL", quantity_btc=half, price_usdt=c.entry_usdt,
              fee_btc=fee_half, order_id="fill-1"),
            e("oco", 31, "PROTECTION_ACK", quantity_btc=net_half,
              stop_size_btc=net_half, take_size_btc=net_half,
              stop_price_usdt=c.stop_usdt, take_price_usdt=c.take_usdt,
              stop_id="stop-1", take_id="take-1", group_id="oco-1", exclusive_oco=True),
            e("fill-b", 41, "BUY_FILL", quantity_btc=half, price_usdt=c.entry_usdt,
              fee_btc=fee_half, order_id="fill-2"))),
        ("Частичный BUY, отмена остатка и поздний SL", (
            e("ack", 0, "BUY_ACK"),
            e("fill", 1, "BUY_FILL", quantity_btc=half, price_usdt=c.entry_usdt,
              fee_btc=fee_half, order_id="fill-1"),
            e("cancel", 20, "BUY_CANCEL"),
            e("oco", 40, "PROTECTION_ACK", quantity_btc=net_half,
              stop_size_btc=net_half, take_size_btc=net_half,
              stop_price_usdt=c.stop_usdt, take_price_usdt=c.take_usdt,
              stop_id="stop-1", take_id="take-1", group_id="oco-1", exclusive_oco=True))),
        ("Отказ BUY без исполнения", (
            e("ack", 0, "BUY_ACK"), e("reject", 1, "BUY_REJECT"))),
        ("Таймаут BUY: биржевое состояние неизвестно", (
            e("ack", 0, "BUY_ACK"), e("timeout", 1, "BUY_TIMEOUT"))),
        ("Отказ защитного SL/TP после полного BUY", (
            e("ack", 0, "BUY_ACK"),
            e("fill", 1, "BUY_FILL", quantity_btc=full, price_usdt=c.entry_usdt,
              fee_btc=fee_full, order_id="fill-1"),
            e("reject-protection", 15, "PROTECTION_REJECT"))),
        ("Частичное защитное исполнение оставило BTC без SL", (
            e("ack", 0, "BUY_ACK"),
            e("fill", 1, "BUY_FILL", quantity_btc=full, price_usdt=c.entry_usdt,
              fee_btc=fee_full, order_id="fill-1"),
            e("oco", 1, "PROTECTION_ACK", quantity_btc=net_full,
              stop_size_btc=net_full, take_size_btc=net_full,
              stop_price_usdt=c.stop_usdt, take_price_usdt=c.take_usdt,
              stop_id="stop-1", take_id="take-1", group_id="oco-1", exclusive_oco=True),
            e("sl-partial", 45, "PROTECTION_FILL", quantity_btc=Decimal("0.005"),
              price_usdt=c.stop_usdt, fee_usdt=Decimal("-0.396"), order_id="stop-1"))),
        ("Дублирующий SL/TP: двойной резерв запрещён", (
            e("ack", 0, "BUY_ACK"),
            e("fill", 1, "BUY_FILL", quantity_btc=full, price_usdt=c.entry_usdt,
              fee_btc=fee_full, order_id="fill-1"),
            e("oco", 1, "PROTECTION_ACK", quantity_btc=net_full,
              stop_size_btc=net_full, take_size_btc=net_full,
              stop_price_usdt=c.stop_usdt, take_price_usdt=c.take_usdt,
              stop_id="stop-1", take_id="take-1", group_id="oco-1", exclusive_oco=True),
            e("duplicate", 20, "PROTECTION_ACK", quantity_btc=net_full,
              stop_size_btc=net_full, take_size_btc=net_full,
              stop_price_usdt=c.stop_usdt, take_price_usdt=c.take_usdt,
              stop_id="stop-2", take_id="take-2", group_id="oco-2", exclusive_oco=True))),
        ("Обрыв, перезапуск и неизвестный статус защиты", (
            e("ack", 0, "BUY_ACK"),
            e("fill", 1, "BUY_FILL", quantity_btc=full, price_usdt=c.entry_usdt,
              fee_btc=fee_full, order_id="fill-1"),
            e("disconnect", 10, "DISCONNECT"), e("restart", 20, "RESTART"))),
    )
    results: list[tuple[str, SimulationResult]] = []
    for name, events in scenarios:
        with tempfile.TemporaryDirectory(prefix="okx-protection-sim-") as directory:
            path = Path(directory) / "simulation.sqlite3"
            ledger = SimulationLedger(path, c)
            for event in events:
                ledger.append(event)
                if event.kind == "RESTART":
                    ledger = SimulationLedger(path, c)
            for index, price in enumerate(prices):
                ledger.append(e(f"price-{index}", 60 + 30 * index, "PRICE", price_usdt=price))
            results.append((name, ledger.replay()))
    return tuple(results)
