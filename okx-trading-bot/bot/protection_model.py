"""Offline evidence model for Demo Spot BUY protection; never authorizes trading.

An exchange response is only evidence after its identity, net BTC size, and
mutual-exclusion relationship have been independently reconciled. This module
has no HTTP transport or order payloads. Every recorded observation HALTs the
local executor before it is appended to SQLite.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum
import json
import re
import sqlite3
from uuid import uuid4

from bot.execution_engine import ExecutionEngine, ExecutionError, ExecutionIntent, _utc


class BuyState(StrEnum):
    PARTIAL_ACTIVE = "PARTIAL_ACTIVE"
    PARTIAL_CANCELED = "PARTIAL_CANCELED"
    FILLED = "FILLED"
    TIMEOUT = "TIMEOUT"


_EXCHANGE_ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")


@dataclass(frozen=True)
class AlgoLeg:
    algo_id: str
    kind: str  # STOP or TAKE
    parent_order_id: str
    parent_client_order_id: str
    group_id: str
    size_btc: Decimal
    trigger_price: Decimal
    state: str  # live only; a pending, canceled, or triggered leg is not coverage
    side: str = "SELL"
    instrument_id: str = "BTC-USDT"
    mode: str = "DEMO"


@dataclass(frozen=True)
class BuyObservation:
    client_order_id: str
    exchange_order_id: str | None
    state: BuyState
    filled_btc: Decimal
    checked_at_utc: datetime
    stop: AlgoLeg | None = None
    take: AlgoLeg | None = None
    exclusive_group_confirmed: bool = False


@dataclass(frozen=True)
class ProtectionAssessment:
    client_order_id: str
    buy_state: BuyState
    owned_btc: Decimal
    confirmed_covered_btc: Decimal
    uncovered_btc: Decimal
    reasons: tuple[str, ...]
    halted: bool = True
    stop_algo_id: str | None = None
    take_algo_id: str | None = None


def _positive(value: object) -> bool:
    return isinstance(value, Decimal) and value.is_finite() and value > 0


def assess_protection(intent: ExecutionIntent, observation: BuyObservation,
                      *, now_utc: datetime) -> ProtectionAssessment:
    """Conservatively assess *owned* net BTC; external wallet BTC is excluded."""
    if not isinstance(intent, ExecutionIntent) or not isinstance(observation, BuyObservation):
        raise ExecutionError("Наблюдение защиты отсутствует")
    if not isinstance(observation.state, BuyState):
        raise ExecutionError("Состояние BUY неизвестно")
    reasons: list[str] = []
    owned = intent.owned_btc
    if not isinstance(owned, Decimal) or not owned.is_finite() or owned < 0:
        raise ExecutionError("Размер собственной позиции противоречив")
    reliable = True
    if (not isinstance(now_utc, datetime) or now_utc.tzinfo is None
            or now_utc.utcoffset() != timedelta(0)
            or not isinstance(observation.checked_at_utc, datetime)
            or observation.checked_at_utc.tzinfo is None
            or observation.checked_at_utc.utcoffset() != timedelta(0)
            or observation.checked_at_utc > now_utc
            or now_utc - observation.checked_at_utc > timedelta(seconds=60)):
        reasons.append("Наблюдение Demo устарело или время неизвестно")
        reliable = False
    if (observation.client_order_id != intent.client_order_id
            or not observation.exchange_order_id
            or observation.exchange_order_id != intent.exchange_order_id):
        reasons.append("BUY не связан с сохранённым clientOrderId и ordId")
        reliable = False
    if (not isinstance(observation.filled_btc, Decimal)
            or not observation.filled_btc.is_finite()
            or observation.filled_btc != intent.filled_quantity):
        reasons.append("Фактический объём BUY и журнал исполнений расходятся")
        reliable = False
    if observation.state == BuyState.TIMEOUT:
        reasons.append("Таймаут: состояние BUY и последующие исполнения неизвестны")
        reliable = False
    elif observation.state == BuyState.PARTIAL_ACTIVE:
        if not 0 < intent.filled_quantity < intent.quantity:
            reasons.append("Активный частичный BUY противоречит объёму исполнения")
            reliable = False
        reasons.append("Активный остаток BUY может исполниться без подтверждённого SL")
    elif observation.state == BuyState.PARTIAL_CANCELED:
        if not 0 < intent.filled_quantity < intent.quantity:
            reasons.append("Отменённый частичный BUY противоречит объёму исполнения")
            reliable = False
        reasons.append("Отмена остатка BUY требует сверки биржевого статуса")
    elif observation.state == BuyState.FILLED:
        if intent.filled_quantity != intent.quantity:
            reasons.append("Полное исполнение BUY не подтверждено журналом")
            reliable = False
    else:
        reasons.append("Состояние BUY неизвестно")
        reliable = False

    if owned > 0 and (owned < intent.min_size or owned % intent.lot_size != 0):
        reasons.append("Чистый BTC после комиссии не соответствует minSz/lotSz")
        reliable = False

    covered = Decimal(0)
    if owned > 0:
        stop, take = observation.stop, observation.take
        if stop is None or take is None:
            reasons.append("Купленный ботом BTC не имеет обеих подтверждённых защитных ног")
        else:
            legs_valid = True
            for leg, kind, price in ((stop, "STOP", intent.stop_price),
                                     (take, "TAKE", intent.take_profit_price)):
                if (not isinstance(leg, AlgoLeg)
                        or not isinstance(leg.algo_id, str) or not _EXCHANGE_ID.fullmatch(leg.algo_id)
                        or leg.kind != kind or leg.parent_order_id != intent.exchange_order_id
                        or leg.parent_client_order_id != intent.client_order_id
                        or leg.mode != "DEMO" or leg.instrument_id != "BTC-USDT"
                        or leg.side != "SELL" or leg.state != "live"
                        or not _positive(leg.size_btc) or leg.size_btc != owned
                        or not _positive(leg.trigger_price) or leg.trigger_price != price):
                    legs_valid = False
            if not legs_valid:
                reasons.append("Algo ID, привязка, статус, цена или фактический размер SL/TP не подтверждены")
            if (stop.algo_id == take.algo_id or not stop.group_id
                    or stop.group_id != take.group_id
                    or observation.exclusive_group_confirmed is not True):
                reasons.append("Взаимоисключение SL/TP не доказано: возможна двойная продажа")
                legs_valid = False
            if legs_valid and reliable:
                covered = owned
    # A complete offline match is still insufficient evidence of atomic
    # protection on every partial fill or regional Demo feature availability.
    reasons.append("Атомарность attached TP/SL для BTC-USDT SPOT Demo и региона не подтверждена")
    if covered < owned:
        reasons.append("Есть BTC бота без подтверждённого SL; новые сделки запрещены")
    return ProtectionAssessment(intent.client_order_id, observation.state, owned,
                                covered, owned - covered, tuple(reasons), True,
                                observation.stop.algo_id if isinstance(observation.stop, AlgoLeg) else None,
                                observation.take.algo_id if isinstance(observation.take, AlgoLeg) else None)


class ProtectionJournal:
    """Append-only local observations. It never clears HALTED or reserves risk."""

    def __init__(self, engine: ExecutionEngine) -> None:
        self.engine = engine

    def record(self, observation: BuyObservation) -> ProtectionAssessment:
        # Halt before any potentially failing validation or SQLite append.
        self.engine.emergency_halt("PROTECTION_UNVERIFIED")
        matches = [i for i in self.engine.diagnose().intents
                   if i.client_order_id == observation.client_order_id]
        if len(matches) != 1:
            raise ExecutionError("Намерение защиты не найдено; HALTED сохранён")
        now = self.engine.clock()
        result = assess_protection(matches[0], observation, now_utc=now)
        try:
            with self.engine._transaction() as db:
                self.engine._schema(db)
                db.execute("""CREATE TABLE IF NOT EXISTS protection_observations (
                    id TEXT PRIMARY KEY, client_order_id TEXT NOT NULL,
                    checked_at_utc TEXT NOT NULL, buy_state TEXT NOT NULL,
                    owned_btc TEXT NOT NULL, covered_btc TEXT NOT NULL,
                    uncovered_btc TEXT NOT NULL, stop_algo_id TEXT, take_algo_id TEXT,
                    reasons_json TEXT NOT NULL,
                    FOREIGN KEY(client_order_id) REFERENCES execution_intents(client_order_id))""")
                db.execute("""INSERT INTO protection_observations VALUES (?,?,?,?,?,?,?,?,?,?)""", (
                    uuid4().hex, result.client_order_id,
                    _utc(observation.checked_at_utc, "Время наблюдения"),
                    result.buy_state.value, str(result.owned_btc),
                    str(result.confirmed_covered_btc), str(result.uncovered_btc),
                    observation.stop.algo_id if isinstance(observation.stop, AlgoLeg) else None,
                    observation.take.algo_id if isinstance(observation.take, AlgoLeg) else None,
                    json.dumps(result.reasons, ensure_ascii=False),
                ))
        except (sqlite3.Error, OSError) as exc:
            raise ExecutionError("Доказательство защиты не записано; HALTED сохранён") from exc
        return result

    def latest(self, client_order_id: str) -> ProtectionAssessment | None:
        if not self.engine.path.is_file():
            return None
        try:
            uri = self.engine.path.resolve().as_uri() + "?mode=ro"
            with closing(sqlite3.connect(uri, uri=True, timeout=3)) as db:
                db.row_factory = sqlite3.Row
                db.execute("PRAGMA query_only=ON")
                row = db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='protection_observations'").fetchone()
                if row is None:
                    return None
                row = db.execute("""SELECT * FROM protection_observations
                    WHERE client_order_id=? ORDER BY rowid DESC LIMIT 1""", (client_order_id,)).fetchone()
                if row is None:
                    return None
                owned, covered, uncovered = (Decimal(row[name]) for name in
                                             ("owned_btc", "covered_btc", "uncovered_btc"))
                if owned < 0 or covered < 0 or uncovered < 0 or covered + uncovered != owned:
                    raise ExecutionError("Доказательство защиты противоречиво")
                reasons = json.loads(row["reasons_json"])
                if not isinstance(reasons, list) or not all(isinstance(x, str) for x in reasons):
                    raise ExecutionError("Причины защиты повреждены")
                return ProtectionAssessment(row["client_order_id"], BuyState(row["buy_state"]),
                                            owned, covered, uncovered, tuple(reasons), True,
                                            row["stop_algo_id"], row["take_algo_id"])
        except (sqlite3.Error, OSError, ValueError, TypeError) as exc:
            raise ExecutionError("Доказательство защиты недоступно; исполнение заблокировано") from exc
