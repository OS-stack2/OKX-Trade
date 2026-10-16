"""Offline, read-only explanation of the Demo execution gate.

This module has no exchange client and never creates the trading journal.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
import sqlite3

from bot.execution_engine import ExecutionEngine, ExecutionError, ExecutionState, _parse_utc
from bot.journal import JournalError


# Only codes emitted by the local execution/recovery modules may reach the console.
# Database text is otherwise untrusted and could contain a secret.
_REASON_CODES = frozenset({
    "CLEAR", "MANUAL_RECONCILED", "MANUAL_HALT", "RECOVERY_CHECK",
    "RECOVERY_MISMATCH", "RECOVERY_NETWORK_ERROR", "RECOVERY_UNPROTECTED",
    "PROTECTION_UNVERIFIED", "CONFLICTING_TRADE_ID", "INVALID_FILL_TOTAL",
    "UNPROTECTABLE_PARTIAL_FILL", "FILLED_BUY_UNPROTECTED",
    "ORDER_STATUS_UNCERTAIN", "PARTIAL_BUY_ACTIVE", "PARTIAL_BUY_CANCELED",
    "UNPROTECTED_DEMO_ORDER", "PLANNED", "RISK_REVALIDATED",
    "PRE_SEND_BOUNDARY", "NETWORK_TIMEOUT", "AMBIGUOUS_RESPONSE",
    "EXCHANGE_ACK", "FILL_OBSERVED", "FILL_OBSERVED_HALTED",
    "LOCAL_PLAN_REJECTED", "RECONCILED_NO_FILL", "MANUAL_RECONCILED_NO_FILL",
})

MISSING_PROTECTION_EVIDENCE = (
    "Статус BUY и полная история каждого fill/комиссии по clOrdId/ordId; чистый BTC бота после комиссии.",
    "Подтверждение SL для каждой частично исполненной порции BTC без временного незащищённого интервала.",
    "Биржевые Algo ID, действующий статус, размер, триггер и связь SL/TP с конкретным BUY.",
    "Взаимоисключение SL и TP и запрет двойной продажи или расходования внешнего BTC.",
    "Поддержка и атомарность attached TP/SL именно для BTC-USDT SPOT Demo в регионе аккаунта.",
)


def _safe_code(value: object) -> str:
    return value if type(value) is str and value in _REASON_CODES else "НЕИЗВЕСТНЫЙ_КОД"


@dataclass(frozen=True)
class IntentSummary:
    client_order_id: str
    state: ExecutionState
    reason_code: str
    candle_closed_at: datetime
    created_at_utc: datetime
    updated_at_utc: datetime
    quantity: Decimal
    filled_quantity: Decimal
    owned_btc: Decimal
    exchange_order_id_known: bool
    flags: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionInspection:
    sqlite_status: str  # CLEAR, BLOCKED or UNKNOWN. Independent of protection.
    sqlite_reason: str
    halted: bool | None
    halt_reason_code: str | None
    halt_updated_at_utc: datetime | None
    intent_count: int
    latest_intent: IntentSummary | None
    protection_status: str = "UNVERIFIED"
    missing_protection_evidence: tuple[str, ...] = MISSING_PROTECTION_EVIDENCE


def inspect_execution(engine: ExecutionEngine) -> ExecutionInspection:
    """Read one consistent SQLite snapshot; never infer OKX protection from local rows."""
    path = engine.path
    if path.is_symlink() or not path.is_file():
        return ExecutionInspection("UNKNOWN", "Торговая SQLite отсутствует или путь недопустим", None, None, None, 0, None)
    try:
        engine.journal.audit_read_only()
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=3)) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA query_only=ON")
            db.execute("BEGIN")
            with closing(db.execute("SELECT name FROM sqlite_master WHERE type='table'")) as cursor:
                names = {row[0] for row in cursor}
            required = {"execution_control", "execution_intents", "execution_fills", "execution_events"}
            present = names & required
            if not present:
                return ExecutionInspection("CLEAR", "Нет локальных намерений", False, None, None, 0, None)
            if present != required:
                raise ExecutionError("Неполная схема исполнения")
            with closing(db.execute("SELECT id, halted, reason_code, updated_at_utc FROM execution_control")) as cursor:
                controls = cursor.fetchall()
            if (len(controls) != 1 or controls[0]["id"] != 1
                    or type(controls[0]["halted"]) is not int or controls[0]["halted"] not in (0, 1)):
                raise ExecutionError("Аварийное состояние неизвестно")
            control = controls[0]
            halted = bool(control["halted"])
            control_time = _parse_utc(control["updated_at_utc"])
            control_reason = _safe_code(control["reason_code"])
            if (control_reason == "НЕИЗВЕСТНЫЙ_КОД"
                    or (halted and control_reason in ("CLEAR", "MANUAL_RECONCILED"))
                    or (not halted and control_reason not in ("CLEAR", "MANUAL_RECONCILED"))):
                raise ExecutionError("Неизвестная причина аварийного состояния")
            with closing(db.execute("SELECT * FROM execution_intents ORDER BY rowid")) as cursor:
                intents = tuple(ExecutionEngine._intent(row) for row in cursor)
            for intent in intents:
                if intent.updated_at_utc < intent.created_at_utc or _safe_code(intent.reason_code) == "НЕИЗВЕСТНЫЙ_КОД":
                    raise ExecutionError("Намерение содержит противоречивые сведения")
            latest = None
            if intents:
                intent = intents[-1]
                flags = []
                if intent.state in (ExecutionState.SUBMITTING, ExecutionState.UNKNOWN):
                    flags.append("Результат отправки или статус BUY не подтверждён")
                if intent.state in (ExecutionState.ACKNOWLEDGED, ExecutionState.PARTIALLY_FILLED):
                    flags.append("BUY может оставаться активным; остаток может исполниться")
                if intent.filled_quantity > 0:
                    flags.append("Есть локально учтённый fill; защита BTC не подтверждена этим журналом")
                if intent.state != ExecutionState.REJECTED:
                    flags.append("Новое намерение заблокировано")
                else:
                    flags.append("Прежнее намерение блокирует --first-demo-trade до EMA/RISK")
                latest = IntentSummary(
                    intent.client_order_id, intent.state, _safe_code(intent.reason_code),
                    intent.candle_closed_at, intent.created_at_utc, intent.updated_at_utc,
                    intent.quantity, intent.filled_quantity, intent.owned_btc,
                    intent.exchange_order_id is not None, tuple(flags),
                )
            active = sum(intent.state != ExecutionState.REJECTED for intent in intents)
            # The first-trade preflight intentionally blocks on *any* previous
            # intent, including a locally rejected one.
            blocked = halted or bool(intents)
            reason = ("Сохранён HALTED" if halted else
                      f"Незавершённых или подтверждённых намерений: {active}" if active else
                      "Сохранено прежнее намерение; первая Demo-проверка останавливается до EMA/RISK" if intents else
                      "Локальных незавершённых намерений нет")
            return ExecutionInspection("BLOCKED" if blocked else "CLEAR", reason, halted,
                                       control_reason if halted else None, control_time if halted else None,
                                       len(intents), latest)
    except (sqlite3.Error, OSError, JournalError, ExecutionError, TypeError, ValueError, KeyError):
        return ExecutionInspection("UNKNOWN", "Торговую SQLite нельзя безопасно проверить: отсутствуют или противоречивы данные", None, None, None, 0, None)
