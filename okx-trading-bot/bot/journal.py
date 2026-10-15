"""Durable, local SQLite journal for non-executing Demo analysis cycles."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import StrEnum
import json
from pathlib import Path
import sqlite3
from uuid import uuid4


DEFAULT_JOURNAL_PATH = Path(__file__).resolve().parents[1] / "data" / "dry_run_journal.sqlite3"
RISK_KEYS = frozenset({
    "risk_per_trade", "stop_loss_pct", "take_profit_pct", "daily_loss_limit_pct",
    "max_open_positions", "max_new_entries_per_day", "equity_usdt", "day_start_capital_usdt",
    "available_usdt", "realized_pnl_today_usdt", "unrealized_pnl_usdt", "open_positions",
    "pending_orders", "entries_today", "entry_fee_rate", "exit_fee_rate",
    "entry_slippage_rate", "stop_slippage_rate", "instrument_min_size", "instrument_lot_size",
    "instrument_tick_size", "instrument_max_size",
})


class JournalError(Exception):
    """A cycle could not be recorded safely."""


class CycleStatus(StrEnum):
    ALLOW = "ALLOW"
    BLOCK = "BLOCK"
    HOLD = "HOLD"


@dataclass(frozen=True)
class CycleDraft:
    created_at_utc: datetime
    candle_closed_at: datetime | None
    signal: str | None
    status: CycleStatus
    reason: str
    ema9: Decimal | None = None
    ema21: Decimal | None = None
    virtual_btc_available: Decimal | None = None
    virtual_usdt_available: Decimal | None = None
    risk_parameters: dict[str, str] | None = None
    quantity: Decimal | None = None
    entry_price: Decimal | None = None
    stop_price: Decimal | None = None
    take_profit_price: Decimal | None = None
    estimated_risk_usdt: Decimal | None = None


@dataclass(frozen=True)
class CycleRecord:
    id: str
    draft: CycleDraft
    reference_id: str | None = None
    executed: bool = False


def _utc(value: datetime | None, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise JournalError(f"{name}: требуется время UTC")
    return value.isoformat().replace("+00:00", "Z")


def _number(value: Decimal | None, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, Decimal) or not value.is_finite():
        raise JournalError(f"{name}: некорректное число")
    return str(value)


class DecisionJournal:
    """Append every cycle; duplicate candles refer to the original decision."""

    def __init__(self, path: Path = DEFAULT_JOURNAL_PATH) -> None:
        self.path = Path(path)

    def append(self, draft: CycleDraft) -> CycleRecord:
        if not isinstance(draft, CycleDraft) or not isinstance(draft.status, CycleStatus):
            raise JournalError("Некорректное решение цикла")
        created = _utc(draft.created_at_utc, "Время решения")
        if created is None:
            raise JournalError("Время решения отсутствует")
        candle = _utc(draft.candle_closed_at, "Время свечи")
        if draft.signal not in (None, "BUY", "SELL", "HOLD") or not isinstance(draft.reason, str) or not draft.reason:
            raise JournalError("Некорректные сигнал или причина")
        risk = draft.risk_parameters or {}
        if not isinstance(risk, dict) or set(risk) - RISK_KEYS:
            raise JournalError("Недопустимые параметры риска в журнале")
        for value in risk.values():
            try:
                number = Decimal(value)
            except (TypeError, ValueError, InvalidOperation) as exc:
                raise JournalError("Некорректный параметр риска в журнале") from exc
            if not number.is_finite():
                raise JournalError("Некорректный параметр риска в журнале")
        values = {
            "ema9": _number(draft.ema9, "EMA9"),
            "ema21": _number(draft.ema21, "EMA21"),
            "virtual_btc_available": _number(draft.virtual_btc_available, "BTC"),
            "virtual_usdt_available": _number(draft.virtual_usdt_available, "USDT"),
            "quantity": _number(draft.quantity, "Размер"),
            "entry_price": _number(draft.entry_price, "Цена"),
            "stop_price": _number(draft.stop_price, "Стоп"),
            "take_profit_price": _number(draft.take_profit_price, "Тейк"),
            "estimated_risk_usdt": _number(draft.estimated_risk_usdt, "Риск"),
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(self.path, timeout=3)) as connection:
                connection.execute("PRAGMA synchronous=FULL")
                connection.execute("""
                    CREATE TABLE IF NOT EXISTS decisions (
                        id TEXT PRIMARY KEY, created_at_utc TEXT NOT NULL,
                        candle_closed_at TEXT, signal TEXT,
                        status TEXT NOT NULL CHECK (status IN ('ALLOW','BLOCK','HOLD')),
                        reason TEXT NOT NULL, ema9 TEXT, ema21 TEXT,
                        virtual_btc_available TEXT, virtual_usdt_available TEXT,
                        risk_parameters TEXT NOT NULL, quantity TEXT, entry_price TEXT,
                        stop_price TEXT, take_profit_price TEXT, estimated_risk_usdt TEXT,
                        reference_id TEXT, executed INTEGER NOT NULL DEFAULT 0 CHECK (executed = 0)
                    )
                """)
                connection.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS one_original_per_candle
                    ON decisions(candle_closed_at)
                    WHERE candle_closed_at IS NOT NULL AND reference_id IS NULL
                """)
                connection.execute("BEGIN IMMEDIATE")
                original = None
                if candle is not None:
                    original = connection.execute(
                        "SELECT id FROM decisions WHERE candle_closed_at=? AND reference_id IS NULL", (candle,)
                    ).fetchone()
                reference_id = original[0] if original else None
                if reference_id is not None:
                    draft = replace(
                        draft, status=CycleStatus.BLOCK,
                        reason="Повторная проверка той же закрытой свечи; новое намерение не создано",
                        quantity=None, entry_price=None, stop_price=None, take_profit_price=None,
                        estimated_risk_usdt=None,
                    )
                    for key in ("quantity", "entry_price", "stop_price", "take_profit_price", "estimated_risk_usdt"):
                        values[key] = None
                record = CycleRecord(str(uuid4()), draft, reference_id)
                connection.execute("""
                    INSERT INTO decisions (
                        id, created_at_utc, candle_closed_at, signal, status, reason,
                        ema9, ema21, virtual_btc_available, virtual_usdt_available,
                        risk_parameters, quantity, entry_price, stop_price, take_profit_price,
                        estimated_risk_usdt, reference_id, executed
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)
                """, (
                    record.id, created, candle, draft.signal, draft.status.value, draft.reason,
                    values["ema9"], values["ema21"], values["virtual_btc_available"],
                    values["virtual_usdt_available"], json.dumps(risk, sort_keys=True),
                    values["quantity"], values["entry_price"], values["stop_price"],
                    values["take_profit_price"], values["estimated_risk_usdt"], reference_id,
                ))
                connection.commit()
                return record
        except (sqlite3.Error, OSError) as exc:
            raise JournalError("Не удалось безопасно сохранить решение в локальном журнале") from exc
