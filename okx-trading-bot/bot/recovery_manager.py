"""Fail-closed reconciliation of local BUY evidence with OKX Demo GET responses.

This module cannot place, amend, cancel or sell an order. It persists HALTED
before network access whenever an execution intent exists. A GET Algo response
does not prove atomic SL coverage or parentage, so recovery never unlocks.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import sqlite3
from typing import Callable

from bot.demo_client import CurrencyBalance, DemoAlgoOrder, DemoClient, DemoFill, DemoOrder
from bot.execution_engine import ExecutionEngine, ExecutionError, ExecutionIntent
from bot.protection_model import ProtectionJournal
from bot.strategy_budget import BudgetError, StrategyBudgetStore
from bot.telegram_alerts import TelegramAlerter


class RecoveryMismatch(ValueError):
    """Evidence is incomplete or contradictory; no inferred execution is allowed."""


@dataclass(frozen=True)
class RecoveryReport:
    status: str
    halted: bool
    bot_btc: Decimal | None
    external_btc: Decimal | None
    confirmed_sl_btc: Decimal | None
    uncovered_btc: Decimal | None
    open_order_count: int | None
    reasons: tuple[str, ...]
    owner_action: str
    notification: str
    connection_restored: bool = False


class RecoveryManager:
    """One diagnostic pass. Only execution halt and verified fills are written."""

    ACTION = ("Вручную проверьте Demo BUY, комиссии, BTC и защитные Algo ID на OKX; "
              "не продавайте внешний BTC. До отдельной подтверждённой процедуры "
              "HALTED и запрет новых сделок сохраняются.")

    def __init__(self, *, engine: ExecutionEngine,
                 client_loader: Callable[[], DemoClient] = DemoClient.from_env_file,
                 alerter_loader: Callable[[], TelegramAlerter | None] = TelegramAlerter.from_env_file,
                 budget_store: StrategyBudgetStore | None = None) -> None:
        self.engine = engine
        self.client_loader = client_loader
        self.alerter_loader = alerter_loader
        self.budget_store = budget_store or StrategyBudgetStore(engine.path)

    def _notify(self, *events: str) -> str:
        try:
            alerter = self.alerter_loader()
            if alerter is None:
                return "не настроен в локальном .env"
            results = [alerter.notify(event) for event in events]
            return "доставлено" if results and all(results) else "доставка не подтверждена"
        except Exception:
            return "доставка не подтверждена"

    @staticmethod
    def _balance(value: object, label: str) -> CurrencyBalance:
        if not isinstance(value, CurrencyBalance):
            raise RecoveryMismatch(f"{label}: баланс не подтверждён")
        if any(not isinstance(x, Decimal) or not x.is_finite() or x < 0
               for x in (value.total, value.available)) or value.available > value.total:
            raise RecoveryMismatch(f"{label}: баланс противоречив")
        return value

    def _local_fills(self, intent: ExecutionIntent) -> dict[str, tuple[Decimal, Decimal, Decimal, str]]:
        try:
            with closing(sqlite3.connect(self.engine.path.resolve().as_uri() + "?mode=ro",
                                         uri=True, timeout=3)) as db:
                rows = db.execute("SELECT trade_id,quantity,price,fee,fee_ccy FROM execution_fills "
                                  "WHERE client_order_id=?", (intent.client_order_id,)).fetchall()
            fills = {row[0]: (Decimal(row[1]), Decimal(row[2]), Decimal(row[3]), row[4]) for row in rows}
            if (len(fills) != len(rows)
                    or sum((v[0] for v in fills.values()), Decimal(0)) != intent.filled_quantity
                    or sum((v[2] for v in fills.values() if v[3] == "BTC"), Decimal(0)) != intent.fee_btc
                    or sum((v[2] for v in fills.values() if v[3] == "USDT"), Decimal(0)) != intent.fee_usdt):
                raise RecoveryMismatch("Локальный журнал исполнений противоречив")
            return fills
        except (sqlite3.Error, OSError, ValueError) as exc:
            raise RecoveryMismatch("Локальный журнал исполнений недоступен") from exc

    @staticmethod
    def _validate_order(intent: ExecutionIntent, order: DemoOrder, now: datetime) -> None:
        if (not isinstance(order, DemoOrder) or order.client_order_id != intent.client_order_id
                or (intent.exchange_order_id is not None and order.order_id != intent.exchange_order_id)
                or order.size_btc != intent.quantity or order.limit_price_usdt != intent.entry_price
                or not isinstance(order.filled_btc, Decimal) or not order.filled_btc.is_finite()
                or order.filled_btc < 0 or order.filled_btc > order.size_btc
                or order.state not in {"live", "partially_filled", "filled", "canceled"}
                or (order.state == "live" and order.filled_btc != 0)
                or (order.state == "partially_filled" and
                    not 0 < order.filled_btc < order.size_btc)
                or (order.state == "filled" and order.filled_btc != order.size_btc)
                or not isinstance(order.checked_at_utc, datetime)
                or order.checked_at_utc.tzinfo is None
                or order.checked_at_utc.utcoffset() != timedelta(0)
                or order.checked_at_utc > now
                or now - order.checked_at_utc > timedelta(seconds=60)):
            raise RecoveryMismatch("BUY по clientOrderId не совпадает со свежим сохранённым намерением")

    @staticmethod
    def _validate_fills(order: DemoOrder, fills: object, intent: ExecutionIntent,
                        now: datetime) -> tuple[DemoFill, ...]:
        if not isinstance(fills, tuple) or len(fills) >= 100:
            raise RecoveryMismatch("Полнота истории BUY fills не доказана")
        ids: set[str] = set()
        for fill in fills:
            if (not isinstance(fill, DemoFill) or fill.order_id != order.order_id
                    or not isinstance(fill.trade_id, str) or not fill.trade_id or fill.trade_id in ids
                    or any(not isinstance(x, Decimal) or not x.is_finite() for x in
                           (fill.size_btc, fill.price_usdt, fill.fee))
                    or fill.size_btc <= 0 or fill.price_usdt <= 0 or fill.fee > 0
                    or fill.fee_currency not in {"BTC", "USDT"}
                    or not isinstance(fill.observed_at_utc, datetime)
                    or fill.observed_at_utc.tzinfo is None
                    or fill.observed_at_utc.utcoffset() != timedelta(0)
                    or fill.observed_at_utc < intent.created_at_utc
                    or fill.observed_at_utc > now):
                raise RecoveryMismatch("BUY fills или комиссии противоречивы")
            ids.add(fill.trade_id)
        if sum((fill.size_btc for fill in fills), Decimal(0)) != order.filled_btc:
            raise RecoveryMismatch("Сумма BUY fills не совпадает с accFillSz")
        return fills

    def _check_algos(self, client: DemoClient, intents: tuple[ExecutionIntent, ...],
                     pending: tuple[DemoAlgoOrder, ...]) -> list[str]:
        reasons: list[str] = []
        pending_by_id = {algo.algo_id: algo for algo in pending}
        if len(pending_by_id) != len(pending):
            raise RecoveryMismatch("Дублирующиеся Algo ID в ответе Demo")
        saved_ids: set[str] = set()
        for intent in intents:
            evidence = ProtectionJournal(self.engine).latest(intent.client_order_id)
            if evidence is None:
                if intent.owned_btc > 0:
                    reasons.append("Для BTC бота нет сохранённых идентификаторов SL/TP")
                continue
            if evidence.owned_btc != intent.owned_btc:
                raise RecoveryMismatch("Локальная запись защиты не совпадает с BUY fills")
            for algo_id, kind in ((evidence.stop_algo_id, "SL"), (evidence.take_algo_id, "TP")):
                if algo_id is None:
                    if intent.owned_btc > 0:
                        reasons.append(f"{kind}: биржевой Algo ID ещё не подтверждён")
                    continue
                if algo_id in saved_ids:
                    raise RecoveryMismatch("Один Algo ID повторно зарезервирован для разных защитных ног")
                saved_ids.add(algo_id)
                exact = client.get_algo_order_by_id(algo_id)
                listed = pending_by_id.get(algo_id)
                if not isinstance(exact, DemoAlgoOrder) or exact.algo_id != algo_id:
                    raise RecoveryMismatch("Точный Algo ID не подтверждён")
                if (listed is None or listed != exact or exact.side != "sell"
                        or exact.state != "live" or exact.size_btc != intent.owned_btc):
                    raise RecoveryMismatch("Algo ID, статус или защитный объём расходятся")
            if intent.owned_btc > 0:
                reasons.append("GET Algo не доказывает привязку SL/TP к BUY и исключение двойной продажи")
        if set(pending_by_id) - saved_ids:
            raise RecoveryMismatch("На Demo-счёте есть чужой или неизвестный Algo-ордер")
        return reasons

    def check_once(self) -> RecoveryReport:
        diagnostic = self.engine.diagnose()
        if not diagnostic.intents and ("Журнал" in diagnostic.reason or "журнал" in diagnostic.reason):
            return RecoveryReport("BLOCKED", diagnostic.halted, None, None, None, None, None,
                                  ("SQLite-журнал недоступен или повреждён",), self.ACTION, "не отправлено")
        intents = diagnostic.intents
        previous_reason = None
        halt_saved = diagnostic.halted
        if intents:
            try:
                previous_reason = self.engine.halt_reason_code()
                # This transaction must commit before any potentially failing GET or alert.
                self.engine.emergency_halt("RECOVERY_CHECK")
                halt_saved = True
            except (ExecutionError, OSError, sqlite3.Error):
                return RecoveryReport("BLOCKED", True, None, None, None, None, None,
                                      ("Не удалось надёжно сохранить HALTED",), self.ACTION, "не отправлено")
        reasons: list[str] = []
        bot_btc: Decimal | None = None
        external_btc: Decimal | None = None
        open_count: int | None = None
        restored = False
        try:
            started = self.engine.clock()
            if (not isinstance(started, datetime) or started.tzinfo is None
                    or started.utcoffset() != timedelta(0)):
                raise RecoveryMismatch("Время UTC для свежей сверки неизвестно")
            baseline = self.budget_store.read()
            if baseline is None:
                raise RecoveryMismatch("Подтверждённый бюджет и происхождение внешнего BTC отсутствуют")
            external_btc = baseline.demo_btc_total
            client = self.client_loader()
            if not isinstance(client, DemoClient) or client.config.mode != "DEMO":
                raise RecoveryMismatch("Подключение к отдельному Demo-режиму не подтверждено")
            btc, usdt = client.get_balances()
            btc = self._balance(btc, "BTC")
            usdt = self._balance(usdt, "USDT")
            open_orders = client.get_open_orders()
            pending_algos = client.get_pending_algo_orders()
            if not isinstance(open_orders, list) or not isinstance(pending_algos, tuple):
                raise RecoveryMismatch("Списки ордеров Demo не подтверждены")
            open_count = len(open_orders)
            observed_order_ids: set[str] = set()
            all_fills: list[DemoFill] = []
            if len(intents) > 1:
                raise RecoveryMismatch("Более одного локального BUY-намерения; двойной резерв исключить нельзя")
            for intent in intents:
                order = client.get_order_by_client_id(intent.client_order_id)
                self._validate_order(intent, order, self.engine.clock())
                observed_order_ids.add(order.order_id)
                fills = self._validate_fills(order, client.get_order_fills(order.order_id),
                                             intent, self.engine.clock())
                local = self._local_fills(intent)
                exchange = {fill.trade_id: (fill.size_btc, fill.price_usdt, fill.fee,
                                            fill.fee_currency) for fill in fills}
                if any(exchange.get(trade_id) != values for trade_id, values in local.items()):
                    raise RecoveryMismatch("Биржевые fills исчезли или не совпадают с SQLite")
                for fill in fills:
                    if fill.trade_id not in local:
                        self.engine.record_fill(intent.client_order_id, trade_id=fill.trade_id,
                                                quantity=fill.size_btc, price=fill.price_usdt,
                                                fee=fill.fee, fee_ccy=fill.fee_currency,
                                                exchange_order_id=order.order_id)
                all_fills.extend(fills)
                listed = [row for row in open_orders if row.get("ordId") == order.order_id]
                if len(listed) > 1 or (order.state in {"live", "partially_filled"}) != bool(listed):
                    raise RecoveryMismatch("Открытый BUY и список pending orders расходятся")
                if order.state in {"live", "partially_filled"}:
                    reasons.append("BUY всё ещё активен; дополнительное исполнение возможно")
            history = client.get_fills_history()
            if not isinstance(history, list) or len(history) >= 100:
                raise RecoveryMismatch("Полнота недавней истории Demo fills не доказана")
            own_trade_ids = {fill.trade_id for fill in all_fills}
            for row in history:
                if not isinstance(row, dict):
                    raise RecoveryMismatch("История Demo fills противоречива")
                if row.get("instId") == "BTC-USDT":
                    try:
                        stamp = datetime.fromtimestamp(int(row["ts"]) / 1000, tz=timezone.utc)
                    except (KeyError, TypeError, ValueError, OverflowError, OSError) as exc:
                        raise RecoveryMismatch("Время Demo fill неизвестно") from exc
                    if stamp >= baseline.confirmed_at_utc and row.get("tradeId") not in own_trade_ids:
                        raise RecoveryMismatch("Есть неучтённое BTC-USDT исполнение после подтверждения бюджета")
            if any(not isinstance(row, dict) or row.get("ordId") not in observed_order_ids
                   for row in open_orders):
                raise RecoveryMismatch("Есть чужой или неизвестный открытый SPOT-ордер")
            # Re-read persisted net fills. No journal row is treated as exchange proof
            # unless the exact trade ID, size, price and fee matched GET above.
            fresh = self.engine.diagnose()
            if intents and (not fresh.halted or len(fresh.intents) != len(intents)):
                raise RecoveryMismatch("Аварийный SQLite-статус изменился во время сверки")
            bot_btc = sum((intent.owned_btc for intent in fresh.intents), Decimal(0))
            expected_usdt = baseline.demo_usdt_total - sum(
                (fill.size_btc * fill.price_usdt -
                 (fill.fee if fill.fee_currency == "USDT" else Decimal(0)))
                for fill in all_fills
            )
            if (btc.total != external_btc + bot_btc or btc.available < external_btc
                    or usdt.total != expected_usdt):
                raise RecoveryMismatch("Движение BTC/USDT не объясняется подтверждёнными BUY fills и комиссиями")
            reasons.extend(self._check_algos(client, fresh.intents, pending_algos))
            finished = self.engine.clock()
            if (not isinstance(finished, datetime) or finished.tzinfo is None
                    or finished.utcoffset() != timedelta(0) or finished < started
                    or finished - started > timedelta(seconds=60)):
                raise RecoveryMismatch("Demo GET-снимок устарел во время сверки")
            if bot_btc > 0:
                reasons.append("Купленный ботом BTC не имеет подтверждённого SL на весь чистый объём")
            if not intents:
                reasons.append("BUY-намерений нет; это диагностическая GET-сверка, разрешение торговли не выдаётся")
            restored = previous_reason == "RECOVERY_NETWORK_ERROR"
            halt_code = "RECOVERY_UNPROTECTED" if bot_btc > 0 else "RECOVERY_CHECK"
            if intents:
                self.engine.emergency_halt(halt_code)
            events = []
            if intents and (previous_reason is None or not previous_reason.startswith("RECOVERY_")):
                events.append("HALTED")
            if bot_btc > 0 and previous_reason != "RECOVERY_UNPROTECTED":
                events.append("UNPROTECTED")
            if restored:
                events.append("CONNECTION_RESTORED")
        except RecoveryMismatch as exc:
            reasons.append(str(exc))
            if any(intent.owned_btc > 0 for intent in intents):
                reasons.append("SQLite показывает BTC бота; живой SL не подтверждён")
            if self.engine.path.is_file():
                try:
                    self.engine.emergency_halt("RECOVERY_MISMATCH")
                    halt_saved = True
                except ExecutionError:
                    reasons.append("Не удалось сохранить причину HALTED; торговля заблокирована")
            events = ["MISMATCH"] if halt_saved and previous_reason != "RECOVERY_MISMATCH" else []
            if halt_saved and previous_reason != "RECOVERY_MISMATCH" and any(intent.owned_btc > 0 for intent in intents):
                events.append("UNPROTECTED")
        except Exception:
            # Never expose exception text: some HTTP errors include credential-bearing URLs.
            reasons.append("Demo GET недоступен или ответ неполон; фактические исполнения неизвестны")
            if any(intent.owned_btc > 0 for intent in intents):
                reasons.append("SQLite показывает BTC бота; живой SL не подтверждён")
            if self.engine.path.is_file():
                try:
                    self.engine.emergency_halt("RECOVERY_NETWORK_ERROR")
                    halt_saved = True
                except ExecutionError:
                    reasons.append("Не удалось сохранить причину HALTED; торговля заблокирована")
            events = ["CONNECTION_LOST"] if halt_saved and previous_reason != "RECOVERY_NETWORK_ERROR" else []
            if halt_saved and previous_reason != "RECOVERY_NETWORK_ERROR" and any(intent.owned_btc > 0 for intent in intents):
                events.append("UNPROTECTED")
        notification = self._notify(*events) if events else "повторное уведомление не требуется"
        return RecoveryReport("BLOCKED", self.engine.diagnose().halted,
                              bot_btc, external_btc, Decimal(0) if bot_btc is not None else None,
                              bot_btc, open_count, tuple(reasons), self.ACTION,
                              notification, restored)
