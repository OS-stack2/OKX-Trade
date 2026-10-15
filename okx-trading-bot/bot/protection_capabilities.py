"""GET-only capability inventory for BTC-USDT SPOT Demo.

This is a diagnostic, not an authorization gate. A successful GET or a field in
the official SDK cannot prove attached protection of every partial fill.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from bot.demo_client import DemoAlgoOrder, DemoClient, DemoError, SpotFeeRates
from bot.execution_engine import ExecutionEngine, ExecutionError, ExecutionState
from bot.market_data import SpotInstrument, read_spot_instrument
from bot.okx_client import OkxClient, OkxError
from bot.protection_model import ProtectionJournal


SDK_EVIDENCE = (
    "Официальный Python SDK OKX содержит attachAlgoOrds у place_order и GET "
    "инструмента, ордеров, исполнений, ожидающих Algo-ордеров и Algo по ID. "
    "Это описание интерфейса, не подтверждение работы attached TP/SL на Demo.",
)
ATOMIC_UNKNOWN = (
    "Не доказано, что каждый частичный BUY атомарно получает действующий SL "
    "на чистый BTC после комиссии; новые сделки BLOCK."
)
REGION_UNKNOWN = (
    "Региональная доступность attached TP/SL для BTC-USDT SPOT Demo не подтверждена."
)


@dataclass(frozen=True)
class CapabilityReport:
    status: str
    sdk_evidence: tuple[str, ...]
    api_observations: tuple[str, ...]
    unknowns: tuple[str, ...]
    instrument: SpotInstrument | None = None
    fees: SpotFeeRates | None = None
    regular_open_orders: int | None = None
    pending_algo_orders: int | None = None


def inspect_protection_capabilities(
    *, public_client: OkxClient, demo_client: DemoClient,
    engine: ExecutionEngine,
) -> CapabilityReport:
    """Read public and Demo GET APIs plus local SQLite, without any writes."""
    observations: list[str] = []
    unknowns: list[str] = [REGION_UNKNOWN, ATOMIC_UNKNOWN]
    instrument = None
    fees = None
    order_count = None
    algo_count = None
    config = getattr(demo_client, "config", None)
    public_config = getattr(public_client, "config", None)
    if (config is None or config.mode != "DEMO" or public_config is None
            or public_config.base_url != config.base_url):
        unknowns.append("DEMO-режим и совпадение региональных REST-доменов не подтверждены; GET не выполнялись.")
        return CapabilityReport("BLOCK", SDK_EVIDENCE, tuple(observations), tuple(unknowns))

    try:
        candidate = read_spot_instrument(public_client)
        if (candidate.state != "live" or candidate.base_currency != "BTC"
                or candidate.quote_currency != "USDT"):
            raise ValueError("unexpected instrument")
        instrument = candidate
        observations.append(
            f"Публичный GET BTC-USDT SPOT: minSz={candidate.min_size}, "
            f"lotSz={candidate.lot_size}, tickSz={candidate.tick_size}, "
            f"maxLmtSz={candidate.max_limit_size if candidate.max_limit_size is not None else 'неизвестен'} "
            "(единицы поля OKX не уточнялись)."
        )
        if candidate.max_limit_size is None:
            unknowns.append("maxLmtSz отсутствует в публичном ответе; верхний размер LIMIT неизвестен.")
    except (OkxError, OSError, ValueError, TypeError, AttributeError, ArithmeticError):
        unknowns.append("Публичный GET ограничений BTC-USDT SPOT недоступен или противоречив.")

    try:
        candidate_fees = demo_client.get_spot_fee_rates()
        if (not isinstance(candidate_fees, SpotFeeRates)
                or any(not isinstance(x, Decimal) or not x.is_finite() or not -1 < x < 1
                       for x in (candidate_fees.maker, candidate_fees.taker))):
            raise ValueError("invalid fees")
        fees = candidate_fees
        observations.append(f"Demo GET комиссии BTC-USDT SPOT (знак как в API): "
                            f"maker={fees.maker}, taker={fees.taker}.")
        unknowns.append("Фактическая комиссия исполнения и её валюта зависят от конкретного fill; ставки не заменяют сверку fills.")
    except (DemoError, OSError, ValueError, TypeError, AttributeError):
        unknowns.append("Demo GET ставок комиссии недоступен или противоречив.")

    try:
        orders = demo_client.get_open_orders()
        if not isinstance(orders, list) or any(not isinstance(row, dict) for row in orders):
            raise ValueError("invalid orders")
        order_count = len(orders)
        observations.append(f"Demo GET открытых SPOT-ордеров: {order_count}.")
        if orders:
            unknowns.append("Есть открытые SPOT-ордера: происхождение и доступный BTC нельзя вывести из числа ордеров.")
        else:
            unknowns.append("Пустой список открытых ордеров не доказывает поддержку attached TP/SL.")
    except (DemoError, OSError, ValueError, TypeError):
        unknowns.append("Demo GET открытых SPOT-ордеров недоступен или неполон.")

    try:
        algos = demo_client.get_pending_algo_orders()
        if not isinstance(algos, tuple) or any(not isinstance(row, DemoAlgoOrder) for row in algos):
            raise ValueError("invalid algos")
        algo_count = len(algos)
        observations.append(f"Demo GET ожидающих conditional/OCO Algo-ордеров BTC-USDT SPOT: {algo_count}.")
        unknowns.append("Списки conditional/OCO не доказывают полноту всех attached Algo-типов или связь с конкретным BUY.")
        if algos:
            unknowns.append("Найденные Algo-ордера нельзя считать защитой BTC бота без подтверждённых ID, объёма и происхождения.")
    except (DemoError, OSError, ValueError, TypeError):
        unknowns.append("Demo GET ожидающих Algo-ордеров недоступен или неполон.")

    diagnostic = engine.diagnose()
    if diagnostic.halted:
        observations.append("Локальный исполнитель: HALTED (сохранено в SQLite).")
    elif diagnostic.intents:
        observations.append(f"Локальный исполнитель: намерений {len(diagnostic.intents)}; новых входов нет.")
    else:
        observations.append("Локальный исполнитель: подтверждённых BUY-намерений нет.")
    if diagnostic.blocked:
        unknowns.append("Локальное состояние исполнителя не разрешает новый BUY.")

    protection_journal = ProtectionJournal(engine)
    for intent in diagnostic.intents:
        if intent.state in (ExecutionState.PLANNED, ExecutionState.VALIDATED,
                            ExecutionState.REJECTED):
            continue
        try:
            order = demo_client.get_order_by_client_id(intent.client_order_id)
            fills = demo_client.get_order_fills(order.order_id)
            if (not isinstance(fills, tuple)
                    or any(fill.order_id != order.order_id or fill.fee_currency not in ("BTC", "USDT")
                           or not fill.fee.is_finite() or fill.fee > 0 for fill in fills)
                    or len({fill.trade_id for fill in fills}) != len(fills)
                    or order.order_id != intent.exchange_order_id
                    or order.filled_btc != intent.filled_quantity
                    or sum((fill.size_btc for fill in fills), Decimal(0)) != order.filled_btc
                    or sum((fill.fee for fill in fills if fill.fee_currency == "BTC"), Decimal(0)) != intent.fee_btc
                    or sum((fill.fee for fill in fills if fill.fee_currency == "USDT"), Decimal(0)) != intent.fee_usdt):
                raise ValueError("order/fill mismatch")
            observations.append(f"Demo GET собственного BUY: state={order.state}, "
                                f"исполнено={order.filled_btc} BTC, fills={len(fills)}.")
            if order.state == "partially_filled":
                unknowns.append("Остаток активного BUY может исполниться до появления SL на дополнительный BTC.")
            elif order.state == "canceled" and order.filled_btc > 0:
                unknowns.append("После отмены остатка BUY нужен SL на уже купленный чистый BTC и окончательный статус отмены.")
            elif order.state == "filled":
                unknowns.append("Полный BUY не доказывает наличие действующего SL на весь чистый BTC.")
        except (DemoError, OSError, ValueError, TypeError, AttributeError):
            unknowns.append("BUY по сохранённому clientOrderId/ordId или его исполнения не сверены.")
        try:
            saved = protection_journal.latest(intent.client_order_id)
            if saved is None or not saved.stop_algo_id or not saved.take_algo_id:
                unknowns.append("В SQLite нет двух подтверждённых Algo ID для собственного BUY.")
                continue
            if saved.stop_algo_id == saved.take_algo_id:
                raise ValueError("duplicate algo IDs")
            stop = demo_client.get_algo_order_by_id(saved.stop_algo_id)
            take = demo_client.get_algo_order_by_id(saved.take_algo_id)
            if (stop.side != "sell" or take.side != "sell"
                    or stop.state != "live" or take.state != "live"
                    or stop.size_btc != saved.owned_btc
                    or take.size_btc != saved.owned_btc):
                raise ValueError("algo size/side mismatch")
            observations.append("Demo GET двух сохранённых Algo ID: размеры совпали с чистым BTC бота.")
            unknowns.append("GET Algo по ID не подтверждает атомарность, взаимоисключение и привязку к каждой частичной покупке.")
        except (DemoError, ExecutionError, OSError, ValueError, TypeError, AttributeError):
            unknowns.append("Сохранённые Algo ID, статус или защитный объём не подтверждены Demo GET.")

    return CapabilityReport("BLOCK", SDK_EVIDENCE, tuple(observations), tuple(unknowns),
                            instrument, fees, order_count, algo_count)
