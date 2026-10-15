"""Read-only, non-actionable OKX Demo BTC-USDT LIMIT BUY preview.

No API transport or order payload is constructed here. The protection gate is
deliberately closed: attached TP/SL is not proven to cover every partial fill
atomically, so no preview can authorize submission in this stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_DOWN

from bot.execution_engine import ExecutionEngine, ExecutionError
from bot.market_data import MarketSnapshot, SpotInstrument
from bot.reconcile import ReconcileReport, ReconcileStatus
from bot.risk_manager import ExecutionCosts, InstrumentLimits
from bot.strategy import Signal, StrategyResult


PROTECTION_BLOCK = "Атомарная защита SL/TP для каждого частичного исполнения не подтверждена"


@dataclass(frozen=True)
class OrderPreview:
    status: str
    reasons: tuple[str, ...]
    candle_closed_at: datetime | None = None
    signal: str | None = None
    reference_price_usdt: Decimal | None = None
    estimated_entry_usdt: Decimal | None = None
    estimated_stop_usdt: Decimal | None = None
    estimated_take_usdt: Decimal | None = None
    quantity_btc: Decimal | None = None
    estimated_risk_usdt: Decimal | None = None
    estimated_cost_usdt: Decimal | None = None
    virtual_usdt_available: Decimal | None = None
    external_btc: Decimal | None = None


def _floor(value: Decimal, step: Decimal) -> Decimal:
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def _ceil(value: Decimal, step: Decimal) -> Decimal:
    return (value / step).to_integral_value(rounding=ROUND_CEILING) * step


def preview_demo_buy(
    *, market: MarketSnapshot | None, signal: StrategyResult | None,
    reconciliation: ReconcileReport | None, engine: ExecutionEngine,
    verified_max_size_btc: Decimal | None = None,
    verified_costs: ExecutionCosts | None = None,
    now_utc: datetime | None = None,
) -> OrderPreview:
    """Show a risk candidate when fully evidenced, but always block execution."""
    now = now_utc or datetime.now(timezone.utc)
    reasons = [PROTECTION_BLOCK]
    entry = stop = take = quantity = risk = cost = reference = None
    candle = signal.candle_closed_at if isinstance(signal, StrategyResult) else None
    signal_text = signal.signal.value if isinstance(signal, StrategyResult) and isinstance(signal.signal, Signal) else None
    available = reconciliation.usdt_available if isinstance(reconciliation, ReconcileReport) else None
    external = reconciliation.external_btc if isinstance(reconciliation, ReconcileReport) else None

    valid_now = isinstance(now, datetime) and now.tzinfo is not None and now.utcoffset() == timedelta(0)
    if not valid_now:
        reasons.append("Текущее UTC-время неизвестно")
    if not isinstance(market, MarketSnapshot) or not isinstance(market.instrument, SpotInstrument):
        reasons.append("Рыночные данные BTC-USDT отсутствуют")
    else:
        instrument = market.instrument
        reference = market.last_price if isinstance(market.last_price, Decimal) else None
        if (instrument.instrument_id != "BTC-USDT" or instrument.state != "live"
                or instrument.base_currency != "BTC" or instrument.quote_currency != "USDT"
                or not isinstance(instrument.tick_size, Decimal) or not instrument.tick_size.is_finite()
                or instrument.tick_size <= 0 or not isinstance(reference, Decimal)
                or not reference.is_finite() or reference <= 0):
            reasons.append("Цена или ограничения BTC-USDT противоречивы")
        else:
            entry = _ceil(reference, instrument.tick_size)
            stop = _floor(entry * Decimal("0.99"), instrument.tick_size)
            take = _ceil(entry * Decimal("1.02"), instrument.tick_size)
            if stop <= 0 or stop >= entry or take <= entry:
                reasons.append("Ценовой шаг не позволяет рассчитать SL 1% / TP 2%")
                entry = stop = take = None
            if (not valid_now or not isinstance(market.ticker_as_of_utc, datetime)
                    or market.ticker_as_of_utc.tzinfo is None
                    or market.ticker_as_of_utc.utcoffset() != timedelta(0)
                    or market.ticker_as_of_utc > now
                    or now - market.ticker_as_of_utc > timedelta(seconds=60)):
                reasons.append("Текущая цена устарела или её время неизвестно")

    if signal_text != "BUY" or candle is None:
        reasons.append("Нет подтверждённого сигнала EMA BUY; внешний BTC продавать запрещено")
    elif (not valid_now or not isinstance(candle, datetime)
          or candle.tzinfo is None or candle.utcoffset() != timedelta(0)
          or candle > now or now - candle > timedelta(minutes=30)):
        reasons.append("Закрытая свеча устарела или её UTC-время противоречиво")

    if not isinstance(reconciliation, ReconcileReport) or reconciliation.status != ReconcileStatus.READY:
        reasons.append("Нет полной свежей сверки Demo-счёта, бюджета и Risk Manager")
        if isinstance(reconciliation, ReconcileReport):
            reasons.extend(reconciliation.reasons)
    if not isinstance(verified_max_size_btc, Decimal) or not verified_max_size_btc.is_finite() or verified_max_size_btc <= 0:
        reasons.append("Максимальный размер LIMIT-ордера в единицах BTC не подтверждён")
    if not isinstance(verified_costs, ExecutionCosts):
        reasons.append("Комиссии и ожидаемое проскальзывание не подтверждены")

    if len(reasons) == 1 and entry is not None and isinstance(market, MarketSnapshot):
        try:
            spot = market.instrument
            limits = InstrumentLimits("BTC-USDT", "live", spot.min_size,
                                      spot.lot_size, spot.tick_size, verified_max_size_btc)
            decision = engine.preview_candidate(
                reconciliation=reconciliation, instrument=limits, costs=verified_costs,
                entry_price=entry, max_size_btc_verified=True, slippage_verified=True,
            )
        except (ExecutionError, ValueError, TypeError, ArithmeticError):
            reasons.append("Risk Manager или локальный исполнитель заблокировал BUY")
        else:
            quantity = decision.quantity
            entry = decision.entry_price
            stop = decision.stop_price
            take = decision.take_profit_price
            risk = decision.estimated_risk_usdt
            cost = decision.estimated_cost_usdt

    return OrderPreview("BLOCK", tuple(reasons), candle, signal_text, reference,
                        entry, stop, take, quantity, risk, cost, available, external)
