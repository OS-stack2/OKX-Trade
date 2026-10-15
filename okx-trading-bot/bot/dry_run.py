"""One-shot, non-executing Demo analysis orchestration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from bot.demo_client import CurrencyBalance, DemoSummary
from bot.journal import CycleDraft, CycleRecord, CycleStatus, DecisionJournal
from bot.market_data import MarketSnapshot, SpotInstrument
from bot.risk_manager import (
    AccountSnapshot, ExecutionCosts, InstrumentLimits, RiskManager, RiskStatus,
)
from bot.strategy import Signal, evaluate_signal
from config import ConfigError


MAX_CANDLE_AGE = timedelta(minutes=30)
MAX_TICKER_AGE = timedelta(seconds=60)
MAX_DEMO_AGE = timedelta(seconds=60)


@dataclass(frozen=True)
class RiskEvidence:
    """Fully verified account, instrument, costs and bot-position provenance.

    The stage 4B.1 CLI deliberately has no provider for missing data. Offline
    callers can supply evidence to exercise a complete hypothetical decision.
    """

    account: AccountSnapshot
    instrument: InstrumentLimits
    costs: ExecutionCosts
    provenance_verified: bool


def _fresh(value: datetime | None, now: datetime, max_age: timedelta) -> bool:
    return (
        isinstance(value, datetime) and value.tzinfo is not None
        and value.utcoffset() == timedelta(0) and value <= now
        and now - value <= max_age
    )


def _valid_balance(total: object, available: object) -> bool:
    return (
        isinstance(total, Decimal) and total.is_finite() and total >= 0
        and isinstance(available, Decimal) and available.is_finite()
        and 0 <= available <= total
    )


def _settings_parameters(manager: RiskManager) -> dict[str, str]:
    settings = manager.settings
    return {name: str(getattr(settings, name)) for name in (
        "risk_per_trade", "stop_loss_pct", "take_profit_pct", "daily_loss_limit_pct",
        "max_open_positions", "max_new_entries_per_day",
    )}


def _evidence_parameters(evidence: RiskEvidence, parameters: dict[str, str]) -> dict[str, str]:
    account, instrument, costs = evidence.account, evidence.instrument, evidence.costs
    fields = {
        "equity_usdt": account.equity_usdt,
        "day_start_capital_usdt": account.day_start_capital_usdt,
        "available_usdt": account.available_usdt,
        "realized_pnl_today_usdt": account.realized_pnl_today_usdt,
        "unrealized_pnl_usdt": account.unrealized_pnl_usdt,
        "open_positions": account.open_positions,
        "pending_orders": account.pending_orders,
        "entries_today": account.entries_today,
        "entry_fee_rate": costs.entry_fee_rate,
        "exit_fee_rate": costs.exit_fee_rate,
        "entry_slippage_rate": costs.entry_slippage_rate,
        "stop_slippage_rate": costs.stop_slippage_rate,
        "instrument_min_size": instrument.min_size,
        "instrument_lot_size": instrument.lot_size,
        "instrument_tick_size": instrument.tick_size,
        "instrument_max_size": instrument.max_size,
    }
    # Only numeric, finite values are eligible for the journal. Invalid input
    # is denied by RiskManager and never copied into the persistent record.
    for name, value in fields.items():
        if isinstance(value, Decimal) and value.is_finite():
            parameters[name] = str(value)
        elif type(value) is int:
            parameters[name] = str(value)
    return parameters


class DryRunOrchestrator:
    def __init__(
        self,
        *,
        market_provider: Callable[[], MarketSnapshot],
        demo_provider: Callable[[], DemoSummary],
        risk_manager: RiskManager,
        journal: DecisionJournal,
        evidence_provider: Callable[[MarketSnapshot, DemoSummary, datetime], RiskEvidence | None] | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.market_provider = market_provider
        self.demo_provider = demo_provider
        self.risk_manager = risk_manager
        self.journal = journal
        self.evidence_provider = evidence_provider
        self.clock = clock

    def run_once(self) -> CycleRecord:
        now = self.clock()
        if not _fresh(now, now, timedelta(0)):
            raise ValueError("Для анализа требуется текущее время UTC")
        try:
            self.risk_manager.validate_settings()
            risk_parameters = _settings_parameters(self.risk_manager)
        except Exception:
            return self.journal.append(CycleDraft(
                created_at_utc=now, candle_closed_at=None, signal=None,
                status=CycleStatus.BLOCK, reason="Настройки Risk Manager отсутствуют или некорректны",
                risk_parameters={},
            ))
        draft = CycleDraft(
            created_at_utc=now, candle_closed_at=None, signal=None,
            status=CycleStatus.BLOCK, reason="Неполные данные для проверки",
            risk_parameters=risk_parameters,
        )

        try:
            market = self.market_provider()
            if not isinstance(market, MarketSnapshot) or not isinstance(market.instrument, SpotInstrument):
                raise ValueError("Invalid market snapshot")
            if market.instrument.instrument_id != "BTC-USDT" or any(
                not isinstance(value, Decimal) or not value.is_finite() or value <= 0
                for value in (market.instrument.tick_size, market.instrument.lot_size, market.instrument.min_size)
            ):
                raise ValueError("Invalid BTC-USDT instrument")
            if not isinstance(market.last_price, Decimal) or not market.last_price.is_finite() or market.last_price <= 0:
                raise ValueError("Invalid ticker price")
            strategy = evaluate_signal(market.candles)
        except ConfigError:
            return self.journal.append(replace(
                draft, reason="Нужен локальный .env с OKX_MODE=DEMO и отдельным Demo API-ключом",
            ))
        except Exception:
            return self.journal.append(replace(draft, reason="Рыночные данные или закрытые свечи недоступны либо некорректны"))
        now = self.clock()
        draft = replace(
            draft, created_at_utc=now, candle_closed_at=strategy.candle_closed_at, signal=strategy.signal.value,
            ema9=strategy.ema9, ema21=strategy.ema21,
        )
        if not _fresh(strategy.candle_closed_at, now, MAX_CANDLE_AGE):
            return self.journal.append(replace(draft, reason="Последняя закрытая свеча устарела или имеет неверное время"))
        if not _fresh(market.ticker_as_of_utc, now, MAX_TICKER_AGE):
            return self.journal.append(replace(draft, reason="Текущая цена BTC-USDT не имеет свежего времени OKX"))

        try:
            demo = self.demo_provider()
        except Exception:
            return self.journal.append(replace(draft, reason="Demo API или авторизация недоступны; состояние счёта неизвестно"))
        now = self.clock()
        draft = replace(draft, created_at_utc=now)
        if not isinstance(demo, DemoSummary) or not _fresh(demo.checked_at_utc, now, MAX_DEMO_AGE):
            return self.journal.append(replace(draft, reason="Снимок Demo-счёта отсутствует или устарел"))
        if (not isinstance(demo.btc, CurrencyBalance) or not isinstance(demo.usdt, CurrencyBalance)
                or not _valid_balance(demo.btc.total, demo.btc.available)
                or not _valid_balance(demo.usdt.total, demo.usdt.available)
                or type(demo.open_order_count) is not int or demo.open_order_count < 0
                or type(demo.recent_fill_count) is not int or demo.recent_fill_count < 0):
            return self.journal.append(replace(draft, reason="Demo-счёт вернул противоречивые данные"))
        draft = replace(
            draft, virtual_btc_available=demo.btc.available,
            virtual_usdt_available=demo.usdt.available,
        )

        if strategy.signal == Signal.HOLD:
            return self.journal.append(replace(draft, status=CycleStatus.HOLD, reason=strategy.reason))
        if strategy.signal == Signal.SELL:
            return self.journal.append(replace(
                draft, reason="SELL заблокирован: собственная позиция бота не подтверждена; внешний BTC не продаётся",
            ))
        if strategy.signal != Signal.BUY:
            return self.journal.append(replace(draft, reason="Неизвестный торговый сигнал"))
        if demo.open_order_count:
            return self.journal.append(replace(draft, reason="Есть открытый SPOT-ордер; новый вход заблокирован"))
        if self.evidence_provider is None:
            return self.journal.append(replace(
                draft, reason="Нет подтверждённых дневного капитала, PnL, собственных позиций, комиссий и лимита размера BTC",
            ))

        try:
            evidence = self.evidence_provider(market, demo, now)
        except Exception:
            evidence = None
        if not isinstance(evidence, RiskEvidence) or evidence.provenance_verified is not True:
            return self.journal.append(replace(draft, reason="Полное подтверждённое состояние риска и происхождение позиций неизвестны"))
        if (not isinstance(evidence.account, AccountSnapshot)
                or not isinstance(evidence.instrument, InstrumentLimits)
                or not isinstance(evidence.costs, ExecutionCosts)
                or evidence.account.available_usdt != demo.usdt.available
                or evidence.account.pending_orders != demo.open_order_count
                or evidence.instrument.instrument_id != market.instrument.instrument_id
                or evidence.instrument.min_size != market.instrument.min_size
                or evidence.instrument.lot_size != market.instrument.lot_size
                or evidence.instrument.tick_size != market.instrument.tick_size):
            return self.journal.append(replace(draft, reason="Подтверждённые данные риска противоречат Demo API или инструменту OKX"))

        parameters = _evidence_parameters(evidence, dict(draft.risk_parameters or {}))
        draft = replace(draft, risk_parameters=parameters)
        try:
            decision = self.risk_manager.preview_buy(
                account=evidence.account, instrument=evidence.instrument, costs=evidence.costs,
                entry_price=market.last_price, now_utc=now,
            )
        except Exception:
            return self.journal.append(replace(draft, reason="Предварительная проверка риска недоступна"))
        if decision.status != RiskStatus.ALLOW:
            return self.journal.append(replace(
                draft, reason=decision.reason,
                entry_price=decision.entry_price, stop_price=decision.stop_price,
                take_profit_price=decision.take_profit_price,
            ))
        return self.journal.append(replace(
            draft, status=CycleStatus.ALLOW, reason=decision.reason,
            quantity=decision.quantity, entry_price=decision.entry_price,
            stop_price=decision.stop_price, take_profit_price=decision.take_profit_price,
            estimated_risk_usdt=decision.estimated_risk_usdt,
        ))
