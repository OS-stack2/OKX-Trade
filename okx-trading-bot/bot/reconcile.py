"""Read-only reconciliation of BTC-USDT Demo evidence with the local risk ledger."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from enum import StrEnum

from bot.demo_client import CurrencyBalance, DemoClient, SpotFeeRates
from bot.dry_run import MAX_DEMO_AGE, MAX_TICKER_AGE, _fresh, _valid_balance
from bot.journal import DecisionJournal, JournalError
from bot.market_data import MarketSnapshot, SpotInstrument
from bot.risk_manager import RiskManager, RiskStateError


class ReconcileStatus(StrEnum):
    READY = "READY"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class ConfirmedDayEvidence:
    """Explicitly verified daily ledger; there is no automatic CLI provider yet."""

    as_of_utc: datetime
    day_start_capital_usdt: Decimal
    realized_pnl_today_usdt: Decimal
    unrealized_pnl_usdt: Decimal
    bot_btc: Decimal
    entries_today: int
    complete_and_verified: bool
    net_external_cash_flow_usdt: Decimal


@dataclass(frozen=True)
class ReconcileReport:
    status: ReconcileStatus
    reasons: tuple[str, ...]
    btc_total: Decimal | None = None
    btc_available: Decimal | None = None
    usdt_total: Decimal | None = None
    usdt_available: Decimal | None = None
    btc_price_usdt: Decimal | None = None
    portfolio_value_usdt: Decimal | None = None
    external_btc: Decimal | None = None
    bot_btc: Decimal | None = None
    open_order_count: int | None = None
    fill_count: int | None = None
    fees_by_currency: tuple[tuple[str, Decimal], ...] | None = None
    maker_fee_rate: Decimal | None = None
    taker_fee_rate: Decimal | None = None
    instrument: SpotInstrument | None = None
    confirmed_realized_pnl_usdt: Decimal | None = None
    confirmed_unrealized_pnl_usdt: Decimal | None = None
    daily_loss_usdt: Decimal | None = None
    daily_loss_limit_usdt: Decimal | None = None
    daily_risk_remaining_usdt: Decimal | None = None


def _decimal(value: object, *, positive: bool = False) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite() or (positive and value <= 0):
        raise ValueError("Некорректная сумма или цена")
    return value


def _fill_fees(rows: list[dict], now: datetime) -> tuple[tuple[str, Decimal], ...]:
    if len(rows) >= 100:
        raise ValueError("История исполнений ограничена 100 записями; полнота не доказана")
    fees: dict[str, Decimal] = {"BTC": Decimal(0), "USDT": Decimal(0)}
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or row.get("instType") != "SPOT" or row.get("instId") != "BTC-USDT":
            raise ValueError("Есть исполнение вне BTC-USDT SPOT; происхождение средств неизвестно")
        trade_id = row.get("tradeId")
        currency = row.get("feeCcy")
        if (not isinstance(trade_id, str) or not trade_id or trade_id in seen
                or row.get("side") not in ("buy", "sell") or currency not in fees):
            raise ValueError("История исполнений неполна или противоречива")
        seen.add(trade_id)
        try:
            fee = Decimal(row["fee"])
            size = Decimal(row["fillSz"])
            price = Decimal(row["fillPx"])
            timestamp = datetime.fromtimestamp(int(row["ts"]) / 1000, tz=timezone.utc)
        except (KeyError, TypeError, ValueError, InvalidOperation, OverflowError, OSError) as exc:
            raise ValueError("В исполнении отсутствуют сумма, комиссия или время") from exc
        if (not fee.is_finite() or not size.is_finite() or size <= 0
                or not price.is_finite() or price <= 0 or timestamp > now):
            raise ValueError("Исполнение содержит противоречивые данные")
        fees[currency] += fee
    return tuple(fees.items())


class DemoReconciler:
    def __init__(
        self, *, market_provider: Callable[[], MarketSnapshot], demo_client: DemoClient,
        journal: DecisionJournal, risk_manager: RiskManager,
        day_provider: Callable[[], ConfirmedDayEvidence] | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.market_provider = market_provider
        self.demo_client = demo_client
        self.journal = journal
        self.risk_manager = risk_manager
        self.day_provider = day_provider
        self.clock = clock

    def run_once(self) -> ReconcileReport:
        """Never initializes the UTC day, reserves risk, or sends a private write."""
        now = self.clock()
        if not _fresh(now, now, timedelta(0)):
            return ReconcileReport(ReconcileStatus.BLOCKED, ("Текущее время UTC неизвестно",))
        reasons: list[str] = []
        values: dict = {}
        try:
            self.risk_manager.validate_settings()
            audit = self.journal.audit_read_only()
            if audit.executed_count:
                reasons.append("Журнал содержит исполнение без проверенного происхождения")
        except (JournalError, RiskStateError, OSError, ValueError):
            reasons.append("Журнал решений или настройки Risk Manager недоступны либо противоречивы")
        try:
            market = self.market_provider()
            instrument = market.instrument
            price = _decimal(market.last_price, positive=True)
            if (not isinstance(instrument, SpotInstrument) or instrument.instrument_id != "BTC-USDT"
                    or instrument.state != "live" or instrument.base_currency != "BTC"
                    or instrument.quote_currency != "USDT"
                    or not _fresh(market.ticker_as_of_utc, self.clock(), MAX_TICKER_AGE)):
                raise ValueError("Цена или инструмент BTC-USDT SPOT неизвестны либо устарели")
            for size in (instrument.min_size, instrument.lot_size, instrument.tick_size,
                         instrument.max_limit_size, instrument.max_market_size):
                _decimal(size, positive=True)
            values.update(instrument=instrument, btc_price_usdt=price)
        except Exception:
            reasons.append("Цена или ограничения BTC-USDT SPOT недоступны, устарели либо противоречивы")
        try:
            btc, usdt = self.demo_client.get_balances()
            checked_at = self.clock()
            if (not isinstance(btc, CurrencyBalance) or not isinstance(usdt, CurrencyBalance)
                    or not _valid_balance(btc.total, btc.available)
                    or not _valid_balance(usdt.total, usdt.available)
                    or not _fresh(checked_at, self.clock(), MAX_DEMO_AGE)):
                raise ValueError("Недостоверный баланс")
            values.update(btc_total=btc.total, btc_available=btc.available,
                          usdt_total=usdt.total, usdt_available=usdt.available)
            if "btc_price_usdt" in values:
                values["portfolio_value_usdt"] = usdt.total + btc.total * values["btc_price_usdt"]
        except Exception:
            reasons.append("Виртуальные балансы BTC/USDT Demo неизвестны или противоречивы")
        try:
            orders = self.demo_client.get_open_orders()
            if not isinstance(orders, list):
                raise ValueError("Нет списка ордеров")
            values["open_order_count"] = len(orders)
            if orders:
                reasons.append("Есть открытые Demo SPOT-ордера; происхождение и резерв средств неизвестны")
        except Exception:
            reasons.append("Список открытых Demo SPOT-ордеров недоступен")
        try:
            fills = self.demo_client.get_fills_history()
            if not isinstance(fills, list):
                raise ValueError("Нет истории исполнений")
            values["fill_count"] = len(fills)
            values["fees_by_currency"] = _fill_fees(fills, self.clock())
            if fills:
                reasons.append("Исполнения не связаны с подтверждёнными позициями бота в журнале")
        except Exception:
            reasons.append("История исполнений или их комиссий неполна либо противоречива")
        try:
            rates = self.demo_client.get_spot_fee_rates()
            if not isinstance(rates, SpotFeeRates):
                raise ValueError("Нет ставки комиссии")
            for rate in (rates.maker, rates.taker):
                if not isinstance(rate, Decimal) or not rate.is_finite() or not -1 < rate < 1:
                    raise ValueError("Недостоверная ставка комиссии")
            values.update(maker_fee_rate=rates.maker, taker_fee_rate=rates.taker)
        except Exception:
            reasons.append("Ставки Demo-комиссии BTC-USDT SPOT недоступны")

        # The existing journal records hypothetical decisions only, never fills.
        # Until a verified ownership ledger exists, all BTC is treated as external.
        if "btc_total" in values:
            values["external_btc"] = values["btc_total"]
            values["bot_btc"] = Decimal(0)
        try:
            state = self.risk_manager.inspect_daily_state(self.clock())
            if state.pending_entry or state.loss_locked or state.entries_used:
                reasons.append("Дневное состояние Risk Manager содержит резерв, блокировку или вход без сверки")
        except (RiskStateError, OSError, ValueError):
            state = None
            reasons.append("Подтверждённая дневная база Risk Manager отсутствует или недоступна")
        try:
            evidence = self.day_provider() if self.day_provider is not None else None
        except Exception:
            evidence = None
        if (not isinstance(evidence, ConfirmedDayEvidence) or evidence.complete_and_verified is not True
                or not _fresh(evidence.as_of_utc, self.clock(), MAX_DEMO_AGE)
                or state is None or evidence.as_of_utc.date() != state.day_utc
                or evidence.day_start_capital_usdt != state.start_capital_usdt
                or type(evidence.entries_today) is not int or evidence.entries_today != state.entries_used
                or evidence.bot_btc != 0 or "fill_count" not in values or values["fill_count"] != 0
                or "open_order_count" not in values or values["open_order_count"] != 0):
            reasons.append("Дневной капитал, PnL и происхождение позиции не подтверждены; риск не рассчитан")
        else:
            try:
                capital = _decimal(evidence.day_start_capital_usdt, positive=True)
                realized = _decimal(evidence.realized_pnl_today_usdt)
                unrealized = _decimal(evidence.unrealized_pnl_usdt)
                cash_flow = _decimal(evidence.net_external_cash_flow_usdt)
                if "portfolio_value_usdt" not in values or "fees_by_currency" not in values:
                    raise ValueError("Неполный счёт")
                if capital + realized + unrealized + cash_flow != values["portfolio_value_usdt"]:
                    raise ValueError("PnL и движения средств не сходятся с оценкой BTC+USDT")
                loss = max(Decimal(0), -realized) + max(Decimal(0), -unrealized)
                limit = capital * self.risk_manager.settings.daily_loss_limit_pct
                values.update(confirmed_realized_pnl_usdt=realized,
                              confirmed_unrealized_pnl_usdt=unrealized,
                              daily_loss_usdt=loss, daily_loss_limit_usdt=limit,
                              daily_risk_remaining_usdt=max(Decimal(0), limit - loss))
                if loss >= limit:
                    reasons.append("Достигнут дневной лимит потерь")
            except (ValueError, TypeError, InvalidOperation):
                reasons.append("Подтверждённый PnL или дневной риск противоречивы")
        return ReconcileReport(ReconcileStatus.BLOCKED if reasons else ReconcileStatus.READY,
                               tuple(reasons), **values)
