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
from bot.market_data import (MarketSnapshot, PublicQuoteEvidence, SpotInstrument,
                             assess_public_quote)
from bot.risk_manager import RiskManager, RiskSettings, RiskStateError
from bot.strategy_budget import BudgetError, StrategyBudgetStore, STRATEGY_BUDGET_USDT


class ReconcileStatus(StrEnum):
    READY = "READY"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class ReconcileReport:
    status: ReconcileStatus
    reasons: tuple[str, ...]
    checked_at_utc: datetime | None = None
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
    strategy_capital_usdt: Decimal | None = None
    strategy_equity_usdt: Decimal | None = None
    strategy_start_utc: datetime | None = None
    per_trade_risk_limit_usdt: Decimal | None = None
    diagnostic_checks: tuple[tuple[str, str], ...] = ()


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
        self, *, market_provider: Callable[[], MarketSnapshot | PublicQuoteEvidence], demo_client: DemoClient,
        journal: DecisionJournal, risk_manager: RiskManager,
        budget_store: StrategyBudgetStore | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.market_provider = market_provider
        self.demo_client = demo_client
        self.journal = journal
        self.risk_manager = risk_manager
        self.budget_store = budget_store
        self.clock = clock

    def run_once(self) -> ReconcileReport:
        """Never initializes the UTC day, reserves risk, or sends a private write."""
        now = self.clock()
        if not _fresh(now, now, timedelta(0)):
            return ReconcileReport(ReconcileStatus.BLOCKED, ("Текущее время UTC неизвестно",))
        reasons: list[str] = []
        values: dict = {}
        checks: list[tuple[str, str]] = []
        try:
            self.risk_manager.validate_settings()
            checks.append(("RISK_SETTINGS", "OK"))
        except (RiskStateError, OSError, ValueError):
            checks.append(("RISK_SETTINGS", "BLOCKED"))
            reasons.append("RISK_SETTINGS: настройки Risk Manager недоступны либо противоречивы")
        try:
            audit = self.journal.audit_read_only()
            if audit.executed_count:
                checks.append(("JOURNAL", "BLOCKED"))
                reasons.append("Журнал содержит исполнение без проверенного происхождения")
            else:
                checks.append(("JOURNAL", "OK"))
        except (JournalError, OSError, ValueError):
            checks.append(("JOURNAL", "BLOCKED"))
            reasons.append("JOURNAL: Журнал решений недоступен либо противоречив")
        try:
            market = self.market_provider()
            assessment = assess_public_quote(market, self.clock(), max_ticker_age=MAX_TICKER_AGE)
        except Exception:
            checks.extend((name, "BLOCKED") for name in ("PRICE", "QUOTE_TIME", "INSTRUMENT", "LIMITS"))
            reasons.append("MARKET: поставщик публичных GET-данных недоступен")
        else:
            checks.extend(assessment.checks)
            reasons.extend(assessment.reasons)
            if assessment.instrument is not None:
                values["instrument"] = assessment.instrument
            if assessment.price is not None:
                values["btc_price_usdt"] = assessment.price
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
            for name, rate in (("maker", rates.maker), ("taker", rates.taker)):
                if not isinstance(rate, Decimal) or not rate.is_finite() or not -1 < rate < 1:
                    raise ValueError(name)
            values.update(maker_fee_rate=rates.maker, taker_fee_rate=rates.taker)
            checks.append(("FEES", "OK"))
        except ValueError as exc:
            checks.append(("FEES", "BLOCKED"))
            field = str(exc) if str(exc) in ("maker", "taker") else "maker/taker"
            reasons.append(f"FEES: ставка {field} отсутствует или некорректна")
        except Exception:
            checks.append(("FEES", "BLOCKED"))
            reasons.append("FEES: Demo GET ставок комиссии недоступен")

        # The existing journal records hypothetical decisions only, never fills.
        # Until a verified ownership ledger exists, all BTC is treated as external.
        if "btc_total" in values:
            values["external_btc"] = values["btc_total"]
            values["bot_btc"] = Decimal(0)
        try:
            state = self.risk_manager.inspect_daily_state(self.clock())
            if state.pending_entry or state.loss_locked or state.entries_used:
                checks.append(("RISK_STATE", "BLOCKED"))
                reasons.append("Дневное состояние Risk Manager содержит резерв, блокировку или вход без сверки")
            else:
                checks.append(("RISK_STATE", "OK"))
        except (RiskStateError, OSError, ValueError):
            state = None
            try:
                self.risk_manager.state_path.stat()
                missing_state = False
            except FileNotFoundError:
                missing_state = True
            except OSError:
                missing_state = False
            checks.append(("RISK_STATE", "UNINITIALIZED" if missing_state else "BLOCKED"))
            reasons.append("Подтверждённая дневная база Risk Manager отсутствует или недоступна")

        if self.budget_store is not None:
            try:
                baseline = self.budget_store.read()
            except BudgetError:
                baseline = None
                reasons.append("Запись бюджета стратегии повреждена или недоступна")
            if baseline is not None:
                values.update(strategy_capital_usdt=baseline.strategy_capital_usdt,
                              strategy_start_utc=baseline.confirmed_at_utc)
                if self.risk_manager.settings != RiskSettings():
                    reasons.append("Настройки Risk Manager отличаются от утверждённых лимитов")
                if (baseline.strategy_capital_usdt != STRATEGY_BUDGET_USDT
                        or "btc_total" not in values or "usdt_total" not in values
                        or values["btc_total"] != baseline.demo_btc_total
                        or values["btc_available"] != baseline.demo_btc_available
                        or values["usdt_total"] != baseline.demo_usdt_total
                        or values["usdt_available"] != baseline.demo_usdt_available):
                    reasons.append("Баланс Demo изменился относительно подтверждённого бюджета; движение средств не объяснено")
                if (state is None or state.start_capital_usdt != baseline.strategy_capital_usdt
                        or state.day_utc != self.clock().date()
                        or baseline.confirmed_at_utc > self.clock()
                        or baseline.confirmed_at_utc.date() != state.day_utc):
                    reasons.append("Дневная база Risk Manager не совпадает с бюджетом стратегии")
                if (values.get("fill_count") != 0 or values.get("open_order_count") != 0
                        or "fees_by_currency" not in values):
                    reasons.append("Нет подтверждения отсутствия исполнений и чужих ордеров")
                if not reasons:
                    zero = Decimal(0)
                    limit = baseline.strategy_capital_usdt * self.risk_manager.settings.daily_loss_limit_pct
                    values.update(strategy_equity_usdt=baseline.strategy_capital_usdt,
                                  confirmed_realized_pnl_usdt=zero,
                                  confirmed_unrealized_pnl_usdt=zero,
                                  daily_loss_usdt=zero, daily_loss_limit_usdt=limit,
                                  daily_risk_remaining_usdt=limit,
                                  per_trade_risk_limit_usdt=(baseline.strategy_capital_usdt
                                                              * self.risk_manager.settings.risk_per_trade))
                return ReconcileReport(ReconcileStatus.BLOCKED if reasons else ReconcileStatus.READY,
                                       tuple(reasons), checked_at_utc=self.clock(),
                                       diagnostic_checks=tuple(checks), **values)
        reasons.append("Дневной капитал, PnL и происхождение позиции не подтверждены; риск не рассчитан")
        return ReconcileReport(ReconcileStatus.BLOCKED if reasons else ReconcileStatus.READY,
                               tuple(reasons), checked_at_utc=self.clock(),
                               diagnostic_checks=tuple(checks), **values)
