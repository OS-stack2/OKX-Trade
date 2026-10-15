"""Two-step, one-time owner confirmation for existing OKX Demo USDT."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
from pathlib import Path

from bot.demo_client import DemoClient
from bot.journal import DecisionJournal, JournalError
from bot.market_data import MarketSnapshot
from bot.reconcile import DemoReconciler, ReconcileReport
from bot.risk_manager import AccountSnapshot, RiskManager, RiskSettings, RiskStateError
from bot.strategy_budget import (
    BudgetBaseline, BudgetError, CONFIRMATION_PHRASE, STRATEGY_BUDGET_USDT,
    StrategyBudgetStore,
)


_EXPECTED_PREFLIGHT_REASONS = frozenset({
    "Подтверждённая дневная база Risk Manager отсутствует или недоступна",
    "Дневной капитал, PnL и происхождение позиции не подтверждены; риск не рассчитан",
})


@dataclass(frozen=True)
class BudgetPreview:
    ready_to_confirm: bool
    reasons: tuple[str, ...]
    report: ReconcileReport
    token: str | None = None
    baseline: BudgetBaseline | None = None


class BudgetInitializer:
    def __init__(
        self, *, market_provider: Callable[[], MarketSnapshot], demo_client: DemoClient,
        journal: DecisionJournal, risk_manager: RiskManager, store: StrategyBudgetStore,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.market_provider = market_provider
        self.demo_client = demo_client
        self.journal = journal
        self.risk_manager = risk_manager
        self.store = store
        self.clock = clock

    def preview(self) -> BudgetPreview:
        """Read only: do not create a budget, risk day, or journal decision."""
        started_at = self.clock()
        report = DemoReconciler(
            market_provider=self.market_provider, demo_client=self.demo_client,
            journal=self.journal, risk_manager=self.risk_manager, clock=self.clock,
        ).run_once()
        now = self.clock()
        reasons = [reason for reason in report.reasons if reason not in _EXPECTED_PREFLIGHT_REASONS]
        if (not isinstance(started_at, datetime) or not isinstance(now, datetime)
                or started_at.tzinfo is None or started_at.utcoffset() != timedelta(0)
                or now.tzinfo is None or now.utcoffset() != timedelta(0)
                or now < started_at or now - started_at > timedelta(seconds=60)
                or now.date() != started_at.date()):
            reasons.append("Снимок пересёк границу UTC-суток или устарел")
        try:
            audit = self.journal.audit_read_only()
            if audit.executed_count:
                reasons.append("Журнал содержит исполнение без сверенной позиции")
        except JournalError:
            audit = None
            reasons.append("SQLite-журнал решений недоступен")
        try:
            existing = self.store.read()
            if existing is not None:
                reasons.append("Стартовый бюджет уже подтверждён; повторное подтверждение запрещено")
        except BudgetError:
            reasons.append("Запись стартового бюджета неизвестна или повреждена")
        try:
            self.risk_manager.validate_settings()
            if self.risk_manager.settings != RiskSettings():
                reasons.append("Настройки Risk Manager отличаются от утверждённых лимитов")
        except RiskStateError:
            reasons.append("Настройки Risk Manager некорректны")
        if (self.risk_manager.state_path.exists()
                or Path(f"{self.risk_manager.state_path}.lock").exists()):
            reasons.append("Дневное состояние Risk Manager уже существует или заблокировано")
        if report.btc_total != Decimal("1") or report.btc_available != Decimal("1"):
            reasons.append("Внешний 1 BTC не подтверждён или зарезервирован")
        if (report.usdt_total is None or report.usdt_available is None
                or report.usdt_total != report.usdt_available
                or report.usdt_available < STRATEGY_BUDGET_USDT):
            reasons.append("Недостаточно свободного Demo USDT или есть неизвестный резерв")
        if report.open_order_count != 0 or report.fill_count != 0 or report.fees_by_currency is None:
            reasons.append("Открытые ордера или история исполнений не позволяют подтвердить чистый старт")
        if report.instrument is None or report.btc_price_usdt is None or report.taker_fee_rate is None:
            reasons.append("Цена, комиссия или ограничения BTC-USDT неизвестны")
        else:
            try:
                instrument = report.instrument
                minimum = (instrument.min_size / instrument.lot_size).to_integral_value(rounding=ROUND_CEILING) * instrument.lot_size
                minimum_cost = minimum * report.btc_price_usdt * (1 + max(Decimal(0), -report.taker_fee_rate))
                if (instrument.max_limit_size < instrument.min_size
                        or minimum_cost >= STRATEGY_BUDGET_USDT):
                    reasons.append("Бюджет не покрывает минимальный размер и комиссию BTC-USDT")
            except (ArithmeticError, TypeError, ValueError):
                reasons.append("Невозможно проверить минимальный размер и комиссию")
        if reasons or audit is None:
            return BudgetPreview(False, tuple(dict.fromkeys(reasons)), report)
        assert report.portfolio_value_usdt is not None
        assert report.maker_fee_rate is not None and report.taker_fee_rate is not None
        assert report.btc_total is not None and report.btc_available is not None
        assert report.usdt_total is not None and report.usdt_available is not None
        baseline = BudgetBaseline(
            confirmed_at_utc=now, strategy_capital_usdt=STRATEGY_BUDGET_USDT,
            demo_btc_total=report.btc_total, demo_btc_available=report.btc_available,
            demo_usdt_total=report.usdt_total, demo_usdt_available=report.usdt_available,
            btc_price_usdt=report.btc_price_usdt,
            portfolio_btc_usdt_value=report.portfolio_value_usdt,
            maker_fee_rate=report.maker_fee_rate, taker_fee_rate=report.taker_fee_rate,
            journal_decision_count=audit.decision_count,
        )
        # The challenge binds the inspected balances, limits and fees to the
        # current UTC five-minute window. Price is deliberately omitted because
        # confirmation fetches a fresh quote and does not execute a trade.
        instrument = report.instrument
        challenge = {
            "window": int(now.timestamp()) // 300, "day": now.date().isoformat(),
            "budget": str(STRATEGY_BUDGET_USDT), "btc": str(report.btc_total),
            "btc_available": str(report.btc_available), "usdt": str(report.usdt_total),
            "usdt_available": str(report.usdt_available), "maker": str(report.maker_fee_rate),
            "taker": str(report.taker_fee_rate), "journal_count": audit.decision_count,
            "instrument": [str(getattr(instrument, name)) for name in (
                "min_size", "lot_size", "tick_size", "max_limit_size", "max_market_size",
            )],
        }
        token = hashlib.sha256(json.dumps(challenge, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return BudgetPreview(True, (), report, token, baseline)

    def confirm(self, phrase: str, token: str) -> BudgetBaseline:
        if phrase != CONFIRMATION_PHRASE or not isinstance(token, str) or len(token) != 16:
            raise BudgetError("Требуется точная фраза подтверждения и токен свежего просмотра")
        preview = self.preview()  # Re-fetch all evidence; never trust the old preview alone.
        if not preview.ready_to_confirm or preview.token != token or preview.baseline is None:
            raise BudgetError("Данные Demo изменились или проверка заблокирована; повторите просмотр")
        baseline = preview.baseline
        self.store.confirm_once(baseline)
        # If this second durable write fails, the saved allocation remains
        # fail-closed; a repeat confirmation cannot reset a prior risk state.
        account = AccountSnapshot(
            as_of_utc=baseline.confirmed_at_utc, day_utc=baseline.confirmed_at_utc.date(),
            equity_usdt=STRATEGY_BUDGET_USDT,
            day_start_capital_usdt=STRATEGY_BUDGET_USDT,
            available_usdt=STRATEGY_BUDGET_USDT,
            realized_pnl_today_usdt=Decimal(0), unrealized_pnl_usdt=Decimal(0),
            open_positions=0, pending_orders=0, entries_today=0,
        )
        try:
            self.risk_manager.initialize_day(account, self.clock())
        except (RiskStateError, OSError, ValueError) as exc:
            raise BudgetError("Бюджет записан, но состояние риска не подтверждено; входы заблокированы") from exc
        return baseline
