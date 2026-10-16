"""Read-only first Demo trade preflight with an explicit, non-executing review gate.

No order transport, payload builder, reservation, or HALTED reset is exposed.
A manual phrase acknowledges only this inspection. In particular, it cannot
clear the unproved atomic protection of partial BTC-USDT SPOT fills.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from bot.demo_client import DemoClient
from bot.execution_engine import ExecutionEngine
from bot.market_data import MarketSnapshot
from bot.order_preview import OrderPreview, PROTECTION_BLOCK, preview_demo_buy
from bot.reconcile import ReconcileReport, ReconcileStatus
from bot.strategy import Signal, StrategyResult, evaluate_signal


REVIEW_PHRASE = "REVIEW FIRST OKX DEMO BTC-USDT TRADE"


@dataclass(frozen=True)
class FirstTradeReport:
    status: str
    manual_review_confirmed: bool
    signal: StrategyResult | None
    reconciliation: ReconcileReport | None
    preview: OrderPreview | None
    stages: tuple[tuple[str, str], ...]
    reasons: tuple[str, ...]

    @property
    def exit_code(self) -> int:
        return 1  # Deliberately never a trade authorization.


class FirstTradePreflight:
    """Wire the existing GET/EMA/reconciliation/risk preview, then stop at BUY."""

    def __init__(
        self, *, demo_client: DemoClient, engine: ExecutionEngine,
        market_provider: Callable[[], MarketSnapshot],
        reconciliation_provider: Callable[[MarketSnapshot], ReconcileReport],
        signal_provider: Callable[[tuple], StrategyResult] = evaluate_signal,
        preview_provider: Callable[..., OrderPreview] = preview_demo_buy,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.demo_client = demo_client
        self.engine = engine
        self.market_provider = market_provider
        self.reconciliation_provider = reconciliation_provider
        self.signal_provider = signal_provider
        self.preview_provider = preview_provider
        self.clock = clock

    def run_once(self, *, confirmation: str | None = None) -> FirstTradeReport:
        stages = [
            ("EMA", "NOT_CHECKED"), ("RISK", "NOT_CHECKED"),
            ("BUY", "POST_DISABLED"), ("BUY_FILLS", "NOT_APPLICABLE"),
            ("SL_TP", "UNVERIFIED"), ("POSITION", "NOT_OPENED"),
            ("CLOSE", "POST_DISABLED"), ("PNL", "NOT_CONFIRMED"),
        ]
        reasons = [PROTECTION_BLOCK,
                   "Нельзя доказать своевременное покрытие каждой частичной покупки SL/TP; торговые POST отключены"]
        if confirmation is not None and confirmation != REVIEW_PHRASE:
            reasons.append("Неверная фраза ручного подтверждения; данные не запрашивались")
            return FirstTradeReport("BLOCK", False, None, None, None, tuple(stages), tuple(reasons))
        confirmed = confirmation == REVIEW_PHRASE
        if not isinstance(self.demo_client, DemoClient) or self.demo_client.config.mode != "DEMO":
            reasons.append("Требуется отдельный OKX Demo-клиент с режимом DEMO")
            return FirstTradeReport("BLOCK", confirmed, None, None, None, tuple(stages), tuple(reasons))
        try:
            diagnostic = self.engine.diagnose()
            if diagnostic.blocked or diagnostic.halted or diagnostic.intents:
                reasons.append("SQLite исполнителя содержит блокировку, HALTED или прежнее намерение")
                return FirstTradeReport("BLOCK", confirmed, None, None, None, tuple(stages), tuple(reasons))
            market = self.market_provider()
            if (not isinstance(market, MarketSnapshot)
                    or market.instrument.instrument_id != "BTC-USDT"
                    or market.instrument.base_currency != "BTC"
                    or market.instrument.quote_currency != "USDT"
                    or len(market.candles) != 100):
                raise ValueError("Рыночный снимок неизвестен")
            signal = self.signal_provider(market.candles)
            if not isinstance(signal, StrategyResult) or not isinstance(signal.signal, Signal):
                raise ValueError("Сигнал неизвестен")
            stages[0] = ("EMA", signal.signal.value)
            reconciliation = self.reconciliation_provider(market)
            if not isinstance(reconciliation, ReconcileReport):
                raise ValueError("Сверка Demo неизвестна")
            preview = self.preview_provider(market=market, signal=signal,
                                            reconciliation=reconciliation, engine=self.engine,
                                            now_utc=self.clock())
            if not isinstance(preview, OrderPreview) or preview.status != "BLOCK":
                raise ValueError("Предварительный план нарушил защитный запрет")
            stages[1] = ("RISK", "PREVIEW_ONLY" if preview.quantity_btc is not None else "BLOCKED")
            if signal.signal != Signal.BUY:
                reasons.append("Закрытые EMA-свечи не дали нового BUY; SELL внешнего BTC запрещён")
            if reconciliation.status != ReconcileStatus.READY:
                reasons.append("Полная сверка виртуального бюджета 5000 USDT отсутствует")
            reasons.extend(preview.reasons)
            if confirmed:
                reasons.append("Ручная фраза подтвердила только просмотр; разрешение на исполнение не выдано")
            return FirstTradeReport("BLOCK", confirmed, signal, reconciliation, preview,
                                    tuple(stages), tuple(dict.fromkeys(reasons)))
        except Exception:
            # Transport and API exceptions may include sensitive request details.
            reasons.append("Публичные/Demo GET, EMA, сверка или предварительная проверка риска недоступны")
            return FirstTradeReport("BLOCK", confirmed, None, None, None,
                                    tuple(stages), tuple(reasons))
