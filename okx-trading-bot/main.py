"""One-shot public checks and a strictly read-only private Demo check."""

from __future__ import annotations

import argparse
import sys

from bot.demo_client import DemoClient, DemoError
from bot.dry_run import DryRunOrchestrator
from bot.execution_engine import ExecutionEngine
from bot.journal import DecisionJournal, JournalError
from bot.market_data import get_market_snapshot
from bot.okx_client import OkxClient, OkxError
from bot.order_preview import preview_demo_buy
from bot.protection_capabilities import inspect_protection_capabilities
from bot.risk_manager import RiskManager
from bot.reconcile import DemoReconciler
from bot.budget_setup import BudgetInitializer
from bot.strategy_budget import BudgetError, CONFIRMATION_PHRASE, StrategyBudgetStore
from bot.strategy import StrategyError, evaluate_signal
from config import Config, ConfigError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only OKX BTC-USDT market data and EMA signal")
    commands = parser.add_mutually_exclusive_group(required=True)
    commands.add_argument("--check", action="store_true", help="Fetch public market data once")
    commands.add_argument("--signal", action="store_true", help="Calculate the EMA 9/21 signal once")
    commands.add_argument("--demo-check", action="store_true", help="Read the virtual OKX Demo account once")
    commands.add_argument("--dry-run", action="store_true", help="Record one non-executing Demo analysis cycle")
    commands.add_argument("--reconcile", action="store_true", help="Read-only Demo/Risk Manager reconciliation")
    commands.add_argument("--budget-preview", action="store_true", help="Inspect the one-time 5000 USDT Demo allocation")
    commands.add_argument("--confirm-budget", metavar="PHRASE", help="Confirm the allocation with the exact phrase")
    commands.add_argument("--execution-check", action="store_true", help="Offline inspection of local execution state")
    commands.add_argument("--order-preview", action="store_true", help="Read-only, blocked Demo LIMIT BUY preview")
    commands.add_argument("--protection-capabilities", action="store_true", help="GET-only Demo attached TP/SL capability inventory")
    parser.add_argument("--budget-token", metavar="TOKEN", help="Fresh token from --budget-preview")
    args = parser.parse_args(argv)
    if args.budget_token and args.confirm_budget is None:
        parser.error("--budget-token используется только с --confirm-budget")

    if args.execution_check:
        report = ExecutionEngine().diagnose()
        print(f"OKX Demo BTC-USDT SPOT — локальное состояние исполнителя: {'BLOCKED' if report.blocked else 'CLEAR'}")
        print(f"Причина: {report.reason}")
        print(f"Аварийная остановка: {'да' if report.halted else 'нет'}; намерений: {len(report.intents)}")
        for intent in report.intents:
            print(f"clientOrderId: {intent.client_order_id}; состояние: {intent.state.value}; "
                  f"объём: {intent.quantity} BTC; исполнено: {intent.filled_quantity} BTC; "
                  f"BTC после комиссии: {intent.owned_btc}")
        print("Команда не подключается к OKX, не разрешает отправку и не создаёт ордера")
        return 1 if report.blocked else 0

    if args.protection_capabilities:
        try:
            demo_client = DemoClient.from_env_file()
        except (ConfigError, DemoError):
            print("Возможности защиты: BLOCK — локальный Demo Read-only ключ или регион не настроен", file=sys.stderr)
            return 1
        try:
            report = inspect_protection_capabilities(
                public_client=OkxClient(demo_client.config), demo_client=demo_client,
                engine=ExecutionEngine(journal=DecisionJournal()),
            )
        except Exception:
            print("Возможности защиты: BLOCK — GET-диагностика недоступна или противоречива", file=sys.stderr)
            return 1
        print(f"OKX Demo BTC-USDT SPOT attached TP/SL: {report.status}")
        for item in report.sdk_evidence:
            print(f"Официальный SDK (описание API): {item}")
        for item in report.api_observations:
            print(f"Наблюдение GET API/SQLite: {item}")
        for item in report.unknowns:
            print(f"Не подтверждено: {item}")
        print("GET-диагностика не записывала SQLite и не отправляла торговые запросы")
        return 1

    if args.order_preview:
        try:
            demo_client = DemoClient.from_env_file()
            market = get_market_snapshot(OkxClient(demo_client.config))
            signal = evaluate_signal(market.candles)
            journal = DecisionJournal()
            risk_manager = RiskManager()
            store = StrategyBudgetStore(journal.path)
            report = DemoReconciler(market_provider=lambda: market, demo_client=demo_client,
                                    journal=journal, risk_manager=risk_manager,
                                    budget_store=store).run_once()
            preview = preview_demo_buy(market=market, signal=signal, reconciliation=report,
                                       engine=ExecutionEngine(journal=journal,
                                                              risk_manager=risk_manager,
                                                              budget_store=store))
        except (ConfigError, DemoError, OkxError, StrategyError, ValueError):
            print("OKX Demo BTC-USDT LIMIT BUY: BLOCK — данные для предварительного плана недоступны", file=sys.stderr)
            return 1
        print(f"OKX Demo BTC-USDT LIMIT BUY — предварительный план: {preview.status}")
        print(f"Сигнал EMA: {preview.signal or 'неизвестен'}")
        if preview.candle_closed_at is not None:
            print(f"Закрытая свеча: {preview.candle_closed_at:%Y-%m-%d %H:%M:%S} UTC")
        if preview.reference_price_usdt is not None:
            print(f"Справочная цена BTC: {preview.reference_price_usdt} USDT")
        if preview.estimated_entry_usdt is not None:
            print(f"Оценка LIMIT-входа: {preview.estimated_entry_usdt}; "
                  f"SL 1%: {preview.estimated_stop_usdt}; TP 2%: {preview.estimated_take_usdt} USDT")
        if preview.virtual_usdt_available is not None:
            print(f"Виртуальный доступный USDT: {preview.virtual_usdt_available}")
        if preview.external_btc is not None:
            print(f"Внешний BTC: {preview.external_btc}; продажа запрещена")
        if report.per_trade_risk_limit_usdt is not None:
            print(f"Лимит Risk Manager на сделку: {report.per_trade_risk_limit_usdt} USDT; "
                  f"дневной остаток: {report.daily_risk_remaining_usdt} USDT")
        if preview.quantity_btc is not None:
            print(f"Гипотетический размер: {preview.quantity_btc} BTC; "
                  f"риск {preview.estimated_risk_usdt} USDT; стоимость {preview.estimated_cost_usdt} USDT")
        for reason in preview.reasons:
            print(f"Причина BLOCK: {reason}")
        print("Ордер и защитные Algo-ордера не создавались; SQLite и лимиты не изменялись")
        return 1

    if args.budget_preview or args.confirm_budget is not None:
        try:
            demo_client = DemoClient.from_env_file()
        except (ConfigError, DemoError):
            print("Бюджет: BLOCKED — локальный Demo-режим или Read-only API-ключ недоступен", file=sys.stderr)
            return 1
        journal = DecisionJournal()
        initializer = BudgetInitializer(
            market_provider=lambda: get_market_snapshot(OkxClient(demo_client.config)),
            demo_client=demo_client, journal=journal, risk_manager=RiskManager(),
            store=StrategyBudgetStore(journal.path),
        )
        if args.budget_preview:
            preview = initializer.preview()
            print(f"Стартовый бюджет OKX Demo BTC-USDT: {'ГОТОВ К ПОДТВЕРЖДЕНИЮ' if preview.ready_to_confirm else 'BLOCKED'}")
            report = preview.report
            if report.usdt_total is not None and report.btc_total is not None:
                print(f"Виртуальный счёт: USDT {report.usdt_total} (доступно {report.usdt_available}), "
                      f"внешний BTC {report.btc_total}")
            print("Выделяемый капитал стратегии: 5000 USDT; внешний BTC в него не входит")
            if preview.ready_to_confirm:
                print("Бюджет ещё не записан. Для отдельного явного подтверждения выполните:")
                print(f'.\\.venv\\Scripts\\python.exe main.py --confirm-budget "{CONFIRMATION_PHRASE}" '
                      f'--budget-token {preview.token}')
                print("Токен действителен в текущем пятиминутном окне UTC при неизменных данных Demo и журнала.")
            else:
                for reason in preview.reasons:
                    print(f"Причина BLOCKED: {reason}")
            return 0 if preview.ready_to_confirm else 1
        try:
            baseline = initializer.confirm(args.confirm_budget, args.budget_token or "")
        except BudgetError as exc:
            print(f"Бюджет: BLOCKED — {exc}", file=sys.stderr)
            return 1
        print("Стартовый бюджет OKX Demo BTC-USDT подтверждён один раз")
        print(f"Капитал стратегии: {baseline.strategy_capital_usdt} USDT; внешний BTC: {baseline.demo_btc_total}")
        print(f"Время UTC: {baseline.confirmed_at_utc:%Y-%m-%d %H:%M:%S}; ордера не отправлялись")
        return 0

    if args.reconcile:
        try:
            demo_client = DemoClient.from_env_file()
        except (ConfigError, DemoError):
            print("Сверка: BLOCKED — локальный Demo-режим или Read-only API-ключ недоступен", file=sys.stderr)
            return 1
        journal = DecisionJournal()
        report = DemoReconciler(
            market_provider=lambda: get_market_snapshot(OkxClient(demo_client.config)),
            demo_client=demo_client, journal=journal, risk_manager=RiskManager(),
            budget_store=StrategyBudgetStore(journal.path),
        ).run_once()
        print(f"Сверка OKX Demo BTC-USDT SPOT: {report.status.value}")
        if report.btc_total is not None and report.usdt_total is not None:
            print(f"Виртуальный баланс: BTC {report.btc_total} (доступно {report.btc_available}), "
                  f"USDT {report.usdt_total} (доступно {report.usdt_available})")
        if report.portfolio_value_usdt is not None:
            print(f"Оценка BTC+USDT: {report.portfolio_value_usdt} USDT при цене BTC {report.btc_price_usdt}")
        if report.external_btc is not None:
            print(f"Внешний BTC: {report.external_btc}; учтённая позиция бота: 0 BTC; продажа запрещена")
        if report.strategy_capital_usdt is not None:
            print(f"Капитал стратегии: {report.strategy_capital_usdt} USDT "
                  f"(подтверждён {report.strategy_start_utc:%Y-%m-%d %H:%M:%S} UTC); "
                  f"текущая подтверждённая стоимость: {report.strategy_equity_usdt if report.strategy_equity_usdt is not None else 'неизвестна'}")
            if report.per_trade_risk_limit_usdt is not None:
                print(f"Лимит риска на сделку: {report.per_trade_risk_limit_usdt} USDT")
        if report.open_order_count is not None and report.fill_count is not None:
            print(f"Открытых SPOT-ордеров: {report.open_order_count}; исполнений в ответе: {report.fill_count}")
        if report.fees_by_currency is not None:
            print("Комиссии исполнений в ответе: " + ", ".join(f"{ccy} {fee}" for ccy, fee in report.fees_by_currency))
        if report.maker_fee_rate is not None:
            print(f"Ставки OKX (знак как в API): maker {report.maker_fee_rate}, taker {report.taker_fee_rate}")
        if report.instrument is not None:
            instrument = report.instrument
            print(f"Ограничения BTC-USDT: min {instrument.min_size}, шаг {instrument.lot_size}, "
                  f"тик {instrument.tick_size}, maxLmtSz {instrument.max_limit_size}, "
                  f"maxMktSz {instrument.max_market_size} (единицы полей OKX)")
        if report.confirmed_realized_pnl_usdt is not None:
            print(f"Подтверждённый PnL: реализованный {report.confirmed_realized_pnl_usdt}, "
                  f"нереализованный {report.confirmed_unrealized_pnl_usdt} USDT")
            print(f"Дневной риск: убыток {report.daily_loss_usdt}, предел {report.daily_loss_limit_usdt}, "
                  f"остаток {report.daily_risk_remaining_usdt} USDT")
        else:
            if report.strategy_capital_usdt is None:
                print("Подтверждённый PnL и дневной риск: неизвестны; дневная база не устанавливалась")
            else:
                print("Подтверждённый PnL и дневной риск: неизвестны; бюджет записан, но сверка заблокирована")
        for reason in report.reasons:
            print(f"Причина BLOCKED: {reason}")
        return 0 if report.status.value == "READY" else 1

    if args.dry_run:
        demo_client: DemoClient | None = None

        def get_demo_client() -> DemoClient:
            nonlocal demo_client
            if demo_client is None:
                demo_client = DemoClient.from_env_file()
            return demo_client

        orchestrator = DryRunOrchestrator(
            market_provider=lambda: get_market_snapshot(OkxClient(get_demo_client().config)),
            demo_provider=lambda: get_demo_client().check_account(),
            risk_manager=RiskManager(),
            journal=DecisionJournal(),
        )
        try:
            record = orchestrator.run_once()
        except (JournalError, ValueError):
            print("Ошибка dry-run: решение не удалось безопасно записать в журнал", file=sys.stderr)
            return 1
        decision = record.draft
        print("OKX Demo Trading — однократный анализ без исполнения")
        print(f"Решение: {decision.status.value}")
        print(f"Сигнал EMA: {decision.signal or 'неизвестен'}")
        if decision.candle_closed_at is not None:
            print(f"Закрытая свеча: {decision.candle_closed_at:%Y-%m-%d %H:%M:%S} UTC")
        if decision.ema9 is not None and decision.ema21 is not None:
            print(f"EMA9: {decision.ema9:.8f}; EMA21: {decision.ema21:.8f}")
        if decision.virtual_btc_available is not None and decision.virtual_usdt_available is not None:
            print(f"Виртуальный баланс доступно: BTC {decision.virtual_btc_available}, USDT {decision.virtual_usdt_available}")
        print(f"Причина: {decision.reason}")
        if decision.quantity is not None:
            print(f"Гипотетический размер: {decision.quantity} BTC")
        if decision.entry_price is not None:
            print(f"Цена входа: {decision.entry_price}; стоп: {decision.stop_price}; тейк: {decision.take_profit_price} USDT")
        if record.reference_id is not None:
            print(f"Повторная свеча; исходное решение: {record.reference_id}")
        print(f"ID записи: {record.id}; executed=false")
        return 0

    if args.demo_check:
        try:
            summary = DemoClient.from_env_file().check_account()
        except (ConfigError, DemoError) as exc:
            print(f"Ошибка проверки Demo-счёта: {exc}", file=sys.stderr)
            return 1
        print("OKX Demo Trading — виртуальный счёт (только чтение)")
        print(f"BTC: всего {summary.btc.total}, доступно {summary.btc.available}")
        print(f"USDT: всего {summary.usdt.total}, доступно {summary.usdt.available}")
        print(f"Открытых SPOT-ордеров: {summary.open_order_count}")
        print(f"Недавних SPOT-исполнений (до 100): {summary.recent_fill_count}")
        if summary.usdt.available == 0:
            print("Доступных виртуальных USDT в Demo сейчас нет; проверка чтения выполнена.")
        return 0

    try:
        config = Config.from_environment()
        snapshot = get_market_snapshot(OkxClient(config))
        result = evaluate_signal(snapshot.candles) if args.signal else None
    except (ConfigError, OkxError, StrategyError) as exc:
        print(f"Ошибка {('сигнала' if args.signal else 'проверки OKX')}: {exc}", file=sys.stderr)
        return 1

    if args.check:
        print(f"Инструмент: {snapshot.instrument.instrument_id}")
        print(f"Текущая цена: {snapshot.last_price} USDT")
        print(
            "Время последней закрытой свечи: "
            f"{snapshot.latest_closed_candle.closed_at:%Y-%m-%d %H:%M:%S} UTC"
        )
        print(f"Количество закрытых свечей: {len(snapshot.candles)}")
    else:
        assert result is not None
        print(f"Инструмент: {snapshot.instrument.instrument_id}")
        print(f"EMA9: {result.ema9:.8f}")
        print(f"EMA21: {result.ema21:.8f}")
        print(f"Сигнал: {result.signal.value}")
        print(f"Время закрытия свечи: {result.candle_closed_at:%Y-%m-%d %H:%M:%S} UTC")
        print(f"Причина: {result.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
