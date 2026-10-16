"""One-shot public checks and a strictly read-only private Demo check."""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
from pathlib import Path
import sys

from bot.demo_client import DemoClient, DemoError
from bot.dry_run import DryRunOrchestrator
from bot.execution_engine import ExecutionEngine
from bot.execution_diagnostics import inspect_execution
from bot.server_sqlite_setup import SQLITE_CONFIRMATION_PHRASE, ServerSqliteSetup, SqliteSetupError
from bot.journal import DecisionJournal, JournalError
from bot.market_data import get_market_snapshot
from bot.okx_client import OkxClient, OkxError
from bot.order_preview import preview_demo_buy
from bot.protection_capabilities import inspect_protection_capabilities
from bot.protection_simulator import SimulationConfig, SimulationError, parse_price_path, run_sample_scenarios
from bot.recovery_manager import RecoveryManager
from bot.risk_manager import RiskManager
from bot.reconcile import DemoReconciler
from bot.budget_setup import BudgetInitializer
from bot.strategy_budget import BudgetError, CONFIRMATION_PHRASE, StrategyBudgetStore
from bot.strategy import StrategyError, evaluate_signal
from bot.telegram_alerts import TelegramAlerter
from bot.watchdog import Watchdog, WatchdogError, WatchdogStore, watchdog_path_from_env
from bot.health_monitor import HealthError, HealthMonitor, HealthStore, health_settings_from_env
from bot.first_trade import FirstTradePreflight, REVIEW_PHRASE
from bot.growth_research import (ResearchConfig, ResearchError, ResearchReport,
                                 load_candles_csv, okx_taker_debit_rate, research)
from bot.history_download import HistoryError, RESEARCH_BARS, download_history, save_history_csv
from bot.market_data import Candle, SpotInstrument
from config import Config, ConfigError


def _print_research_report(report: ResearchReport, historical_candles: tuple[Candle, ...],
                           minimum: Decimal, lot: Decimal, tick: Decimal, maximum: Decimal,
                           fee: Decimal, slippage: Decimal, metric) -> None:
    print(f"Данные: {report.split_indices[3]} закрытых 15m свечей; "
          f"TRAIN 0:{report.split_indices[1]}, VALIDATION {report.split_indices[1]}:{report.split_indices[2]}, "
          f"FINAL_TEST {report.split_indices[2]}:{report.split_indices[3]}")
    for label, first, last in (("TRAIN", 0, report.split_indices[1]),
                               ("VALIDATION", report.split_indices[1], report.split_indices[2]),
                               ("FINAL_TEST", report.split_indices[2], report.split_indices[3])):
        print(f"{label} UTC: {historical_candles[first].opened_at.isoformat()} — "
              f"{historical_candles[last - 1].closed_at.isoformat()}")
    print(f"Затраты (допущение): taker {fee}, проскальзывание {slippage}; "
          f"minSz {minimum} BTC, lotSz {lot} BTC, tickSz {tick} USDT, "
          f"maxMktSz {maximum} USDT стоимости рыночного BUY")
    print("Вариант | Период | Доходность % | Просадка % | Profit factor | Сделки | К Buy & Hold, п.п.")
    for item in (*report.training, *report.validation, report.final_test):
        print(f"{item.variant} | {item.period} | {metric(item.return_pct)} | "
              f"{metric(item.max_drawdown_pct)} | {metric(item.profit_factor)} | "
              f"{item.trades} | {metric(item.relative_buy_hold_pct)}")
    print(f"Выбор по TRAIN/VALIDATION: {report.selected_variant}; финальный период не использовался для выбора")
    for item in report.stresses:
        print(f"Стресс {item.name}: доходность {metric(item.return_pct)}%; "
              f"просадка {metric(item.max_drawdown_pct)}%; {item.note}")
    print(f"Рекомендация: {report.recommendation}")
    for warning in report.warnings:
        print(f"ПРЕДУПРЕЖДЕНИЕ: {warning}")


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
    commands.add_argument("--execution-diagnose", action="store_true", help="Read-only explanation of SQLite and protection blocks")
    commands.add_argument("--sqlite-prepare", action="store_true", help="Read-only preflight for a new server trading SQLite")
    commands.add_argument("--sqlite-init", metavar="PHRASE", help="Create a new Demo SQLite after explicit confirmation")
    commands.add_argument("--order-preview", action="store_true", help="Read-only, blocked Demo LIMIT BUY preview")
    commands.add_argument("--first-demo-trade", action="store_true", help="One controlled Demo trade preflight; execution remains blocked")
    commands.add_argument("--protection-capabilities", action="store_true", help="GET-only Demo attached TP/SL capability inventory")
    commands.add_argument("--protection-simulate", action="store_true", help="Offline BUY/SL/TP lifecycle scenarios in temporary SQLite")
    commands.add_argument("--recovery-check", action="store_true", help="Read-only Demo GET recovery and durable emergency halt")
    commands.add_argument("--telegram-test", action="store_true", help="Send one owner alert through the local Demo Telegram configuration")
    commands.add_argument("--watchdog-once", action="store_true", help="One Demo GET observation using a separate Watchdog journal")
    commands.add_argument("--health-once", action="store_true", help="Check Watchdog observation freshness without OKX requests")
    commands.add_argument("--growth-research", metavar="CSV", help="Offline 15m EMA strategy comparison from a local CSV")
    commands.add_argument("--download-history", action="store_true", help="Public GET-only BTC-USDT 15m history to CSV")
    parser.add_argument("--history-bars", type=int, default=RESEARCH_BARS, metavar="COUNT", help="Closed candles to download (default 17280)")
    parser.add_argument("--history-output", type=Path, default=Path("data/btc_usdt_15m.csv"), metavar="CSV")
    parser.add_argument("--overwrite-history", action="store_true", help="Explicitly replace an existing history CSV")
    parser.add_argument("--budget-token", metavar="TOKEN", help="Fresh token from --budget-preview")
    parser.add_argument("--sqlite-token", metavar="TOKEN", help="Fresh token from --sqlite-prepare")
    parser.add_argument("--first-demo-confirm", metavar="PHRASE", help="Explicit acknowledgement of the non-executing first Demo trade review")
    parser.add_argument("--sim-price-path", metavar="PRICES", help="Comma-separated hypothetical BTC prices, e.g. 80000,79200,78400")
    for name in ("min-sz", "lot-sz", "tick-sz", "max-mkt-sz", "taker-fee", "slippage"):
        parser.add_argument(f"--research-{name}", metavar="VALUE", help="Verified offline research assumption")
    parser.add_argument("--research-okx-taker-rate", metavar="SIGNED_RATE",
                        help="Signed OKX taker rate; negative debit becomes positive research cost")
    args = parser.parse_args(argv)
    if args.budget_token and args.confirm_budget is None:
        parser.error("--budget-token используется только с --confirm-budget")
    if args.sqlite_token and args.sqlite_init is None:
        parser.error("--sqlite-token используется только с --sqlite-init")
    if args.first_demo_confirm is not None and not args.first_demo_trade:
        parser.error("--first-demo-confirm используется только с --first-demo-trade")
    if args.sim_price_path is not None and not args.protection_simulate:
        parser.error("--sim-price-path используется только с --protection-simulate")
    if not args.download_history and (args.history_bars != RESEARCH_BARS or
                                      args.history_output != Path("data/btc_usdt_15m.csv") or args.overwrite_history):
        parser.error("--history-* и --overwrite-history используются только с --download-history")

    if args.download_history:
        if args.history_output.exists() and not args.overwrite_history:
            print("История: BLOCK — CSV уже существует; требуется явный --overwrite-history", file=sys.stderr)
            return 1
        try:
            report = download_history(OkxClient(Config.from_environment()), bars=args.history_bars)
            save_history_csv(report, args.history_output, overwrite=args.overwrite_history)
        except (ConfigError, OkxError, HistoryError, OSError, ValueError):
            print("История: BLOCK — API, ряд свечей или путь CSV не прошли проверку; "
                  "существующий файл не изменён", file=sys.stderr)
            return 1
        train, validation, final = report.split_counts
        print(f"OKX BTC-USDT SPOT 15m: {len(report.candles)} подтверждённых свечей; "
              f"{report.candles[0].opened_at.isoformat()} — {report.candles[-1].closed_at.isoformat()}")
        print(f"CSV: {args.history_output}; TRAIN {train}, VALIDATION {validation}, FINAL_TEST {final}")
        print("Достаточность истории: " + ("ДОСТАТОЧНО ПО ДЛИНЕ" if report.sufficient else "НЕДОСТАТОЧНО"))
        print(f"Причина: {report.reason}")
        print("Комиссии и проскальзывание из публичных свечей неизвестны; "
              "исследование без их подтверждения требует сценарного анализа")
        print(f"Исследование: ./.venv/bin/python main.py --growth-research {args.history_output} "
              "--research-min-sz ... --research-lot-sz ... --research-tick-sz ... --research-max-mkt-sz ...")
        return 0 if report.sufficient else 1

    research_limits = (args.research_min_sz, args.research_lot_sz, args.research_tick_sz,
                       args.research_max_mkt_sz)
    research_inputs = (*research_limits, args.research_taker_fee,
                       args.research_slippage, args.research_okx_taker_rate)
    if args.growth_research is not None:
        if any(value is None for value in research_limits):
            parser.error("--growth-research требует проверенные --research-* ограничения инструмента")
        if args.research_taker_fee is not None and args.research_okx_taker_rate is not None:
            parser.error("Выберите одну форму taker-ставки: положительную стоимость или подписанную OKX")
        if ((args.research_taker_fee is not None or args.research_okx_taker_rate is not None)
                != (args.research_slippage is not None)):
            parser.error("Укажите taker-ставку и проскальзывание вместе либо используйте сценарный анализ")
        try:
            minimum, lot, tick, maximum = (Decimal(value) for value in research_limits)
            instrument = SpotInstrument("BTC-USDT", tick, lot, minimum, "live", "BTC", "USDT",
                                        max_market_size=maximum)
            historical_candles = load_candles_csv(Path(args.growth_research))
            if args.research_taker_fee is None and args.research_okx_taker_rate is None:
                scenarios = (("СЦЕНАРИЙ НИЗКИХ ЗАТРАТ", Decimal("0.0005"), Decimal("0.0005")),
                             ("СЦЕНАРИЙ БАЗОВЫХ ЗАТРАТ", Decimal("0.001"), Decimal("0.001")),
                             ("СЦЕНАРИЙ ВЫСОКИХ ЗАТРАТ", Decimal("0.002"), Decimal("0.003")))
            else:
                fee = (okx_taker_debit_rate(Decimal(args.research_okx_taker_rate))
                       if args.research_okx_taker_rate is not None else Decimal(args.research_taker_fee))
                scenarios = (("ЗАТРАТЫ ПОЛЬЗОВАТЕЛЯ (НЕ ПРОВЕРЕНЫ OKX)",
                              fee, Decimal(args.research_slippage)),)
            reports = [(label, fee, slip, research(historical_candles, ResearchConfig(instrument, fee, slip)))
                       for label, fee, slip in scenarios]
        except (ResearchError, InvalidOperation, OSError, ValueError):
            print("Исследование: BLOCK — CSV или параметры некорректны; "
                  "данные не записывались", file=sys.stderr)
            return 1
        def metric(value: Decimal | None) -> str:
            return "N/A" if value is None else str(value.quantize(Decimal("0.01")))
        print("OKX Demo BTC-USDT SPOT — только офлайн-исследование, торговля отключена")
        print("ПРЕДУПРЕЖДЕНИЕ: публичные свечи не подтверждают комиссию аккаунта или проскальзывание; "
              "все затраты ниже — сценарные/введённые пользователем допущения")
        for label, fee, slippage, report in reports:
            print(f"\n{label}: taker {fee}, проскальзывание {slippage}")
            _print_research_report(report, historical_candles, minimum, lot, tick, maximum, fee, slippage, metric)
        print("Исторические результаты не гарантируют рост; BUY/SELL, OKX API и рабочая SQLite не вызывались")
        return 0
    if any(value is not None for value in research_inputs):
        parser.error("--research-* параметры используются только с --growth-research")

    if args.sqlite_prepare or args.sqlite_init is not None:
        setup = ServerSqliteSetup()
        if args.sqlite_init is not None:
            try:
                report = setup.initialize(args.sqlite_init, args.sqlite_token or "")
            except SqliteSetupError:
                print("SQLite: BLOCKED — подтверждение или состояние пути не проверены; "
                      "существующая база не перезаписывалась. Повторите --sqlite-prepare", file=sys.stderr)
                return 1
            print("Новая локальная Demo SQLite создана однократно; торговля отключена")
        else:
            report = setup.preview()
        print(f"Подготовка Demo SQLite: {report.status}")
        print(f"Путь: {report.path}")
        print(f"Права каталога: {report.directory_mode or 'неизвестны'}; "
              f"права SQLite: {report.file_mode or 'файл отсутствует'}")
        print(f"Схема: {report.schema}; намерений: "
              f"{report.intent_count if report.intent_count is not None else 'неизвестно'}; "
              f"HALTED: {'неизвестно' if report.halted is None else 'да' if report.halted else 'нет'}")
        print(f"Бюджет: {report.budget}; Risk Manager: {report.risk}")
        print(f"Причина: {report.reason}")
        if report.token is not None and args.sqlite_prepare:
            print("Для однократной подготовки выполните вручную:")
            print(f'./.venv/bin/python main.py --sqlite-init "{SQLITE_CONFIRMATION_PHRASE}" --sqlite-token {report.token}')
        print("Это не подтверждает бюджет 5000 Demo USDT и не разрешает торговлю; "
              "SL/TP каждой частичной покупки остаются UNVERIFIED")
        return 0 if report.status in ("MISSING", "EXISTS") else 1

    if args.first_demo_trade:
        if args.first_demo_confirm is not None and args.first_demo_confirm != REVIEW_PHRASE:
            print("Первая Demo-сделка: BLOCK — фраза ручного просмотра неверна; запросы не выполнялись", file=sys.stderr)
            return 1
        try:
            demo_client = DemoClient.from_env_file()
            journal = DecisionJournal()
            risk_manager = RiskManager()
            budget_store = StrategyBudgetStore(journal.path)
            engine = ExecutionEngine(journal=journal, risk_manager=risk_manager,
                                     budget_store=budget_store)
            report = FirstTradePreflight(
                demo_client=demo_client, engine=engine,
                market_provider=lambda: get_market_snapshot(OkxClient(demo_client.config)),
                reconciliation_provider=lambda market: DemoReconciler(
                    market_provider=lambda: market, demo_client=demo_client,
                    journal=journal, risk_manager=risk_manager,
                    budget_store=budget_store).run_once(),
            ).run_once(confirmation=args.first_demo_confirm)
        except Exception:
            print("Первая Demo-сделка: BLOCK — данные или локальное состояние недоступны", file=sys.stderr)
            return 1
        print(f"Первая OKX Demo BTC-USDT SPOT сделка: {report.status}; торговые POST отключены")
        print(f"Ручной просмотр: {'подтверждён' if report.manual_review_confirmed else 'не подтверждён'}")
        if report.signal is not None:
            signal = report.signal
            print(f"EMA: {signal.signal.value}; EMA9 {signal.ema9}; EMA21 {signal.ema21}; "
                  f"закрытая свеча {signal.candle_closed_at:%Y-%m-%d %H:%M:%S} UTC")
        if report.reconciliation is not None:
            recon = report.reconciliation
            print(f"Demo-сверка: {recon.status.value}; стратегия 5000 USDT; "
                  f"доступно USDT {recon.usdt_available if recon.usdt_available is not None else 'неизвестно'}; "
                  f"внешний BTC {recon.external_btc if recon.external_btc is not None else 'неизвестен'} — продажа запрещена")
        if report.preview is not None and report.preview.quantity_btc is not None:
            preview = report.preview
            print(f"Гипотетический BUY: {preview.quantity_btc} BTC; вход {preview.estimated_entry_usdt}; "
                  f"SL {preview.estimated_stop_usdt}; TP {preview.estimated_take_usdt}; "
                  f"риск {preview.estimated_risk_usdt} USDT")
        for stage, result in report.stages:
            print(f"Этап {stage}: {result}")
        for reason in report.reasons:
            print(f"Причина BLOCK: {reason}")
        print("Ни BUY, ни SL/TP, ни SELL не отправлены; рабочая SQLite и риск не изменялись")
        return report.exit_code

    if args.health_once:
        try:
            report = HealthMonitor(HealthStore(health_settings_from_env())).run_once()
        except (HealthError, OSError):
            print("Health Monitor: UNKNOWN — локальная конфигурация недоступна", file=sys.stderr)
            return 1
        print(f"OKX Demo Watchdog Health: {report.status}")
        print(f"Последний успешный GET: {report.last_success_utc.isoformat() if report.last_success_utc else 'неизвестен'}")
        print(f"Причина: {report.reason}")
        if report.notification:
            print(f"Telegram: {report.notification}")
        print("Свежесть GET не подтверждает защиту позиции; торговля отключена")
        return report.exit_code

    if args.watchdog_once:
        try:
            report = Watchdog(store=WatchdogStore(watchdog_path_from_env())).run_once()
        except (WatchdogError, OSError):
            print("Watchdog: отдельный журнал или локальная конфигурация недоступны", file=sys.stderr)
            return 1
        print(f"OKX Watchdog Demo BTC-USDT SPOT: {report.status}")
        if report.snapshot is not None:
            snap = report.snapshot
            print(f"Виртуальный баланс: BTC {snap.btc_total} (доступно {snap.btc_available}), "
                  f"USDT {snap.usdt_total} (доступно {snap.usdt_available})")
            print(f"Открытых SPOT-ордеров: {snap.spot_order_count}; ожидающих Algo: {snap.algo_order_count}")
        for reason in report.reasons:
            print(f"Наблюдение: {reason}")
        for notification in report.notifications:
            print(f"Telegram: {notification}")
        print("Принадлежность BTC и защита не подтверждаются Watchdog; торговля отключена")
        return report.exit_code

    if args.telegram_test:
        try:
            alerter = TelegramAlerter.from_env_file()
            delivered = alerter is not None and alerter.notify("TEST")
        except Exception:
            delivered = False
        if delivered:
            print("Telegram: тестовое сообщение отправлено; торговля отключена")
            return 0
        print("Telegram: тестовое сообщение не доставлено; проверьте локальный .env и доступ к чату", file=sys.stderr)
        return 1

    if args.recovery_check:
        report = RecoveryManager(engine=ExecutionEngine()).check_once()
        print(f"OKX Demo BTC-USDT SPOT — аварийная сверка: {report.status}")
        print(f"HALTED в SQLite: {'да' if report.halted else 'нет'}")
        print(f"BTC бота по подтверждённым GET fills: {report.bot_btc if report.bot_btc is not None else 'неизвестно'}")
        print(f"Внешний BTC: {report.external_btc if report.external_btc is not None else 'неизвестно'}; продажа запрещена")
        print(f"Подтверждённый SL: {report.confirmed_sl_btc if report.confirmed_sl_btc is not None else 'неизвестно'} BTC")
        print(f"BTC без подтверждённого SL: {report.uncovered_btc if report.uncovered_btc is not None else 'неизвестно'}")
        print(f"Открытых ордеров: {report.open_order_count if report.open_order_count is not None else 'неизвестно'}")
        for reason in report.reasons:
            print(f"Причина BLOCKED: {reason}")
        print(f"Действие владельца: {report.owner_action}")
        print(f"Telegram: {report.notification}")
        print("Торговые запросы и снятие HALTED не выполнялись")
        return 1

    if args.protection_simulate:
        try:
            prices = parse_price_path(args.sim_price_path or "80000,79200,78400")
            settings = SimulationConfig()
            reports = run_sample_scenarios(prices)
        except SimulationError:
            print("Симуляция защиты: BLOCK — цены или временные события некорректны", file=sys.stderr)
            return 1
        print("OKX Demo BTC-USDT SPOT — только офлайн-симуляция, решение: BLOCK")
        print(f"Учебный BUY: {settings.planned_btc} BTC по {settings.entry_usdt} USDT; "
              f"SL 1%: {settings.stop_usdt}; TP 2%: {settings.take_usdt} USDT")
        print(f"Учебные шаги: minSz {settings.min_size_btc}, lotSz {settings.lot_size_btc}, "
              f"tickSz {settings.tick_usdt}; оценка выхода: комиссия {settings.exit_fee_rate}, "
              f"проскальзывание {settings.exit_slippage_rate}")
        for name, result in reports:
            duration = ("неизвестна" if result.unprotected_seconds is None
                        else f"{result.unprotected_seconds} с")
            print(f"\n{name}: {result.decision}; BUY {result.buy_state}; "
                  f"HALTED {'да' if result.halted else 'нет'}")
            print(f"BTC бота: куплено {result.bot_owned_btc}, осталось {result.bot_remaining_btc}; "
                  f"учебный OCO-резерв {result.simulated_reserved_btc}, "
                  f"без подтверждённого SL {result.uncovered_btc}; "
                  f"внешний BTC {result.external_btc} не использовался")
            print(f"Время известной незащищённой позиции: {duration}")
            for price, loss in result.price_losses_usdt:
                estimate = "неизвестен" if loss is None else f"{loss} USDT"
                print(f"При условной цене {price} USDT возможный убыток: {estimate}")
            if result.simulated_realized_pnl_usdt is not None:
                print(f"Учебный PnL полностью закрытой позиции: {result.simulated_realized_pnl_usdt} USDT")
            for reason in result.reasons:
                print(f"Причина BLOCK/HALTED: {reason}")
        print("Потери — условная оценка выхода по заданным ценам, не верхняя граница риска.")
        print("Исполнение не разрешено; API, рабочая SQLite, бюджет и Risk Manager не изменялись.")
        return 1

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

    if args.execution_diagnose:
        report = inspect_execution(ExecutionEngine())
        print(f"OKX Demo BTC-USDT SPOT — диагностика исполнения: BLOCK")
        print(f"SQLite: {report.sqlite_status}; причина: {report.sqlite_reason}")
        if report.halted is None:
            print("HALTED: неизвестно")
        else:
            print(f"HALTED: {'да' if report.halted else 'нет'}")
        if report.halted:
            print(f"Причина HALTED: {report.halt_reason_code}; записано: "
                  f"{report.halt_updated_at_utc.isoformat()}")
        print(f"Намерений в SQLite: {report.intent_count}")
        intent = report.latest_intent
        if intent is None:
            print("Последнее намерение: отсутствует или недоступно")
        else:
            print(f"Последнее намерение: {intent.client_order_id}; состояние {intent.state.value}; "
                  f"причина {intent.reason_code}")
            print(f"Свеча: {intent.candle_closed_at.isoformat()}; создано: "
                  f"{intent.created_at_utc.isoformat()}; обновлено: {intent.updated_at_utc.isoformat()}")
            print(f"BUY: план {intent.quantity} BTC; локально учтено {intent.filled_quantity} BTC; "
                  f"BTC после комиссии {intent.owned_btc}; биржевой ID "
                  f"{'сохранён' if intent.exchange_order_id_known else 'не сохранён'}")
            for flag in intent.flags:
                print(f"Локальный блокирующий признак: {flag}")
        print(f"Защита частичных BUY: {report.protection_status} (отдельная блокировка)")
        for missing in report.missing_protection_evidence:
            print(f"Нет доказательства: {missing}")
        print("SQLite только прочитана; OKX не вызывался; HALTED не снят; ордера не отправлены")
        return 1

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
