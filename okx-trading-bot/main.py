"""One-shot public checks and a strictly read-only private Demo check."""

from __future__ import annotations

import argparse
import sys

from bot.demo_client import DemoClient, DemoError
from bot.dry_run import DryRunOrchestrator
from bot.journal import DecisionJournal, JournalError
from bot.market_data import get_market_snapshot
from bot.okx_client import OkxClient, OkxError
from bot.risk_manager import RiskManager
from bot.strategy import StrategyError, evaluate_signal
from config import Config, ConfigError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only OKX BTC-USDT market data and EMA signal")
    commands = parser.add_mutually_exclusive_group(required=True)
    commands.add_argument("--check", action="store_true", help="Fetch public market data once")
    commands.add_argument("--signal", action="store_true", help="Calculate the EMA 9/21 signal once")
    commands.add_argument("--demo-check", action="store_true", help="Read the virtual OKX Demo account once")
    commands.add_argument("--dry-run", action="store_true", help="Record one non-executing Demo analysis cycle")
    args = parser.parse_args(argv)

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
