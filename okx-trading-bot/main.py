"""OKX BTC-USDT bots: Demo trading, news radar, monitoring and diagnostics.

Run `python main.py --help` for the command list; README.md explains each one.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import json
import logging
from pathlib import Path
import sqlite3
import sys

from bot.bot_ledger import read_bot_holdings
from bot.demo_client import DemoClient, DemoError
from bot.demo_report import build_report
from bot.demo_trader import (CLOSE_PHRASE, DemoTradeClient, DemoTrader, TraderError, TraderSettings, TraderStore,
                             abandon_position, resume, trader_state_path_from_env)
from bot.health_monitor import HealthError, HealthMonitor, HealthStore, health_settings_from_env
from bot.market_data import (assess_public_quote, get_market_snapshot, get_public_quote_evidence,
                             read_spot_instrument)
from bot.news_analyst import NewsAnalyst, build_context, price_before, read_fear_greed, recent_headlines
from bot.news_radar import (RSS_FEEDS, TELEGRAM_CHANNELS, DeepSeekClassifier, NewsRadar, NewsStore,
                            active_brake, news_state_path_from_env, paper_stats, run_forever)
from bot.news_trader import (RESUME_PHRASE as NEWS_RESUME_PHRASE, AnalystCases, NewsFeed, NewsTrader,
                             NewsTraderSettings, analyst_summary, cached_market,
                             news_trader_state_path_from_env, run_forever as news_run_forever)
from bot.okx_client import OkxClient, OkxError
from bot.run_monitor import RunMonitor, monitor_path_from_env
from bot.strategy import StrategyError, evaluate_signal
from bot.telegram_alerts import TelegramAlerter
from bot.trend_trader import (TrendSettings, TrendStore, TrendTrader, read_daily_closes,
                              resume as trend_resume, trend_state_path_from_env)
from bot.watchdog import Watchdog, WatchdogError, WatchdogStore, watchdog_path_from_env
from config import Config, ConfigError, _read_env_file


def _demo_trader_command(args: argparse.Namespace) -> int:
    try:
        store = TraderStore(trader_state_path_from_env())
        now = datetime.now(timezone.utc)
        if args.demo_trade_resume is not None:
            if resume(store, args.demo_trade_resume, now):
                print("Demo-трейдер: HALT снят; следующий цикл сверит позицию с OKX")
                return 0
            print("Demo-трейдер: HALT не снят — неверная фраза или остановки нет", file=sys.stderr)
            return 1
        if args.demo_trade_abandon is not None:
            if abandon_position(store, args.demo_trade_abandon, now):
                print("Demo-трейдер: активная позиция больше не отслеживается")
                return 0
            print("Demo-трейдер: неверная фраза или нет активной позиции", file=sys.stderr)
            return 1
        if args.demo_trade_status:
            halted = store.get_meta("halted")
            position = store.active_position()
            closed = store.closed_positions()
            settings_budget = TraderSettings().budget_usdt
            pnl = sum((p.pnl_usdt or Decimal(0)) for p in closed)
            wins = sum(1 for p in closed if (p.pnl_usdt or 0) > 0)
            print(f"Demo-трейдер: {'HALTED ' + halted if halted else 'работает'}")
            print(f"Последняя обработанная свеча: {store.get_meta('last_candle') or 'нет'}")
            if position is None:
                print("Позиция: нет")
            else:
                print(f"Позиция: {position.state} {position.net_btc or position.planned_btc} BTC, "
                      f"вход {position.entry_price}, SL {position.stop_price}, TP {position.take_price}")
            print(f"Закрыто сделок: {len(closed)}, прибыльных {wins}; PnL {pnl:+.2f} USDT")
            for item in closed[-10:]:
                print(f"  {item.closed_at} {item.exit_reason}: {item.net_btc} BTC, вход {item.entry_price}, "
                      f"PnL {item.pnl_usdt:+.2f}")
            return 1 if halted else 0
        if args.demo_trade_close is not None and args.demo_trade_close != CLOSE_PHRASE:
            print("Demo-трейдер: неверная фраза закрытия", file=sys.stderr)
            return 1
        settings = (TraderSettings() if args.demo_trade_close is not None
                    else TraderSettings.from_env_file())
        client = DemoTradeClient.from_env_file()
        alerter = TelegramAlerter.from_env_file()
        public = OkxClient(client.config)
        trader = DemoTrader(client=client, store=store, settings=settings,
                            market_provider=lambda: get_market_snapshot(public),
                            notifier=alerter.send_text if alerter is not None else None,
                            brake=active_brake)
        report = trader.close_now() if args.demo_trade_close is not None else trader.run_once()
    except (ConfigError, TraderError, OSError, sqlite3.Error) as exc:
        print(f"Demo-трейдер: BLOCK — {exc}", file=sys.stderr)
        _record_run("demo-trader", False, str(exc))
        return 1
    except OkxError as exc:
        print(f"Demo-трейдер: публичные данные OKX недоступны — {exc}", file=sys.stderr)
        _record_run("demo-trader", False, f"публичные данные OKX: {exc}")
        return 1
    print(f"OKX Demo-трейдер BTC-USDT: {report.status}")
    for line in report.lines:
        print(line)
    _record_run("demo-trader", report.status != "ERROR", report.lines[-1] if report.lines else "")
    return report.exit_code


def _record_run(name: str, ok: bool, detail: str) -> None:
    """Silent-failure tracking; it must never break the bot run itself."""
    try:
        message = RunMonitor(monitor_path_from_env()).record(name, ok, datetime.now(timezone.utc), detail)
        if message:
            alerter = TelegramAlerter.from_env_file()
            if alerter is not None:
                alerter.send_text("OKX Demo: " + message)
    except Exception:
        pass


def _news_trade_command(args: argparse.Namespace) -> int:
    store = TraderStore(news_trader_state_path_from_env())
    if args.news_trade_resume is not None:
        if args.news_trade_resume == NEWS_RESUME_PHRASE and store.get_meta("halted"):
            store.set_meta("halted", None)
            print("Новостной бот: HALT снят; следующий тик сверит позицию с OKX")
            return 0
        print("Новостной бот: неверная фраза или остановки нет", file=sys.stderr)
        return 1
    if args.news_trade_status:
        halted = store.get_meta("halted")
        position = store.active_position()
        closed = store.closed_positions()
        print(f"Новостной бот: {'HALTED ' + halted if halted else 'работает'}")
        print("Позиция: " + ("нет" if position is None else
                             f"{position.state} {position.net_btc or position.planned_btc} BTC, "
                             f"вход {position.entry_price} в {position.opened_at[11:19]} UTC"))
        pnl = sum((p.pnl_usdt or Decimal(0)) for p in closed)
        print(f"Закрыто сделок: {len(closed)}; PnL {pnl:+.2f} USDT")
        for item in closed[-10:]:
            print(f"  {item.opened_at[:16]} → {item.exit_reason}: PnL {item.pnl_usdt:+.2f}")
        print("Аналитик:")
        for line in analyst_summary(AnalystCases(store.path).all(), {p.id: p.pnl_usdt for p in closed},
                                    news_state_path_from_env()):
            print(line)
        for case in AnalystCases(store.path).all()[-5:]:
            last = (json.loads(case["decisions"]) or [{}])[-1]
            print(f"  {case['seen_at'][:16]} {case['state']}: «{case['summary'][:70]}» "
                  f"{last.get('decision', '')} {last.get('confidence', '')}%")
        return 1 if halted else 0
    try:
        settings = NewsTraderSettings.from_env_file()
        client = DemoTradeClient.from_env_file()
    except ConfigError as exc:
        print(f"Новостной бот: BLOCK — {exc}", file=sys.stderr)
        return 1
    public = OkxClient(client.config)
    alerter = TelegramAlerter.from_env_file()

    def price() -> Decimal:
        evidence = get_public_quote_evidence(public)
        if evidence.last_price is None:
            raise OkxError("ticker unavailable")
        return evidence.last_price

    radar_path = news_state_path_from_env()
    values = _read_env_file(Path(__file__).resolve().parent / ".env")
    analyst = None
    if values.get("OKX_NEWSTRADER_ANALYST", "1") == "1":
        analyst = NewsAnalyst(values.get("OKX_DEEPSEEK_API_KEY", ""))

    def context_for(case, now):
        return build_context(public, price=float(price()),
                             published_at=datetime.fromisoformat(case["published_at"]), now=now,
                             price_before_news=price_before(radar_path, case["news_key"]),
                             recent_news=recent_headlines(radar_path, now, case["news_key"]),
                             fear_greed=read_fear_greed)

    trader = NewsTrader(feed=NewsFeed(radar_path, store), news_settings=settings,
                        analyst=analyst, context_builder=context_for,
                        client=client, store=store,
                        market_provider=cached_market(lambda: read_spot_instrument(public), price),
                        notifier=alerter.send_text if alerter is not None else None, brake=active_brake)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    def on_report(report) -> None:
        for line in report.lines:
            logging.info("%s: %s", report.status, line)

    print("Новостной бот запущен", flush=True)
    news_run_forever(trader, on_report=on_report,
                     on_heartbeat=lambda ok: _record_run("news-trader", ok, "ошибки чтения OKX подряд"))
    return 0


def _news_command(args: argparse.Namespace) -> int:
    store = NewsStore(news_state_path_from_env())
    if args.news_status:
        rows = store.classified()
        stats = paper_stats(rows)
        print(f"Новостной радар: разобрано {stats.items}, оповещений {stats.alerts}, "
              f"медианная задержка {stats.median_delay_s if stats.median_delay_s is not None else '—'} с")
        brake = active_brake(datetime.now(timezone.utc), store.path)
        print(f"Тормоз: {brake or 'нет'}")
        if stats.pre_move_pct is not None:
            print(f"Цена ушла в сторону новости ещё до того, как мы её увидели: {stats.pre_move_pct:+.3f}%")
        for minutes, (n, avg, win) in stats.horizons.items():
            print(f"  бумажная сделка, выход через {minutes} мин: {n} шт, в среднем {avg:+.3f}% "
                  f"после комиссий, прибыльных {win:.0f}%")
        return 0
    values = _read_env_file(Path(__file__).resolve().parent / ".env")
    try:
        classifier = DeepSeekClassifier(values.get("OKX_DEEPSEEK_API_KEY", ""))
        public = OkxClient(Config.from_environment())
    except ConfigError as exc:
        print(f"Новостной радар: BLOCK — {exc}", file=sys.stderr)
        return 1
    alerter = TelegramAlerter.from_env_file()

    def price_now() -> Decimal:
        evidence = get_public_quote_evidence(public)
        if evidence.last_price is None:
            raise OkxError("ticker unavailable")
        return evidence.last_price

    def on_cycle(result) -> None:
        sources = len(TELEGRAM_CHANNELS) + len(RSS_FEEDS)
        failed = sum(1 for e in result.errors if e.startswith(("tg:", "rss:")))
        _record_run("news-radar", failed < sources, "; ".join(result.errors)[:300])

    radar = NewsRadar(store=store, classifier=classifier, price_now=price_now,
                      minute_candles=lambda: public.get_minute_candles(300),
                      notifier=alerter.send_text if alerter is not None else None)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print("Новостной радар запущен", flush=True)
    run_forever(radar, on_cycle=on_cycle)
    return 0


def _trend_command(args: argparse.Namespace) -> int:
    try:
        store = TrendStore(trend_state_path_from_env())
        if args.trend_resume is not None:
            if trend_resume(store, args.trend_resume):
                print("Тренд-бот: HALT снят")
                return 0
            print("Тренд-бот: неверная фраза", file=sys.stderr)
            return 1
        if args.trend_status:
            try:
                budget = TrendSettings.from_env_file().budget_usdt
            except ConfigError:
                budget = TrendSettings().budget_usdt
            book = store.book(budget)
            print(f"Тренд-бот: {'HALTED ' + book.halted if book.halted else 'работает'}")
            print(f"Книга: {book.usdt:.2f} USDT + {book.btc} BTC (стартовый бюджет {budget} USDT)")
            print(f"Последний обработанный день: {book.last_day or 'нет'}")
            for row in store.trades()[-10:]:
                print(f"  {row['day']} {row['side']} {row['btc']} BTC @ {row['price']}, "
                      f"цель {Decimal(row['target']) * 100:.0f}%, капитал {Decimal(row['equity']):.2f}")
            return 1 if book.halted else 0
        settings = TrendSettings.from_env_file()
        client = DemoTradeClient.from_env_file()
        public = OkxClient(client.config)
        alerter = TelegramAlerter.from_env_file()
        def price() -> Decimal:
            snapshot = get_public_quote_evidence(public)
            if snapshot.last_price is None:
                raise OkxError("ticker unavailable")
            return snapshot.last_price
        def best_quote() -> tuple[Decimal, Decimal]:
            row = public.get_ticker()[0]
            bid, ask = Decimal(row["bidPx"]), Decimal(row["askPx"])
            if not (0 < bid < ask) or ask / bid > Decimal("1.01"):
                raise OkxError("bid/ask unavailable or implausible")
            return bid, ask

        trader = TrendTrader(client=client, store=store, settings=settings,
                             closes_provider=lambda: read_daily_closes(public),
                             instrument_provider=lambda: read_spot_instrument(public),
                             price_provider=price,
                             notifier=alerter.send_text if alerter is not None else None,
                             brake=active_brake, quote_provider=best_quote)
        report = trader.run_once()
    except (ConfigError, OkxError, OSError, sqlite3.Error) as exc:
        print(f"Тренд-бот: BLOCK — {exc}", file=sys.stderr)
        _record_run("demo-trend", False, str(exc))
        return 1
    print(f"OKX Demo тренд-бот BTC: {report.status}")
    for line in report.lines:
        print(line)
    _record_run("demo-trend", report.status != "ERROR", report.lines[-1] if report.lines else "")
    return report.exit_code


def _suspicious_count() -> int | None:
    try:
        return WatchdogStore(watchdog_path_from_env()).counter("suspicious")[0]
    except (WatchdogError, OSError):
        return None


def _report_command(args: argparse.Namespace) -> int:
    try:
        try:
            budget = TrendSettings.from_env_file().budget_usdt
        except ConfigError:
            budget = TrendSettings().budget_usdt
        try:
            price = get_public_quote_evidence(OkxClient(Config.from_environment())).last_price
        except (ConfigError, OkxError, OSError):
            price = None
        text = build_report(trader=TraderStore(trader_state_path_from_env()),
                            trend=TrendStore(trend_state_path_from_env()), trend_budget=budget,
                            price=price, now=datetime.now(timezone.utc), final=args.demo_report_final,
                            news=NewsStore(news_state_path_from_env()),
                            news_trader=TraderStore(news_trader_state_path_from_env()),
                            suspicious=_suspicious_count())
    except (OSError, sqlite3.Error, TraderError) as exc:
        print(f"Сводка: BLOCK — {exc}", file=sys.stderr)
        return 1
    print(text)
    alerter = TelegramAlerter.from_env_file()
    sent = alerter is not None and alerter.send_text(text)
    print("Telegram: " + ("отправлено" if sent else "не отправлено"))
    return 0 if sent else 1


def _market_diagnose() -> int:
    try:
        evidence = get_public_quote_evidence(OkxClient(Config.from_environment()))
        assessment = assess_public_quote(evidence, datetime.now(timezone.utc))
    except (ConfigError, OkxError, ValueError, OSError):
        print("Публичные данные BTC-USDT: BLOCKED — регион или конфигурация недоступны", file=sys.stderr)
        return 1
    print("Публичные данные BTC-USDT SPOT: " + ("OK" if not assessment.reasons else "BLOCKED"))
    for name, status in assessment.checks:
        print(f"{name}: {status}")
    if evidence.last_price is not None:
        print(f"ticker.last: {evidence.last_price} USDT")
    if evidence.ticker_as_of_utc is not None:
        print(f"ticker.ts UTC: {evidence.ticker_as_of_utc.isoformat()}")
    if evidence.instrument is not None:
        instrument = evidence.instrument
        print(f"Инструмент: {instrument.instrument_id}, state={instrument.state}, "
              f"baseCcy={instrument.base_currency}, quoteCcy={instrument.quote_currency}")
        print(f"minSz={instrument.min_size}, lotSz={instrument.lot_size}, tickSz={instrument.tick_size}, "
              f"maxLmtSz={instrument.max_limit_size}, maxMktSz={instrument.max_market_size}")
    for reason in assessment.reasons:
        print(f"Причина BLOCKED: {reason}")
    print("Только публичные GET; Demo-ключи и торговая SQLite не использовались")
    return 0 if not assessment.reasons else 1


def _health_once() -> int:
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


def _watchdog_once() -> int:
    try:
        report = Watchdog(store=WatchdogStore(watchdog_path_from_env()),
                          ledger_loader=read_bot_holdings).run_once()
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


def _telegram_test() -> int:
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


def _demo_check() -> int:
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


def _public_check(args: argparse.Namespace) -> int:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OKX BTC-USDT Demo bots, news radar and monitoring")
    commands = parser.add_mutually_exclusive_group(required=True)
    add = commands.add_argument
    # diagnostics
    add("--check", action="store_true", help="Public BTC-USDT price and closed 15m candles")
    add("--signal", action="store_true", help="EMA 9/21 signal on closed 15m candles")
    add("--market-diagnose", action="store_true", help="Public instrument limits and quote check")
    add("--demo-check", action="store_true", help="Read the Demo account balances")
    add("--telegram-test", action="store_true", help="Send one test message to the owner")
    # monitoring
    add("--watchdog-once", action="store_true", help="One Demo account observation, reconciled with bot journals")
    add("--health-once", action="store_true", help="Check that the Watchdog keeps observing")
    add("--demo-report", action="store_true", help="Weekly summary of the Demo bots to Telegram")
    add("--demo-report-final", action="store_true", help="Final summary of the 2-month Demo test to Telegram")
    # EMA 9/21 trader (stopped; commands kept for its journal)
    add("--demo-trade-once", action="store_true", help="One EMA 9/21 Demo trading cycle")
    add("--demo-trade-status", action="store_true", help="EMA trader journal")
    add("--demo-trade-resume", metavar="PHRASE", help="Clear an EMA trader HALT after manual review")
    add("--demo-trade-close", metavar="PHRASE", help="Close the EMA trader position at market")
    add("--demo-trade-abandon", metavar="PHRASE", help="Forget a position closed manually on OKX")
    # daily trend bot
    add("--trend-once", action="store_true", help="One daily trend-allocator cycle")
    add("--trend-status", action="store_true", help="Trend-allocator book")
    add("--trend-resume", metavar="PHRASE", help="Clear a trend-allocator HALT after manual review")
    # news
    add("--news-radar", action="store_true", help="Run the news radar loop")
    add("--news-status", action="store_true", help="News radar paper statistics")
    add("--news-trade", action="store_true", help="Run the Demo news trader loop")
    add("--news-trade-status", action="store_true", help="News trader journal and analyst decisions")
    add("--news-trade-resume", metavar="PHRASE", help="Clear a news trader HALT after manual review")
    args = parser.parse_args(argv)

    if args.demo_trade_once or args.demo_trade_status or args.demo_trade_resume is not None \
            or args.demo_trade_abandon is not None or args.demo_trade_close is not None:
        return _demo_trader_command(args)
    if args.trend_once or args.trend_status or args.trend_resume is not None:
        return _trend_command(args)
    if args.news_trade or args.news_trade_status or args.news_trade_resume is not None:
        return _news_trade_command(args)
    if args.news_radar or args.news_status:
        return _news_command(args)
    if args.demo_report or args.demo_report_final:
        return _report_command(args)
    if args.market_diagnose:
        return _market_diagnose()
    if args.health_once:
        return _health_once()
    if args.watchdog_once:
        return _watchdog_once()
    if args.telegram_test:
        return _telegram_test()
    if args.demo_check:
        return _demo_check()
    return _public_check(args)


if __name__ == "__main__":
    raise SystemExit(main())
