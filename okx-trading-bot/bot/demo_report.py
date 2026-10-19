"""Owner summary of both Demo bots, built from their local journals plus one public price."""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal

from bot.demo_trader import TraderStore
from bot.news_radar import NewsStore, paper_stats
from bot.trend_trader import TrendStore


def build_report(*, trader: TraderStore, trend: TrendStore, trend_budget: Decimal,
                 price: Decimal | None, now: datetime, final: bool = False,
                 news: NewsStore | None = None, news_trader: TraderStore | None = None,
                 suspicious: int | None = None) -> str:
    since = now - timedelta(days=7)
    lines = ["ИТОГ 2 МЕСЯЦЕВ Demo-теста" if final else "Еженедельная сводка Demo-ботов",
             f"на {now:%Y-%m-%d %H:%M} UTC" + (f", BTC {price:.0f} USDT" if price else "")]

    closed = trader.closed_positions()
    week = [p for p in closed if p.closed_at and datetime.fromisoformat(p.closed_at) >= since]
    total = sum((p.pnl_usdt or Decimal(0)) for p in closed)
    wins = sum(1 for p in closed if (p.pnl_usdt or 0) > 0)
    halted = trader.get_meta("halted")
    active = trader.active_position()
    lines += ["", "EMA 9/21 (механика):",
              f"  статус: {'HALTED — ' + halted if halted else 'работает'}",
              f"  сделок всего {len(closed)} (прибыльных {wins}), за неделю {len(week)}",
              f"  PnL всего {total:+.2f} USDT, за неделю {sum((p.pnl_usdt or Decimal(0)) for p in week):+.2f}",
              f"  позиция: {'нет' if active is None else f'{active.state} {active.net_btc} BTC от {active.entry_price}'}"]
    reasons: dict[str, int] = {}
    for p in closed:
        reasons[p.exit_reason or "?"] = reasons.get(p.exit_reason or "?", 0) + 1
    if reasons:
        lines.append("  выходы: " + ", ".join(f"{k} {v}" for k, v in sorted(reasons.items())))

    if news_trader is not None:
        closed_news = news_trader.closed_positions()
        news_week = [p for p in closed_news if p.closed_at and datetime.fromisoformat(p.closed_at) >= since]
        halted_news = news_trader.get_meta("halted")
        reasons_news: dict[str, int] = {}
        for p in closed_news:
            reasons_news[p.exit_reason or "?"] = reasons_news.get(p.exit_reason or "?", 0) + 1
        lines += ["", "Новостной бот (покупки по сильным новостям):",
                  f"  статус: {'HALTED — ' + halted_news if halted_news else 'работает'}",
                  f"  сделок всего {len(closed_news)} (прибыльных "
                  f"{sum(1 for p in closed_news if (p.pnl_usdt or 0) > 0)}), за неделю {len(news_week)}",
                  f"  PnL всего {sum((p.pnl_usdt or Decimal(0)) for p in closed_news):+.2f} USDT"]
        if reasons_news:
            lines.append("  выходы: " + ", ".join(f"{k} {v}" for k, v in sorted(reasons_news.items())))
        if news is not None:
            from bot.news_trader import AnalystCases, analyst_summary
            pnl_by_id = {p.id: p.pnl_usdt for p in closed_news}
            lines.append("  аналитик:")
            lines += ["  " + line for line in analyst_summary(AnalystCases(news_trader.path).all(),
                                                             pnl_by_id, news.path)]

    book = trend.book(trend_budget)
    trades = trend.trades()
    lines += ["", f"Дневной тренд (книга {trend_budget} USDT):",
              f"  статус: {'HALTED — ' + book.halted if book.halted else 'работает'}",
              f"  сделок всего {len(trades)}, за неделю "
              f"{sum(1 for t in trades if datetime.fromisoformat(t['at']) >= since)}",
              f"  сейчас: {book.usdt:.2f} USDT + {book.btc} BTC"]
    if price and trades:
        equity = book.usdt + book.btc * price
        start_price = Decimal(trades[0]["price"])
        hold_btc = trend_budget / start_price
        lines += [f"  капитал {equity:.2f} USDT ({(equity / trend_budget - 1) * 100:+.1f}%), "
                  f"доля BTC {book.btc * price / equity * 100:.0f}%",
                  f"  в пересчёте на BTC: {equity / price:.8f} против {hold_btc:.8f} при удержании "
                  f"с первой сделки (×{equity / price / hold_btc:.3f})"]
    if news is not None:
        rows = news.classified()
        week_rows = [r for r in rows if datetime.fromisoformat(r["seen_at"]) >= since]
        stats = paper_stats(rows)
        lines += ["", "Новостной радар:",
                  f"  разобрано новостей {stats.items} (за неделю {len(week_rows)}), оповещений {stats.alerts}",
                  f"  медианная задержка от публикации до нас: "
                  f"{'—' if stats.median_delay_s is None else str(stats.median_delay_s) + ' с'}"]
        if stats.pre_move_pct is not None:
            lines.append(f"  цена успела сдвинуться в сторону новости до нас: {stats.pre_move_pct:+.3f}%")
        if stats.horizons:
            lines.append("  бумажные сделки по сильным новостям (после комиссий 0,2%):")
            for minutes, (n, avg, win) in stats.horizons.items():
                lines.append(f"    выход через {minutes} мин: {n} шт, в среднем {avg:+.3f}%, прибыльных {win:.0f}%")
        else:
            lines.append("  бумажных сделок пока нет — сильных подтверждённых новостей не было")
    if news is not None and news_trader is not None and suspicious is not None:
        from bot.criteria import describe, evaluate
        lines += [""] + describe(evaluate(news_trader=news_trader, radar_path=news.path, trend=trend,
                                          suspicious=suspicious), final=final)
    if final:
        lines += ["", "Тест окончен. Следующий шаг — разбор с Claude по критериям выше. "
                      "Боты продолжают работать, пока их не выключат."]
    return "\n".join(lines)
