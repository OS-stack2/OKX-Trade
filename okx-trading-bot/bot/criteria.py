"""Automatic check of the pre-registered 2026-12-16 criteria (docs/CRITERIA_2026-12-16.md)."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
import sqlite3

from bot.demo_trader import TraderStore
from bot.trend_trader import TrendStore


MIN_NEWS_TRADES = 10
PAPER_COST_PCT = 0.2


@dataclass(frozen=True)
class Criterion:
    code: str
    ok: bool
    text: str


def naive_paper_30m_pct(radar_path: Path) -> tuple[float | None, int]:
    """Average 30-min net result of buying every strong confirmed bullish radar item."""
    if not radar_path.is_file():
        return None, 0
    with closing(sqlite3.connect(f"file:{radar_path}?mode=ro", uri=True, timeout=3)) as db:
        rows = db.execute("SELECT price_seen, p30 FROM items WHERE state='CLASSIFIED' AND relevant=1 "
                          "AND confirmed=1 AND stale=0 AND direction='bullish' AND impact>=2 "
                          "AND price_seen IS NOT NULL AND p30 IS NOT NULL").fetchall()
    moves = [(float(p30) / float(seen) - 1) * 100 - PAPER_COST_PCT for seen, p30 in rows]
    return (sum(moves) / len(moves), len(moves)) if moves else (None, 0)


def evaluate(*, news_trader: TraderStore, radar_path: Path, trend: TrendStore,
             suspicious: int) -> list[Criterion]:
    closed = news_trader.closed_positions()
    pnl = sum((p.pnl_usdt or Decimal(0)) for p in closed)
    returns = [float(p.pnl_usdt / p.entry_cost_usdt * 100) for p in closed
               if p.pnl_usdt is not None and p.entry_cost_usdt]
    analyst = sum(returns) / len(returns) if returns else None
    naive, naive_n = naive_paper_30m_pct(radar_path)
    news_halts = news_trader.count_events("HALT")
    trend_halts = int(trend.meta_value("halts_total") or "0")
    a3 = analyst is not None and naive is not None and analyst > naive
    a3_text = ("нет данных для сравнения" if analyst is None or naive is None else
               f"аналитик {analyst:+.3f}% на сделку против {naive:+.3f}% у «покупать всё» ({naive_n} новостей)")
    return [
        Criterion("A1", len(closed) >= MIN_NEWS_TRADES, f"закрытых сделок {len(closed)} (нужно ≥ {MIN_NEWS_TRADES})"),
        Criterion("A2", pnl > 0, f"результат после комиссий {pnl:+.2f} USDT (нужно > 0)"),
        Criterion("A3", a3, a3_text),
        Criterion("A4", news_halts == 0, f"аварийных остановок новостного бота: {news_halts}"),
        Criterion("B1", trend_halts == 0, f"аварийных остановок тренд-бота: {trend_halts}"),
        Criterion("B2", suspicious == 0, f"расхождений учёта (SUSPICIOUS с 19.10): {suspicious}"),
    ]


def describe(criteria: list[Criterion], *, final: bool) -> list[str]:
    status = {c.code: c.ok for c in criteria}
    lines = ["Критерии 16.12 (docs/CRITERIA_2026-12-16.md)" + ("" if final else " — промежуточно:")]
    lines += [f"  {'✅' if c.ok else '❌'} {c.code}: {c.text}" for c in criteria]
    if final:
        news_ok = all(status[k] for k in ("A1", "A2", "A3", "A4"))
        trend_ok = all(status[k] for k in ("B1", "B2"))
        lines += ["Решение по правилам:",
                  "  новости: " + ("автоторговля на реальном счёте допустима (после подготовки LIVE)" if news_ok
                                   else "аналитик — только советник в Telegram"),
                  "  тренд-бот: " + ("кандидат для реального счёта (держать BTC + трендовый фильтр)" if trend_ok
                                     else "НЕ кандидат: были сбои учёта или аварийные остановки")]
    return lines
