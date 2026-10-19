"""Analyst layer between a bullish headline and a Demo buy.

A good analyst does not buy the second good news appears. They ask: is it
already priced in? Was the event expected (buy the rumour, sell the news)? Is the
market overheated — price at highs after a run, expensive funding, crowded
longs? Is the bigger trend even up? This module gathers that market context
from public OKX data and the radar's recent headlines, asks DeepSeek in
reasoning mode for BUY_NOW / WAIT / SKIP with reasons, and applies a few hard
rules the model cannot override.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Any
from urllib.request import Request, urlopen

from bot.news_radar import DEEPSEEK_URL, NewsError
from bot.okx_client import OkxClient
from bot.trend_trader import read_daily_closes, target_fraction


BUY_NOW, WAIT, SKIP = "BUY_NOW", "WAIT", "SKIP"
MAX_PRE_MOVE_PCT = 1.5          # already moved this much toward the news -> priced in
MAX_NEWS_AGE = timedelta(minutes=60)
MIN_CONFIDENCE = 60
FLAG_LABELS = {
    "priced_in": "уже в цене", "sell_the_news": "покупай на слухах — продавай на фактах",
    "overheated": "перегрев", "crowded_longs": "перегружены лонги", "against_trend": "против тренда",
    "weak_source": "слабый источник", "news_saturation": "рынок пресыщен хорошими новостями",
    "high_volatility": "высокая волатильность", "stale": "новость устарела",
}


def _pct(new: float, old: float) -> float:
    return (new / old - 1) * 100


@dataclass(frozen=True)
class MarketContext:
    price: float
    minutes_since_news: float
    pre_move_pct: float | None = None           # price now vs the minute before publication
    change_pct: dict[str, float] = field(default_factory=dict)   # 1h, 4h, 24h, 7d
    from_high_pct: dict[str, float] = field(default_factory=dict)  # 24h, 7d (<= 0)
    hourly_vol_pct: float | None = None         # stdev of hourly returns over 24h
    trend_share: float | None = None            # share of rising 20..60-day lookbacks
    funding_pct: float | None = None            # per 8h
    oi_change_24h_pct: float | None = None
    long_short: float | None = None
    long_short_change_24h: float | None = None
    taker_buy_share_4h: float | None = None     # 0..1 of spot taker volume
    recent_news: tuple[str, ...] = ()

    def describe(self) -> str:
        def num(value: float | None, unit: str = "%", digits: int = 2) -> str:
            return "n/a" if value is None else f"{value:+.{digits}f}{unit}"
        lines = [f"BTC price: {self.price:.0f} USDT; news published {self.minutes_since_news:.0f} min ago",
                 f"Move since publication (toward the news): {num(self.pre_move_pct)}",
                 "Price change: " + ", ".join(f"{k} {num(v)}" for k, v in self.change_pct.items()),
                 "Distance from high: " + ", ".join(f"{k} {num(v)}" for k, v in self.from_high_pct.items()),
                 f"Hourly volatility (24h stdev): {num(self.hourly_vol_pct)}",
                 "Daily trend: " + ("n/a" if self.trend_share is None else
                                    f"{self.trend_share * 5:.0f} of 5 lookbacks (20-60 days) rising"),
                 f"Perpetual funding per 8h: {num(self.funding_pct, '%', 4)} (neutral ~ +0.01%)",
                 f"Open interest change 24h: {num(self.oi_change_24h_pct)}",
                 "Long/short account ratio: " + ("n/a" if self.long_short is None else
                                                 f"{self.long_short:.2f} (24h change {num(self.long_short_change_24h, '', 2)})"),
                 "Spot taker buy share 4h: " + ("n/a" if self.taker_buy_share_4h is None else
                                                f"{self.taker_buy_share_4h * 100:.0f}%")]
        if self.recent_news:
            lines.append("Other BTC-relevant headlines in the last 24h:")
            lines += [f"- {item}" for item in self.recent_news]
        return "\n".join(lines)

    def short_ru(self) -> str:
        parts = []
        if "24h" in self.change_pct:
            parts.append(f"24ч {self.change_pct['24h']:+.1f}%")
        if "7d" in self.change_pct:
            parts.append(f"7д {self.change_pct['7d']:+.1f}%")
        if self.pre_move_pct is not None:
            parts.append(f"с новости {self.pre_move_pct:+.2f}%")
        if self.funding_pct is not None:
            parts.append(f"фандинг {self.funding_pct:+.4f}%")
        if self.long_short is not None:
            parts.append(f"лонг/шорт {self.long_short:.2f}")
        if self.trend_share is not None:
            parts.append(f"тренд {self.trend_share * 5:.0f}/5")
        return ", ".join(parts)


def _rows(fetch: Callable[[], list[Any]]) -> list[list[Any]] | None:
    try:
        rows = fetch()
        return [r for r in rows if isinstance(r, list)] or None
    except Exception:
        return None


def build_context(public: OkxClient, *, price: float, published_at: datetime, now: datetime,
                  price_before_news: float | None, recent_news: tuple[str, ...]) -> MarketContext:
    """Every piece is optional: a failed endpoint becomes 'n/a', never a crash."""
    change: dict[str, float] = {}
    from_high: dict[str, float] = {}
    vol = None
    hourly = _rows(lambda: public.get_hourly_candles(168))
    if hourly:
        closes = [float(r[4]) for r in hourly]            # newest first
        highs = [float(r[2]) for r in hourly]
        for label, hours in (("1h", 1), ("4h", 4), ("24h", 24), ("7d", 167)):
            if len(closes) > hours:
                change[label] = _pct(price, closes[hours])
        from_high["24h"] = _pct(price, max(highs[:24] + [price]))
        from_high["7d"] = _pct(price, max(highs + [price]))
        returns = [math.log(closes[i] / closes[i + 1]) for i in range(min(24, len(closes) - 1))]
        if len(returns) > 2:
            mean = sum(returns) / len(returns)
            vol = math.sqrt(sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)) * 100
    trend = None
    try:
        trend = float(target_fraction(read_daily_closes(public)))
    except Exception:
        pass
    funding = None
    try:
        funding = float(public.get_funding_rate()[0]["fundingRate"]) * 100
    except Exception:
        pass
    oi = _rows(public.get_open_interest_history)
    oi_change = _pct(float(oi[0][1]), float(oi[24][1])) if oi and len(oi) > 24 else None
    ls = _rows(public.get_long_short_ratio)
    long_short = float(ls[0][1]) if ls else None
    ls_change = float(ls[0][1]) - float(ls[24][1]) if ls and len(ls) > 24 else None
    taker = _rows(public.get_taker_volume)
    taker_share = None
    if taker and len(taker) >= 4:
        sell = sum(float(r[1]) for r in taker[:4])
        buy = sum(float(r[2]) for r in taker[:4])
        taker_share = buy / (buy + sell) if buy + sell > 0 else None
    return MarketContext(
        price=price, minutes_since_news=(now - published_at).total_seconds() / 60,
        pre_move_pct=_pct(price, price_before_news) if price_before_news else None,
        change_pct=change, from_high_pct=from_high, hourly_vol_pct=vol, trend_share=trend,
        funding_pct=funding, oi_change_24h_pct=oi_change, long_short=long_short,
        long_short_change_24h=ls_change, taker_buy_share_4h=taker_share, recent_news=recent_news)


def recent_headlines(radar_path: Path, now: datetime, exclude_key: str, limit: int = 12) -> tuple[str, ...]:
    if not radar_path.is_file():
        return ()
    since = (now - timedelta(hours=24)).isoformat()
    with closing(sqlite3.connect(f"file:{radar_path}?mode=ro", uri=True, timeout=3)) as db:
        rows = db.execute("SELECT seen_at, direction, impact, summary FROM items WHERE state='CLASSIFIED' "
                          "AND relevant=1 AND impact>=1 AND seen_at>=? AND key!=? ORDER BY seen_at DESC LIMIT ?",
                          (since, exclude_key, limit)).fetchall()
    return tuple(f"{r[0][11:16]} UTC [{r[1]}, impact {r[2]}] {r[3]}" for r in rows)


def price_before(radar_path: Path, key: str) -> float | None:
    """The radar's price for the minute before publication, once it has filled it."""
    if not radar_path.is_file():
        return None
    with closing(sqlite3.connect(f"file:{radar_path}?mode=ro", uri=True, timeout=3)) as db:
        row = db.execute("SELECT price_pub, price_seen FROM items WHERE key=?", (key,)).fetchone()
    for value in row or ():
        if value:
            return float(value)
    return None


# --- decision ------------------------------------------------------------------

ANALYST_PROMPT = """You are a senior crypto market analyst advising a small spot-only bot (it can only buy
BTC and sell it later; no shorting, no leverage). A headline was classified as bullish for BTC.
Decide whether a short-term long (holding 30 minutes to a few hours) is worth entering NOW.

Think like a professional, not a headline chaser:
- Priced in: how much has price already moved since publication? Chasing a spike is poor risk/reward.
- Sell the news: was this event widely anticipated (scheduled decision, long-rumoured approval)?
  Then the good news is often the top.
- Overheated: price near 24h/7d highs after a strong run, funding well above +0.01%/8h,
  rising open interest with a high long/short ratio = crowded longs, fragile.
- Trend: buying with the daily trend is safer than against it.
- Source quality and whether other recent headlines confirm or contradict it.
- Magnitude: is the news big enough to matter beyond minutes?

Answer ONLY with JSON:
{"decision": "BUY_NOW" | "WAIT" | "SKIP",
 "confidence": 0-100,
 "flags": subset of ["priced_in","sell_the_news","overheated","crowded_longs","against_trend",
                     "weak_source","news_saturation","high_volatility"],
 "reasons_ru": ["2-4 short reasons in Russian"],
 "recheck_minutes": 15 or 30 (only for WAIT: when a pullback or confirmation could make it a buy),
 "hold_minutes": 30-240 (only for BUY_NOW: sensible holding time)}
WAIT means: interesting, but not at this price/moment — look again later. SKIP means: not a trade.
Most headlines should NOT be bought."""


@dataclass(frozen=True)
class AnalystDecision:
    decision: str
    confidence: int
    flags: tuple[str, ...]
    reasons_ru: tuple[str, ...]
    recheck_minutes: int = 15
    hold_minutes: int = 30
    vetoed: bool = False

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


def parse_decision(text: str) -> AnalystDecision:
    match = re.search(r"\{.*\}", text or "", re.S)
    try:
        data = json.loads(match.group(0)) if match else None
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict) or data.get("decision") not in (BUY_NOW, WAIT, SKIP):
        raise NewsError("Аналитик вернул некорректное решение")
    try:
        confidence = max(0, min(100, int(data.get("confidence", 0))))
        recheck = int(data.get("recheck_minutes") or 15)
        hold = int(data.get("hold_minutes") or 30)
    except (TypeError, ValueError):
        raise NewsError("Аналитик вернул некорректные числа") from None
    flags = tuple(f for f in data.get("flags") or () if f in FLAG_LABELS)
    reasons = tuple(str(r)[:200] for r in (data.get("reasons_ru") or ())[:4])
    return AnalystDecision(data["decision"], confidence, flags, reasons,
                           15 if recheck < 22 else 30, max(30, min(240, hold)))


def apply_hard_rules(decision: AnalystDecision, context: MarketContext) -> AnalystDecision:
    """Rules the model cannot talk its way around."""
    veto: list[tuple[str, str]] = []
    if context.minutes_since_news > MAX_NEWS_AGE.total_seconds() / 60:
        veto.append(("stale", f"новости больше {MAX_NEWS_AGE.seconds // 60} минут"))
    if context.pre_move_pct is not None and context.pre_move_pct >= MAX_PRE_MOVE_PCT:
        veto.append(("priced_in", f"цена уже выросла на {context.pre_move_pct:.2f}% с публикации"))
    if veto and decision.decision != SKIP:
        flags = tuple(dict.fromkeys(decision.flags + tuple(f for f, _ in veto)))
        return AnalystDecision(SKIP, decision.confidence, flags,
                               decision.reasons_ru + tuple(f"правило: {text}" for _, text in veto),
                               decision.recheck_minutes, decision.hold_minutes, True)
    if decision.decision == BUY_NOW and decision.confidence < MIN_CONFIDENCE:
        return AnalystDecision(WAIT, decision.confidence, decision.flags,
                               decision.reasons_ru + (f"правило: уверенность {decision.confidence}% "
                                                      f"ниже порога {MIN_CONFIDENCE}%",),
                               15, decision.hold_minutes, True)
    return decision


class NewsAnalyst:
    def __init__(self, api_key: str, transport: Callable[[Request, float], Any] | None = None,
                 model: str = "deepseek-reasoner") -> None:
        self._key = api_key
        self._model = model
        self._transport = transport or (lambda req, t: json.load(urlopen(req, timeout=t)))

    def evaluate(self, *, headline: str, summary: str, source: str, impact: int,
                 context: MarketContext, previous: tuple[str, ...] = ()) -> AnalystDecision:
        user = (f"Headline ({source}, classifier impact {impact}/3): {headline[:1200]}\n"
                f"Classifier summary: {summary}\n\nMarket context:\n{context.describe()}")
        if previous:
            user += "\n\nYour earlier decisions on this headline:\n" + "\n".join(previous)
        body = json.dumps({"model": self._model, "messages": [
            {"role": "system", "content": ANALYST_PROMPT}, {"role": "user", "content": user}]}).encode("utf-8")
        request = Request(DEEPSEEK_URL, data=body, method="POST", headers={
            "Authorization": f"Bearer {self._key}", "Content-Type": "application/json"})
        try:
            reply = self._transport(request, 120)
            text = reply["choices"][0]["message"]["content"]
        except Exception as exc:
            raise NewsError(f"Аналитик DeepSeek не ответил: {type(exc).__name__}") from None
        return apply_hard_rules(parse_decision(text), context)


def describe_ru(decision: AnalystDecision) -> str:
    label = {BUY_NOW: "ПОКУПАТЬ", WAIT: "ПОДОЖДАТЬ", SKIP: "ПРОПУСТИТЬ"}[decision.decision]
    lines = [f"🧠 Аналитик: {label} (уверенность {decision.confidence}%)"]
    if decision.flags:
        lines.append("Риски: " + ", ".join(FLAG_LABELS[f] for f in decision.flags))
    lines += [f"• {reason}" for reason in decision.reasons_ru]
    return "\n".join(lines)
