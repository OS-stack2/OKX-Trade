"""News radar: crypto/macro headlines -> DeepSeek classification -> alerts, brake, paper trades.

Three jobs, none of which places orders by itself:
* radar  — important BTC news goes to Telegram with direction and our delay;
* brake  — a confirmed high-impact bearish item blocks new bot entries for a while;
* paper  — every directional item records the BTC price when it was published,
  when we saw it, and 1/5/30/60 minutes later. That answers, with data, whether
  trading on news at our speed would have earned anything.

Sources are polled over plain HTTPS: public Telegram previews (t.me/s/<name>)
and RSS. A source is "primed" on its first read: whatever is already there is
stored without classification, so a restart never re-alerts on the backlog.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from email.utils import parsedate_to_datetime
import html
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Iterator
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

from config import ConfigError, _read_env_file


log = logging.getLogger(__name__)

TELEGRAM_CHANNELS = ("watcherguru", "wublockchainenglish", "cointelegraph")
RSS_FEEDS = {
    "coindesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "theblock": "https://www.theblock.co/rss.xml",
    "decrypt": "https://decrypt.co/feed",
    "bitcoinmagazine": "https://bitcoinmagazine.com/.rss/full/",
    "fed": "https://www.federalreserve.gov/feeds/press_all.xml",
    "sec": "https://www.sec.gov/news/pressreleases.rss",
}
TELEGRAM_EVERY = 30      # seconds
RSS_EVERY = 120
MAX_AGE_TO_CLASSIFY = timedelta(minutes=30)   # older items are history, not news
HORIZONS = (1, 5, 30, 60)                     # minutes after we saw the item
ROUND_TRIP_COST = Decimal("0.002")            # taker 0.1% in and out
BRAKE_FOR = timedelta(hours=2)
ALERT_IMPACT = 2
BRAKE_IMPACT = 3
DEFAULT_STATE_PATH = Path(__file__).resolve().parents[1] / "data" / "news_radar.sqlite3"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
USER_AGENT = "Mozilla/5.0 (okx-news-radar)"


class NewsError(Exception):
    """A source or the classifier failed; the loop logs it and carries on."""


@dataclass(frozen=True)
class NewsItem:
    source: str
    key: str
    url: str
    published_at: datetime
    text: str


@dataclass(frozen=True)
class Classification:
    btc_relevant: bool
    direction: str        # bullish, bearish, neutral
    impact: int           # 0..3
    confirmed: bool       # official/confirmed, not rumour or opinion
    stale: bool           # old event, recap or repost
    summary_ru: str

    @property
    def actionable(self) -> bool:
        return (self.btc_relevant and self.direction in ("bullish", "bearish")
                and self.confirmed and not self.stale)


def _http_get(url: str, timeout: float = 15) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def _clean(markup: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", markup)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"[ \t\xa0]+", " ", html.unescape(text)).strip()


# --- sources -----------------------------------------------------------------

_TG_TEXT = re.compile(r'<div class="tgme_widget_message_text js-message_text"[^>]*>(.*?)</div>', re.S)
_TG_TIME = re.compile(r'<time datetime="([^"]+)"')


def parse_telegram(channel: str, page: str) -> list[NewsItem]:
    items = []
    for chunk in page.split('data-post="')[1:]:
        post = chunk[:chunk.find('"')]
        if "/" not in post:
            continue
        text_match, time_match = _TG_TEXT.search(chunk), _TG_TIME.search(chunk)
        if not text_match or not time_match:
            continue  # photo/video without caption, service message
        text = _clean(text_match.group(1))
        try:
            published = datetime.fromisoformat(time_match.group(1)).astimezone(timezone.utc)
        except ValueError:
            continue
        if text:
            items.append(NewsItem(f"tg:{channel}", f"tg:{post.lower()}", f"https://t.me/{post}",
                                  published, text))
    return items


def parse_rss(name: str, document: str, now: datetime) -> list[NewsItem]:
    try:
        root = ET.fromstring(document.encode("utf-8"))
    except ET.ParseError as exc:
        raise NewsError(f"RSS {name}: некорректный XML") from exc
    items = []
    for node in root.iter("item"):
        title = _clean(node.findtext("title") or "")
        link = (node.findtext("link") or "").strip()
        guid = (node.findtext("guid") or link or title).strip()
        description = _clean(node.findtext("description") or "")[:600]
        raw_date = node.findtext("pubDate")
        try:
            published = parsedate_to_datetime(raw_date).astimezone(timezone.utc) if raw_date else now
        except (TypeError, ValueError):
            published = now
        if title:
            items.append(NewsItem(f"rss:{name}", f"rss:{name}:{guid}"[:300], link,
                                  published, f"{title}\n{description}".strip()))
    return items


# --- classifier --------------------------------------------------------------

PROMPT = """You classify one news item for a Bitcoin (BTC-USDT spot) trading bot.
Return ONLY a JSON object with keys:
  "btc_relevant": true if the item can plausibly move the BTC price,
  "direction": "bullish", "bearish" or "neutral" for BTC price,
  "impact": integer 0-3 —
      3 = could move BTC 2%+ within an hour (a state adopts or bans bitcoin, spot ETF
          approval/denial, top exchange hack or insolvency, surprise Fed decision,
          major sanctions/war shock);
      2 = notable, likely 0.5-2% (large institutional buy/sell, big regulatory step,
          large ETF flows, major macro data surprise);
      1 = minor or local; 0 = irrelevant,
  "confirmed": true only if it is an announced/official fact, false for rumours,
      "reportedly", opinions, predictions, plans "to consider", or denials of a rumour,
  "stale": true if it is a recap, analysis, price commentary, or an old event,
  "summary_ru": one short sentence in Russian with the essence.
Be conservative: most items are impact 0-1."""


class DeepSeekClassifier:
    def __init__(self, api_key: str, transport: Callable[[Request, float], Any] | None = None,
                 model: str = "deepseek-chat") -> None:
        if not api_key:
            raise ConfigError("Нужен OKX_DEEPSEEK_API_KEY в .env")
        self._key = api_key
        self._model = model
        self._transport = transport or (lambda req, t: json.load(urlopen(req, timeout=t)))

    def classify(self, item: NewsItem) -> Classification:
        body = json.dumps({
            "model": self._model, "temperature": 0, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": PROMPT},
                         {"role": "user", "content": f"Source: {item.source}\n"
                                                     f"Published: {item.published_at:%Y-%m-%d %H:%M} UTC\n\n"
                                                     f"{item.text[:1500]}"}],
        }).encode("utf-8")
        request = Request(DEEPSEEK_URL, data=body, method="POST", headers={
            "Authorization": f"Bearer {self._key}", "Content-Type": "application/json"})
        try:
            reply = self._transport(request, 30)
            data = json.loads(reply["choices"][0]["message"]["content"])
        except Exception as exc:
            raise NewsError(f"DeepSeek не ответил: {type(exc).__name__}") from None
        return parse_classification(data)


def parse_classification(data: Any) -> Classification:
    if not isinstance(data, dict):
        raise NewsError("DeepSeek вернул не объект")
    direction = data.get("direction")
    impact = data.get("impact")
    if direction not in ("bullish", "bearish", "neutral") or type(impact) is not int or not 0 <= impact <= 3:
        raise NewsError("DeepSeek вернул некорректную классификацию")
    return Classification(bool(data.get("btc_relevant")), direction, impact,
                          bool(data.get("confirmed")), bool(data.get("stale")),
                          str(data.get("summary_ru") or "")[:300])


# --- state -------------------------------------------------------------------

class NewsStore:
    def __init__(self, path: Path = DEFAULT_STATE_PATH) -> None:
        self.path = path

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=5, isolation_level=None)) as db:
            db.row_factory = sqlite3.Row
            db.execute("BEGIN IMMEDIATE")
            try:
                db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                db.execute("""CREATE TABLE IF NOT EXISTS items (
                    key TEXT PRIMARY KEY, source TEXT NOT NULL, url TEXT NOT NULL,
                    published_at TEXT NOT NULL, seen_at TEXT NOT NULL, text TEXT NOT NULL,
                    state TEXT NOT NULL,          -- PRIMED, OLD, NEW, CLASSIFIED, FAILED
                    relevant INTEGER, direction TEXT, impact INTEGER, confirmed INTEGER,
                    stale INTEGER, summary TEXT, alerted INTEGER NOT NULL DEFAULT 0,
                    price_pub TEXT, price_seen TEXT, p1 TEXT, p5 TEXT, p30 TEXT, p60 TEXT)""")
                db.execute("CREATE INDEX IF NOT EXISTS items_state ON items(state)")
                yield db
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def meta(self, key: str) -> str | None:
        with self._db() as db:
            row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            return None if row is None else row["value"]

    def set_meta(self, key: str, value: str | None) -> None:
        with self._db() as db:
            if value is None:
                db.execute("DELETE FROM meta WHERE key=?", (key,))
            else:
                db.execute("INSERT INTO meta VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                           (key, value))

    def add(self, items: Sequence[NewsItem], now: datetime, *, primed: bool) -> list[NewsItem]:
        """Store unseen items; return those that are fresh news to classify."""
        fresh = []
        with self._db() as db:
            for item in items:
                if db.execute("SELECT 1 FROM items WHERE key=?", (item.key,)).fetchone():
                    continue
                if not primed:
                    state = "PRIMED"
                elif now - item.published_at > MAX_AGE_TO_CLASSIFY:
                    state = "OLD"
                else:
                    state = "NEW"
                db.execute("INSERT INTO items(key, source, url, published_at, seen_at, text, state) "
                           "VALUES(?, ?, ?, ?, ?, ?, ?)",
                           (item.key, item.source, item.url, item.published_at.isoformat(),
                            now.isoformat(), item.text[:4000], state))
                if state == "NEW":
                    fresh.append(item)
        return fresh

    def save_classification(self, key: str, result: Classification | None) -> None:
        with self._db() as db:
            if result is None:
                db.execute("UPDATE items SET state='FAILED' WHERE key=?", (key,))
                return
            db.execute("UPDATE items SET state='CLASSIFIED', relevant=?, direction=?, impact=?, "
                       "confirmed=?, stale=?, summary=? WHERE key=?",
                       (int(result.btc_relevant), result.direction, result.impact,
                        int(result.confirmed), int(result.stale), result.summary_ru, key))

    def mark_alerted(self, key: str) -> None:
        with self._db() as db:
            db.execute("UPDATE items SET alerted=1 WHERE key=?", (key,))

    def directional_missing_prices(self, since: datetime) -> list[sqlite3.Row]:
        """Directional items still missing prices, within the reach of 1m candles."""
        with self._db() as db:
            return db.execute("SELECT * FROM items WHERE state='CLASSIFIED' AND relevant=1 "
                              "AND direction!='neutral' AND impact>=1 AND p60 IS NULL AND seen_at>=?",
                              (since.isoformat(),)).fetchall()

    def set_prices(self, key: str, **prices: Decimal | None) -> None:
        values = {k: format(v, "f") for k, v in prices.items() if v is not None
                  and k in ("price_pub", "price_seen", "p1", "p5", "p30", "p60")}
        if values:
            with self._db() as db:
                db.execute(f"UPDATE items SET {', '.join(f'{k}=COALESCE({k}, ?)' for k in values)} "
                           "WHERE key=?", (*values.values(), key))

    def classified(self, since: datetime | None = None) -> list[sqlite3.Row]:
        with self._db() as db:
            query = "SELECT * FROM items WHERE state='CLASSIFIED'"
            if since is None:
                return db.execute(query).fetchall()
            return db.execute(query + " AND seen_at>=?", (since.isoformat(),)).fetchall()


# --- prices ------------------------------------------------------------------

def minute_closes(fetch: Callable[[], list[Any]]) -> dict[datetime, Decimal]:
    """Map minute-open time -> close for the last confirmed 1m candles."""
    result = {}
    for row in fetch():
        try:
            if row[8] != "1":
                continue
            opened = datetime.fromtimestamp(int(row[0]) / 1000, tz=timezone.utc)
            result[opened] = Decimal(row[4])
        except (IndexError, TypeError, ValueError, InvalidOperation, OverflowError, OSError):
            continue
    return result


def _minute(moment: datetime) -> datetime:
    return moment.replace(second=0, microsecond=0)


# --- brake (read by the trading bots) ------------------------------------------

def news_state_path_from_env(path: Path | None = None) -> Path:
    values = _read_env_file(path or Path(__file__).resolve().parents[1] / ".env")
    raw = values.get("OKX_NEWS_STATE_PATH")
    return Path(raw) if raw else DEFAULT_STATE_PATH


def active_brake(now: datetime, path: Path | None = None) -> str | None:
    """Reason text while a news brake is active, else None. Never raises."""
    try:
        state = path or news_state_path_from_env()
        if not state.is_file():
            return None
        with closing(sqlite3.connect(f"file:{state}?mode=ro", uri=True, timeout=3)) as db:
            rows = dict(db.execute("SELECT key, value FROM meta WHERE key IN ('brake_until','brake_reason')"))
        until = rows.get("brake_until")
        if until and datetime.fromisoformat(until) > now:
            return f"{rows.get('brake_reason', 'тревожная новость')} (до {until[11:16]} UTC)"
    except Exception:
        return None
    return None


# --- radar -------------------------------------------------------------------

@dataclass(frozen=True)
class CycleResult:
    new_items: int
    classified: int
    alerts: int
    errors: tuple[str, ...]


class NewsRadar:
    def __init__(self, *, store: NewsStore, classifier: DeepSeekClassifier,
                 fetch_page: Callable[[str], str] = _http_get,
                 price_now: Callable[[], Decimal] | None = None,
                 minute_candles: Callable[[], list[Any]] | None = None,
                 notifier: Callable[[str], object] | None = None,
                 clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> None:
        self.store = store
        self.classifier = classifier
        self.fetch_page = fetch_page
        self.price_now = price_now
        self.minute_candles = minute_candles
        self.notifier = notifier
        self.clock = clock

    def _notify(self, text: str) -> None:
        if self.notifier is not None:
            try:
                self.notifier(text)
            except Exception:
                pass

    def poll(self, *, telegram: bool, rss: bool) -> CycleResult:
        now = self.clock()
        errors: list[str] = []
        fresh: list[NewsItem] = []
        sources: list[tuple[str, Callable[[], list[NewsItem]]]] = []
        if telegram:
            sources += [(f"tg:{c}", lambda c=c: parse_telegram(c, self.fetch_page(f"https://t.me/s/{c}")))
                        for c in TELEGRAM_CHANNELS]
        if rss:
            sources += [(f"rss:{n}", lambda n=n, u=u: parse_rss(n, self.fetch_page(u), now))
                        for n, u in RSS_FEEDS.items()]
        for name, read in sources:
            try:
                items = read()
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}")
                continue
            primed = self.store.meta(f"primed:{name}") is not None
            fresh += self.store.add(items, now, primed=primed)
            if not primed:
                self.store.set_meta(f"primed:{name}", now.isoformat())
        price = None
        if fresh and self.price_now is not None:
            try:
                price = self.price_now()
            except Exception as exc:
                errors.append(f"price: {type(exc).__name__}")
        classified = alerts = 0
        for item in sorted(fresh, key=lambda i: i.published_at):
            try:
                result = self.classifier.classify(item)
            except NewsError as exc:
                self.store.save_classification(item.key, None)
                errors.append(str(exc))
                continue
            classified += 1
            self.store.save_classification(item.key, result)
            if price is not None:
                self.store.set_prices(item.key, price_seen=price)
            if result.btc_relevant and result.impact >= ALERT_IMPACT:
                alerts += 1
                self._alert(item, result, price, now)
            if result.actionable and result.direction == "bearish" and result.impact >= BRAKE_IMPACT:
                self._brake(result, now)
        try:
            self.update_prices()
        except Exception as exc:
            errors.append(f"prices: {type(exc).__name__}")
        return CycleResult(len(fresh), classified, alerts, tuple(errors))

    def _alert(self, item: NewsItem, result: Classification, price: Decimal | None, now: datetime) -> None:
        mark = {"bullish": "🟢 рост", "bearish": "🔴 падение"}.get(result.direction, "⚪ нейтрально")
        delay = int((now - item.published_at).total_seconds())
        flags = []
        if not result.confirmed:
            flags.append("слух/не подтверждено")
        if result.stale:
            flags.append("старое/обзор")
        lines = [f"📰 Новость для BTC: {mark}, сила {result.impact}/3" + (f" ({', '.join(flags)})" if flags else ""),
                 result.summary_ru or item.text[:200],
                 f"Источник: {item.source.split(':', 1)[1]}, мы увидели через {delay} с после публикации"
                 + (f"; BTC {price:.0f} USDT" if price else ""),
                 item.url]
        self._notify("\n".join(lines))
        self.store.mark_alerted(item.key)

    def _brake(self, result: Classification, now: datetime) -> None:
        until = now + BRAKE_FOR
        reason = f"тревожная новость: {result.summary_ru[:120]}"
        self.store.set_meta("brake_until", until.isoformat())
        self.store.set_meta("brake_reason", reason)
        self._notify(f"⛔ Тормоз: новые покупки ботов приостановлены до {until:%H:%M} UTC. {reason}")

    def update_prices(self) -> None:
        """Fill publication and horizon prices from 1m candles once they exist."""
        pending = self.store.directional_missing_prices(self.clock() - timedelta(hours=4))
        if not pending or self.minute_candles is None:
            return
        closes = minute_closes(self.minute_candles)
        if not closes:
            return
        for row in pending:
            published = datetime.fromisoformat(row["published_at"])
            seen = datetime.fromisoformat(row["seen_at"])
            prices: dict[str, Decimal | None] = {
                # close of the minute before publication = price the news met
                "price_pub": closes.get(_minute(published) - timedelta(minutes=1)),
            }
            for minutes in HORIZONS:
                prices[f"p{minutes}"] = closes.get(_minute(seen + timedelta(minutes=minutes)))
            self.store.set_prices(row["key"], **prices)


# --- paper statistics ----------------------------------------------------------

@dataclass(frozen=True)
class PaperStats:
    items: int
    alerts: int
    median_delay_s: int | None
    pre_move_pct: Decimal | None                       # move in our favour before we saw it
    horizons: dict[int, tuple[int, Decimal, Decimal]]  # minutes -> (n, avg net %, win rate %)


def paper_stats(rows: Sequence[sqlite3.Row], *, min_impact: int = ALERT_IMPACT) -> PaperStats:
    delays = sorted(int((datetime.fromisoformat(r["seen_at"]) - datetime.fromisoformat(r["published_at"]))
                        .total_seconds()) for r in rows)
    trades = [r for r in rows if r["relevant"] and r["confirmed"] and not r["stale"]
              and r["direction"] in ("bullish", "bearish") and (r["impact"] or 0) >= min_impact
              and r["price_seen"]]
    pre_moves = []
    horizons: dict[int, tuple[int, Decimal, Decimal]] = {}
    for minutes in HORIZONS:
        results = []
        for r in trades:
            if r[f"p{minutes}"]:
                sign = 1 if r["direction"] == "bullish" else -1
                gross = (Decimal(r[f"p{minutes}"]) / Decimal(r["price_seen"]) - 1) * sign
                results.append(gross - ROUND_TRIP_COST)
        if results:
            n = len(results)
            horizons[minutes] = (n, sum(results) / n * 100, Decimal(sum(1 for x in results if x > 0)) / n * 100)
    for r in trades:
        if r["price_pub"]:
            sign = 1 if r["direction"] == "bullish" else -1
            pre_moves.append((Decimal(r["price_seen"]) / Decimal(r["price_pub"]) - 1) * sign * 100)
    return PaperStats(len(rows), sum(1 for r in rows if r["alerted"]),
                      delays[len(delays) // 2] if delays else None,
                      sum(pre_moves) / len(pre_moves) if pre_moves else None, horizons)


def run_forever(radar: NewsRadar, *, on_cycle: Callable[[CycleResult], None] | None = None,
                sleeper: Callable[[float], None] = time.sleep) -> None:
    last_rss = 0.0
    while True:
        started = time.monotonic()
        do_rss = started - last_rss >= RSS_EVERY
        result = radar.poll(telegram=True, rss=do_rss)
        if do_rss:
            last_rss = started
        for error in result.errors:
            log.warning("Радар: %s", error)
        if result.new_items:
            log.info("Радар: новых %d, разобрано %d, оповещений %d",
                     result.new_items, result.classified, result.alerts)
        if on_cycle is not None:
            on_cycle(result)
        sleeper(max(1.0, TELEGRAM_EVERY - (time.monotonic() - started)))
