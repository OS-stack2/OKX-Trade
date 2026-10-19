"""Demo bot that buys BTC on strong, confirmed bullish news from the news radar.

It reuses DemoTrader's execution: market BUY, exchange OCO (SL -1% / TP +2%) on
the filled BTC, durable HALT on contradictions, recovery by clOrdId. What is
different: the trigger is a radar item instead of an EMA cross, the size is a
fixed share of its own book, and the position is closed after a fixed holding
time if the OCO has not fired.

It runs as a loop: the radar database is checked every couple of seconds so a
classified headline turns into an order with little extra delay. Only items the
radar classified while this bot was running are considered — never a backlog.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import sqlite3
import time

from bot.demo_client import DemoError
from bot.demo_trader import CycleReport, DemoTrader, OrderUncertain, TraderError, TraderSettings, TraderStore
from bot.market_data import MarketSnapshot
from bot.strategy import Signal, StrategyError, StrategyResult
from config import ConfigError, _read_env_file


DEFAULT_STATE_PATH = Path(__file__).resolve().parents[1] / "data" / "news_trader.sqlite3"
RESUME_PHRASE = "RESUME OKX DEMO NEWS TRADER"
MAX_SIGNAL_AGE = timedelta(minutes=3)     # radar saw it at most this long ago
RECONCILE_EVERY = timedelta(seconds=10)


@dataclass(frozen=True)
class NewsTraderSettings:
    trader: TraderSettings
    fraction: Decimal = Decimal("0.2")
    hold: timedelta = timedelta(minutes=30)
    min_impact: int = 2

    @classmethod
    def from_env_file(cls, path: Path | None = None) -> NewsTraderSettings:
        values = _read_env_file(path or Path(__file__).resolve().parents[1] / ".env")
        if values.get("OKX_NEWSTRADER_ENABLED") != "1":
            raise ConfigError("Новостной Demo-бот выключен: установите OKX_NEWSTRADER_ENABLED=1 в .env")
        try:
            budget = Decimal(values.get("OKX_NEWSTRADER_BUDGET_USDT", "1000"))
            fraction = Decimal(values.get("OKX_NEWSTRADER_FRACTION", "0.2"))
            hold = int(values.get("OKX_NEWSTRADER_HOLD_MINUTES", "30"))
            impact = int(values.get("OKX_NEWSTRADER_MIN_IMPACT", "2"))
        except (InvalidOperation, ValueError) as exc:
            raise ConfigError("Некорректные настройки OKX_NEWSTRADER_*") from exc
        if not (10 <= budget <= 100_000 and 0 < fraction <= 1 and 1 <= hold <= 1440 and impact in (2, 3)):
            raise ConfigError("OKX_NEWSTRADER_*: бюджет 10..100000, доля 0..1, удержание 1..1440 мин, сила 2 или 3")
        return cls(TraderSettings(budget_usdt=budget), fraction, timedelta(minutes=hold), impact)


def news_trader_state_path_from_env(path: Path | None = None) -> Path:
    values = _read_env_file(path or Path(__file__).resolve().parents[1] / ".env")
    raw = values.get("OKX_NEWSTRADER_STATE_PATH")
    return Path(raw) if raw else DEFAULT_STATE_PATH


@dataclass(frozen=True)
class NewsSignal:
    rowid: int
    seen_at: datetime
    summary: str
    impact: int
    source: str
    url: str
    key: str = ""
    text: str = ""
    published_at: datetime | None = None


class NewsFeed:
    """Read-only cursor over radar items, starting from 'now' on first use."""

    def __init__(self, radar_path: Path, store: TraderStore) -> None:
        self.radar_path = radar_path
        self.store = store

    def _query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        if not self.radar_path.is_file():
            return []
        with closing(sqlite3.connect(f"file:{self.radar_path}?mode=ro", uri=True, timeout=3)) as db:
            db.row_factory = sqlite3.Row
            try:
                return db.execute(sql, params).fetchall()
            except sqlite3.OperationalError:
                return []

    def next_signal(self, now: datetime, min_impact: int) -> NewsSignal | None:
        cursor = self.store.get_meta("news_cursor")
        if cursor is None:
            top = self._query("SELECT COALESCE(MAX(rowid), 0) AS m FROM items")
            self.store.set_meta("news_cursor", str(top[0]["m"] if top else 0))
            return None
        found = None
        last = int(cursor)
        for row in self._query("SELECT rowid AS rid, * FROM items WHERE rowid>? ORDER BY rowid", (last,)):
            if row["state"] == "NEW":
                break  # still being classified; look again next tick
            last = row["rid"]
            if (found is None and row["state"] == "CLASSIFIED" and row["relevant"] and row["confirmed"]
                    and not row["stale"] and row["direction"] == "bullish"
                    and (row["impact"] or 0) >= min_impact):
                seen = datetime.fromisoformat(row["seen_at"])
                if now - seen <= MAX_SIGNAL_AGE:
                    found = NewsSignal(row["rid"], seen, row["summary"] or "", row["impact"],
                                       row["source"], row["url"], row["key"], row["text"],
                                       datetime.fromisoformat(row["published_at"]))
        if last != int(cursor):
            self.store.set_meta("news_cursor", str(last))
        return found


class AnalystCases:
    """One row per headline sent to the analyst, with every decision it made."""

    MAX_CHECKS = 3

    def __init__(self, path: Path) -> None:
        self.path = path

    def _db(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=5, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("""CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT, news_rowid INTEGER NOT NULL UNIQUE,
            news_key TEXT NOT NULL, source TEXT NOT NULL, url TEXT NOT NULL, headline TEXT NOT NULL,
            summary TEXT NOT NULL, impact INTEGER NOT NULL, published_at TEXT NOT NULL,
            seen_at TEXT NOT NULL, state TEXT NOT NULL, checks INTEGER NOT NULL DEFAULT 0,
            next_check_at TEXT NOT NULL, hold_minutes INTEGER, position_id INTEGER,
            decisions TEXT NOT NULL DEFAULT '[]', note TEXT)""")
        return db

    def create(self, signal: NewsSignal, now: datetime) -> None:
        with closing(self._db()) as db:
            db.execute("INSERT OR IGNORE INTO cases(news_rowid, news_key, source, url, headline, summary, "
                       "impact, published_at, seen_at, state, next_check_at) "
                       "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?)",
                       (signal.rowid, signal.key, signal.source, signal.url, signal.text[:1500],
                        signal.summary, signal.impact,
                        (signal.published_at or signal.seen_at).isoformat(), signal.seen_at.isoformat(),
                        now.isoformat()))

    def due(self, now: datetime) -> sqlite3.Row | None:
        with closing(self._db()) as db:
            return db.execute("SELECT * FROM cases WHERE state='PENDING' AND next_check_at<=? "
                              "ORDER BY next_check_at LIMIT 1", (now.isoformat(),)).fetchone()

    def update(self, case_id: int, **fields: object) -> None:
        allowed = {"state", "checks", "next_check_at", "hold_minutes", "position_id", "decisions", "note"}
        if set(fields) - allowed:
            raise ValueError("Unsupported case update")
        with closing(self._db()) as db:
            db.execute(f"UPDATE cases SET {', '.join(f'{k}=?' for k in fields)} WHERE id=?",
                       (*fields.values(), case_id))

    def hold_for_position(self, position_id: int) -> int | None:
        with closing(self._db()) as db:
            row = db.execute("SELECT hold_minutes FROM cases WHERE position_id=?", (position_id,)).fetchone()
        return None if row is None else row["hold_minutes"]

    def all(self) -> list[sqlite3.Row]:
        with closing(self._db()) as db:
            return db.execute("SELECT * FROM cases ORDER BY id").fetchall()


class NewsTrader(DemoTrader):
    CLOSE_REASON = "TIME"
    RESUME_COMMAND = "--news-trade-resume"

    def __init__(self, *, feed: NewsFeed, news_settings: NewsTraderSettings,
                 analyst: object | None = None,
                 context_builder: Callable[[sqlite3.Row, datetime], object] | None = None,
                 cases: AnalystCases | None = None, **kwargs) -> None:
        super().__init__(settings=news_settings.trader, **kwargs)
        self.feed = feed
        self.news_settings = news_settings
        self.analyst = analyst
        self.context_builder = context_builder
        self.cases = cases if cases is not None else (AnalystCases(self.store.path) if analyst else None)
        self._next_reconcile = datetime.min.replace(tzinfo=timezone.utc)

    def _hold(self, position_id: int) -> timedelta:
        minutes = self.cases.hold_for_position(position_id) if self.cases is not None else None
        return timedelta(minutes=minutes) if minutes else self.news_settings.hold

    def _analyze(self, case: sqlite3.Row, now: datetime) -> CycleReport:
        """Ask the analyst about one pending headline and act on the answer."""
        from bot.news_analyst import BUY_NOW, SKIP, WAIT, describe_ru
        from bot.news_radar import NewsError
        decisions = json.loads(case["decisions"])
        checks = case["checks"] + 1
        if self.store.active_position() is not None:
            self.cases.update(case["id"], state="BLOCKED", checks=checks, note="позиция уже открыта")
            self._say(f"Новость «{case['summary']}» не разбирается: позиция уже открыта")
            return CycleReport("HOLD", tuple(self._lines))
        try:
            context = self.context_builder(case, now)
            previous = tuple(f"{d['at'][11:16]} UTC: {d['decision']} ({d['confidence']}%)" for d in decisions)
            decision = self.analyst.evaluate(headline=case["headline"], summary=case["summary"],
                                             source=case["source"], impact=case["impact"],
                                             context=context, previous=previous)
        except NewsError as exc:
            final = checks >= AnalystCases.MAX_CHECKS
            self.cases.update(case["id"], checks=checks, state="EXPIRED" if final else "PENDING",
                              next_check_at=(now + timedelta(minutes=2)).isoformat(), note=str(exc))
            self._say(f"Аналитик недоступен: {exc}")
            return CycleReport("ERROR", tuple(self._lines))
        decisions.append({"at": now.isoformat(), "decision": decision.decision,
                          "confidence": decision.confidence, "flags": list(decision.flags),
                          "reasons": list(decision.reasons_ru), "vetoed": decision.vetoed,
                          "price": context.price, "context": context.short_ru()})
        record = json.dumps(decisions, ensure_ascii=False)
        header = f"📰 «{case['summary']}» ({case['source'].split(':', 1)[1]})\n{describe_ru(decision)}\n" \
                 f"Рынок: {context.short_ru()}"
        self._say(f"{decision.decision} ({decision.confidence}%): {case['summary']}")
        if decision.decision == BUY_NOW:
            market = self.market_provider()
            trigger = datetime.fromisoformat(case["seen_at"]) + timedelta(microseconds=case["news_rowid"] % 1_000_000)
            result = StrategyResult(Signal.BUY, Decimal(0), Decimal(0), Decimal(0), Decimal(0),
                                    trigger, case["summary"])
            self._notify(header + f"\nПокупаю, держу до {decision.hold_minutes} мин.\n{case['url']}")
            status = self._open(market, result)
            position = self.store.active_position()
            if status == "OPENED" and position is not None:
                self.cases.update(case["id"], state="BOUGHT", checks=checks, decisions=record,
                                  hold_minutes=decision.hold_minutes, position_id=position.id)
            else:
                self.cases.update(case["id"], state="BLOCKED", checks=checks, decisions=record,
                                  note=self._lines[-1] if self._lines else status)
            self._next_reconcile = now
            return CycleReport(status, tuple(self._lines))
        if decision.decision == WAIT and checks < AnalystCases.MAX_CHECKS:
            when = now + timedelta(minutes=decision.recheck_minutes)
            self.cases.update(case["id"], checks=checks, decisions=record, next_check_at=when.isoformat())
            if checks == 1:
                self._notify(header + f"\nПерепроверю в {when:%H:%M} UTC.\n{case['url']}")
            return CycleReport("HOLD", tuple(self._lines))
        state = "SKIPPED" if decision.decision == SKIP else "EXPIRED"
        self.cases.update(case["id"], state=state, checks=checks, decisions=record)
        tail = "Не покупаю." if state == "SKIPPED" else f"После {checks} проверок момент так и не наступил — не покупаю."
        self._notify(header + f"\n{tail}\n{case['url']}")
        return CycleReport("HOLD", tuple(self._lines))

    def _notify(self, text: str) -> None:
        if self.notifier is not None:
            try:
                self.notifier("OKX Demo новостной бот: " + text)
            except Exception:
                pass

    def _size(self, *, price: Decimal, fee: Decimal, equity: Decimal, risk_left: Decimal) -> Decimal:
        return equity * self.news_settings.fraction / (price * (1 + fee + self.settings.slippage))

    def step(self) -> CycleReport | None:
        """One loop tick. Returns a report only when something happened."""
        self._lines = []
        if self.store.get_meta("halted"):
            return None
        try:
            now = self.clock()
            closed = False
            if now >= self._next_reconcile:
                self._next_reconcile = now + RECONCILE_EVERY
                closed = self._reconcile()
                position = self.store.active_position()
                if (position is not None and position.state == "OPEN"
                        and now - datetime.fromisoformat(position.opened_at) >= self._hold(position.id)):
                    self._close_by_signal(position)
                    closed = True
            signal = self.feed.next_signal(now, self.news_settings.min_impact)
            if self.analyst is not None:
                if signal is not None:
                    self.cases.create(signal, now)
                case = self.cases.due(now)
                if case is not None:
                    return self._analyze(case, now)
                return CycleReport("CLOSED", tuple(self._lines)) if closed else None
            if signal is None:
                return CycleReport("CLOSED", tuple(self._lines)) if closed else None
            self._say(f"Новость (сила {signal.impact}, {signal.source}): {signal.summary}")
            if self.store.active_position() is not None:
                self._say("BUY пропущен: позиция уже открыта")
                return CycleReport("HOLD", tuple(self._lines))
            market = self.market_provider()
            # A unique per-news key for the journal's one-entry-per-trigger rule.
            trigger = signal.seen_at + timedelta(microseconds=signal.rowid % 1_000_000)
            result = StrategyResult(Signal.BUY, Decimal(0), Decimal(0), Decimal(0), Decimal(0),
                                    trigger, signal.summary)
            status = self._open(market, result)
            self._next_reconcile = now
            return CycleReport(status, tuple(self._lines))
        except TraderError as exc:
            return self._halt(str(exc))
        except OrderUncertain as exc:
            return self._halt(f"Неопределённый результат торгового запроса: {exc}")
        except (DemoError, StrategyError, ConfigError, ValueError, OSError) as exc:
            self._say(f"Тик пропущен: {exc}")
            return CycleReport("ERROR", tuple(self._lines))


def run_forever(trader: NewsTrader, *, on_report: Callable[[CycleReport], None],
                on_heartbeat: Callable[[bool], None], tick: float = 2.0,
                sleeper: Callable[[float], None] = time.sleep) -> None:
    last_beat = 0.0
    errors = 0
    while True:
        report = trader.step()
        if report is not None:
            on_report(report)
            errors = errors + 1 if report.status == "ERROR" else 0
        if time.monotonic() - last_beat >= 60:
            on_heartbeat(errors == 0)
            last_beat = time.monotonic()
        sleeper(tick)


def cached_market(instrument_reader: Callable[[], object], price_reader: Callable[[], Decimal],
                  clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
                  ) -> Callable[[], MarketSnapshot]:
    """Instrument limits change rarely: refresh hourly, read only the ticker per trade."""
    cache: dict[str, object] = {}

    def provider() -> MarketSnapshot:
        now = clock()
        if "instrument" not in cache or now - cache["at"] > timedelta(hours=1):  # type: ignore[operator]
            cache["instrument"], cache["at"] = instrument_reader(), now
        return MarketSnapshot(cache["instrument"], price_reader(), (), now)  # type: ignore[arg-type]
    return provider


def analyst_summary(cases: list[sqlite3.Row], positions: dict[int, Decimal | None],
                    radar_path: Path) -> list[str]:
    """Analyst vs. 'buy every bullish headline': real results and counterfactuals."""
    if not cases:
        return ["  аналитику пока не попадалось подходящих новостей"]
    counts: dict[str, int] = {}
    for case in cases:
        counts[case["state"]] = counts.get(case["state"], 0) + 1
    labels = {"BOUGHT": "купил", "SKIPPED": "пропустил", "EXPIRED": "ждал и не купил",
              "BLOCKED": "не смог (позиция/лимит/тормоз)", "PENDING": "в работе"}
    lines = ["  решения: " + ", ".join(f"{labels.get(k, k)} {v}" for k, v in sorted(counts.items()))]
    bought = [positions.get(c["position_id"]) for c in cases if c["state"] == "BOUGHT"]
    realized = [p for p in bought if p is not None]
    if realized:
        lines.append(f"  купленные: {len(realized)} сделок, PnL {sum(realized):+.2f} USDT")
    passed = [c["news_key"] for c in cases if c["state"] in ("SKIPPED", "EXPIRED")]
    if passed and radar_path.is_file():
        with closing(sqlite3.connect(f"file:{radar_path}?mode=ro", uri=True, timeout=3)) as db:
            rows = db.execute(f"SELECT price_seen, p30, p60 FROM items WHERE key IN "
                              f"({','.join('?' * len(passed))})", passed).fetchall()
        for column, label in ((1, "30"), (2, "60")):
            moves = [(float(r[column]) / float(r[0]) - 1) * 100 - 0.2 for r in rows if r[0] and r[column]]
            if moves:
                lines.append(f"  если бы купили пропущенные сразу, через {label} мин: в среднем "
                             f"{sum(moves) / len(moves):+.3f}% (прибыльных {sum(m > 0 for m in moves)} из {len(moves)})")
    return lines
