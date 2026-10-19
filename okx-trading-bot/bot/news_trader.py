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
                                       row["source"], row["url"])
        if last != int(cursor):
            self.store.set_meta("news_cursor", str(last))
        return found


class NewsTrader(DemoTrader):
    CLOSE_REASON = "TIME"
    RESUME_COMMAND = "--news-trade-resume"

    def __init__(self, *, feed: NewsFeed, news_settings: NewsTraderSettings, **kwargs) -> None:
        super().__init__(settings=news_settings.trader, **kwargs)
        self.feed = feed
        self.news_settings = news_settings
        self._next_reconcile = datetime.min.replace(tzinfo=timezone.utc)

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
                        and now - datetime.fromisoformat(position.opened_at) >= self.news_settings.hold):
                    self._close_by_signal(position)
                    closed = True
            signal = self.feed.next_signal(now, self.news_settings.min_impact)
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
