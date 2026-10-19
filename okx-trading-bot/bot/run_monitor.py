"""Silent-failure alerts for the Demo bots.

Two layers:
* every bot run records success/failure; the 3rd consecutive failure sends one
  Telegram alert, the next success sends one "restored" message;
* `python -m bot.run_monitor` (own timer) alerts when a bot with an enabled
  systemd timer has had no successful run for too long — this catches crashes
  before the bot code even starts.

The module imports only the standard library, config and Telegram on purpose,
so a broken trading module cannot also break the staleness check.
"""

from __future__ import annotations

from contextlib import closing, contextmanager
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from typing import Callable, Iterator

from config import _read_env_file


FAILURES_BEFORE_ALERT = 3
DEFAULT_STATE_PATH = Path(__file__).resolve().parents[1] / "data" / "run_monitor.sqlite3"
# name -> (systemd timer, maximum age of the last successful run, label)
BOTS = {
    "demo-trader": ("okx-demo-trader.timer", timedelta(minutes=30), "EMA Demo-трейдер"),
    "demo-trend": ("okx-demo-trend.timer", timedelta(hours=3), "Тренд-бот"),
    "news-radar": ("okx-news-radar.service", timedelta(minutes=10), "Новостной радар"),
}


def monitor_path_from_env(path: Path | None = None) -> Path:
    values = _read_env_file(path or Path(__file__).resolve().parents[1] / ".env")
    raw = values.get("OKX_RUNMON_STATE_PATH")
    return Path(raw) if raw else DEFAULT_STATE_PATH


class RunMonitor:
    def __init__(self, path: Path = DEFAULT_STATE_PATH) -> None:
        self.path = path

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=5, isolation_level=None)) as db:
            db.row_factory = sqlite3.Row
            db.execute("BEGIN IMMEDIATE")
            try:
                db.execute("""CREATE TABLE IF NOT EXISTS runs (
                    name TEXT PRIMARY KEY, last_ok TEXT, last_run TEXT, failures INTEGER NOT NULL DEFAULT 0,
                    alerted INTEGER NOT NULL DEFAULT 0, stale_alerted INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT)""")
                yield db
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    def _row(self, db: sqlite3.Connection, name: str) -> sqlite3.Row:
        db.execute("INSERT OR IGNORE INTO runs(name) VALUES(?)", (name,))
        return db.execute("SELECT * FROM runs WHERE name=?", (name,)).fetchone()

    def record(self, name: str, ok: bool, now: datetime, detail: str = "") -> str | None:
        """Store one run outcome; return a message to send, if any."""
        label = BOTS.get(name, (None, None, name))[2]
        with self._db() as db:
            row = self._row(db, name)
            if ok:
                was_alerted = row["alerted"] or row["stale_alerted"]
                db.execute("UPDATE runs SET last_ok=?, last_run=?, failures=0, alerted=0, stale_alerted=0, "
                           "last_error=NULL WHERE name=?", (now.isoformat(), now.isoformat(), name))
                return f"{label}: снова работает нормально." if was_alerted else None
            failures = row["failures"] + 1
            alert = failures >= FAILURES_BEFORE_ALERT and not row["alerted"]
            db.execute("UPDATE runs SET last_run=?, failures=?, alerted=?, last_error=? WHERE name=?",
                       (now.isoformat(), failures, 1 if alert or row["alerted"] else 0, detail[:300], name))
        if alert:
            return (f"{label}: {failures} неудачных запусков подряд. Последняя ошибка: {detail[:300] or 'нет текста'}. "
                    "Бот не торгует, пока ошибка не исчезнет; проверьте journalctl.")
        return None

    def stale(self, name: str, max_age: timedelta, now: datetime) -> str | None:
        label = BOTS.get(name, (None, None, name))[2]
        with self._db() as db:
            row = self._row(db, name)
            reference = row["last_ok"] or row["last_run"]
            if reference is None:
                db.execute("UPDATE runs SET last_run=? WHERE name=?", (now.isoformat(), name))
                return None  # first sighting: start the clock
            age = now - datetime.fromisoformat(reference)
            if age <= max_age or row["stale_alerted"]:
                return None
            db.execute("UPDATE runs SET stale_alerted=1 WHERE name=?", (name,))
        hours = age.total_seconds() / 3600
        return (f"{label}: нет ни одного успешного запуска {hours:.1f} ч. Возможно, сервис падает при старте "
                "или таймер остановлен. Проверьте systemctl status и journalctl.")


def _timer_enabled(timer: str) -> bool:
    try:
        result = subprocess.run(["systemctl", "is-enabled", timer], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.stdout.strip() == "enabled"


def check_stale(monitor: RunMonitor, now: datetime, send: Callable[[str], object] | None,
                timer_enabled: Callable[[str], bool] = _timer_enabled) -> list[str]:
    messages = []
    for name, (timer, max_age, _label) in BOTS.items():
        if not timer_enabled(timer):
            continue
        message = monitor.stale(name, max_age, now)
        if message:
            messages.append(message)
            if send is not None:
                send("OKX Demo: " + message)
    return messages


def main() -> int:
    from bot.telegram_alerts import TelegramAlerter
    try:
        monitor = RunMonitor(monitor_path_from_env())
        alerter = TelegramAlerter.from_env_file()
        messages = check_stale(monitor, datetime.now(timezone.utc),
                               alerter.send_text if alerter is not None else None)
    except (OSError, sqlite3.Error, ValueError) as exc:
        print(f"Run monitor: BLOCK — {exc}", file=sys.stderr)
        return 1
    print("Run monitor: " + ("; ".join(messages) if messages else "все боты работают"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
