"""Optional owner alerts; delivery never controls the local safety state.

Only a local .env may supply credentials. No token, URL or transport exception
is returned to callers or written to the journal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from threading import Lock
from typing import Callable
from urllib.request import Request, urlopen

from config import ConfigError, _read_env_file


TEST_MESSAGE = "OKX Demo — тест аварийного уведомления. Торговля отключена"


def _post(request: Request, timeout: float) -> object:
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


@dataclass(frozen=True, repr=False)
class TelegramAlerter:
    token: str = field(repr=False)
    chat_id: str = field(repr=False)
    transport: Callable[[Request, float], object] = field(default=_post, repr=False)
    _test_lock: Lock = field(default_factory=Lock, init=False, repr=False, compare=False)
    _test_attempted: bool = field(default=False, init=False, repr=False, compare=False)

    @classmethod
    def from_env_file(cls, path: Path | None = None) -> TelegramAlerter | None:
        env_path = path or Path(__file__).resolve().parents[1] / ".env"
        if not env_path.is_file():
            return None
        try:
            values = _read_env_file(env_path)
        except (ConfigError, OSError, UnicodeError):
            return None
        if values.get("OKX_MODE") != "DEMO":
            return None
        token = values.get("OKX_TELEGRAM_BOT_TOKEN", "")
        chat = values.get("OKX_TELEGRAM_CHAT_ID", "")
        if not token and not chat:
            return None
        if not re.fullmatch(r"[0-9]{5,20}:[A-Za-z0-9_-]{20,}", token):
            return None
        if not re.fullmatch(r"-?[0-9]{1,20}", chat):
            return None
        return cls(token, chat)

    def notify(self, event: str) -> bool:
        messages = {
            "TEST": TEST_MESSAGE,
            "HALTED": "OKX Demo BTC-USDT: аварийная остановка сохранена. Новые сделки запрещены. Проверьте --recovery-check.",
            "UNPROTECTED": "OKX Demo BTC-USDT: BTC бота без подтверждённого SL. HALTED сохранён. Требуется ручная проверка Demo-счёта.",
            "MISMATCH": "OKX Demo BTC-USDT: данные биржи и SQLite расходятся. HALTED сохранён. Требуется ручная сверка.",
            "CONNECTION_LOST": "OKX Demo BTC-USDT: связь с Demo API потеряна. HALTED сохранён; исполнения неизвестны.",
            "CONNECTION_RESTORED": "OKX Demo BTC-USDT: связь восстановлена. HALTED остаётся; проверьте позицию и защиту вручную.",
            "WATCHDOG_API_DOWN": "OKX Watchdog Demo BTC-USDT: публичный или Demo GET недоступен. Состояние счёта неизвестно; проверьте сервис и OKX вручную.",
            "WATCHDOG_RESTORED": "OKX Watchdog Demo BTC-USDT: связь с публичным и Demo GET восстановлена. Принадлежность BTC и защита всё ещё неизвестны.",
            "WATCHDOG_CHANGED": "OKX Watchdog Demo BTC-USDT: баланс или список ордеров изменился. Проверьте виртуальный счёт вручную; защита не подтверждена.",
            "WATCHDOG_SUSPICIOUS": "OKX Watchdog Demo BTC-USDT: обнаружено подозрительное расхождение баланса или ордеров. Проверьте виртуальный счёт вручную; защита не подтверждена.",
            "HEALTH_STALE": "OKX Health Monitor: последнее успешное наблюдение Watchdog просрочено. Проверьте таймер и журнал; состояние Demo-счёта неизвестно.",
            "HEALTH_API_UNAVAILABLE": "OKX Health Monitor: последний цикл Watchdog сообщил об ошибке GET. Проверьте OKX и журнал; состояние Demo-счёта неизвестно.",
            "HEALTH_RESTORED": "OKX Health Monitor: новое успешное наблюдение Watchdog получено. Принадлежность BTC и защита не подтверждены; торговля отключена.",
        }
        if event not in messages:
            return False
        if event == "TEST":
            # Claim the attempt before transport so a timeout cannot trigger a
            # second send through this instance (even from another thread).
            with self._test_lock:
                if self._test_attempted:
                    return False
                object.__setattr__(self, "_test_attempted", True)
        return self._send(messages[event])

    def send_text(self, text: str) -> bool:
        """Free-form owner message built by the bot itself (never from API text)."""
        if not isinstance(text, str) or not text.strip() or len(text) > 3500:
            return False
        return self._send(text)

    def _send(self, text: str) -> bool:
        try:
            body = json.dumps({"chat_id": self.chat_id, "text": text},
                              ensure_ascii=False).encode("utf-8")
            request = Request(
                f"https://api.telegram.org/bot{self.token}/sendMessage", data=body,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            response = self.transport(request, 5.0)
            return isinstance(response, dict) and response.get("ok") is True
        except Exception:
            return False
