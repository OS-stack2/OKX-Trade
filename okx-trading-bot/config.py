"""Configuration for public checks and the read-only OKX Demo integration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
import os


REGION_BASE_URLS = {
    "global": "https://www.okx.com",
    "eea": "https://eea.okx.com",
    "us": "https://us.okx.com",
}


class ConfigError(ValueError):
    """The local configuration is invalid."""


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    result: dict[str, str] = {}
    for number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ConfigError(f"Invalid .env line {number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key.startswith("OKX_"):
            raise ConfigError(f"Invalid .env line {number}: only OKX_* names are supported")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        result[key] = value
    return result


@dataclass(frozen=True)
class Config:
    region: str = "global"
    mode: str = "READ_ONLY"
    timeout_seconds: float = 5.0
    max_attempts: int = 3

    def __post_init__(self) -> None:
        if self.region not in REGION_BASE_URLS:
            raise ConfigError("OKX_REGION must be global, eea, or us")
        if self.mode not in {"READ_ONLY", "DEMO"}:
            raise ConfigError("Only READ_ONLY and DEMO modes are allowed; LIVE is forbidden")
        if not 0 < self.timeout_seconds <= 60:
            raise ConfigError("OKX_TIMEOUT_SECONDS must be between 0 and 60")
        if not 1 <= self.max_attempts <= 5:
            raise ConfigError("OKX_MAX_ATTEMPTS must be between 1 and 5")

    @property
    def base_url(self) -> str:
        return REGION_BASE_URLS[self.region]

    @classmethod
    def from_environment(
        cls, env_file: Path | None = None, environ: Mapping[str, str] | None = None
    ) -> Config:
        values = _read_env_file(env_file or Path(__file__).resolve().parent / ".env")
        values.update(environ if environ is not None else os.environ)
        try:
            timeout = float(values.get("OKX_TIMEOUT_SECONDS", "5"))
            attempts = int(values.get("OKX_MAX_ATTEMPTS", "3"))
        except ValueError as exc:
            raise ConfigError("Invalid OKX timeout or attempt count") from exc
        return cls(
            region=values.get("OKX_REGION", "global").lower(),
            mode=values.get("OKX_MODE", "READ_ONLY").upper(),
            timeout_seconds=timeout,
            max_attempts=attempts,
        )


def demo_request_headers(config: Config) -> dict[str, str]:
    """Mandatory header for private Demo requests."""
    if config.mode != "DEMO":
        raise ConfigError("Private requests require DEMO mode")
    return {"x-simulated-trading": "1"}
