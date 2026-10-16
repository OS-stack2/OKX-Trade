"""Offline BTC-USDT SPOT research. No exchange, execution or SQLite imports.

Signals use only candles closed at decision time and fill at the *next* open.
This is a conservative bar simulator, not evidence of executable OKX protection.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from pathlib import Path

from bot.market_data import Candle, SpotInstrument
from bot.risk_manager import RiskSettings


D = Decimal
ZERO = D(0)
ONE = D(1)
VARIANTS = ("EMA_9_21", "EMA_TREND", "EMA_VOLATILITY", "EMA_STRENGTH", "EMA_ALL_FILTERS")
MIN_BARS = 100
MAX_CSV_BYTES = 64 * 1024 * 1024


class ResearchError(ValueError):
    """Historical data or offline assumptions are invalid."""


@dataclass(frozen=True)
class ResearchConfig:
    instrument: SpotInstrument
    taker_fee_rate: Decimal
    slippage_rate: Decimal
    initial_usdt: Decimal = D("5000")  # Virtual research capital, never read from budget SQLite.
    max_position_fraction: Decimal = D("0.5")
    max_position_usdt: Decimal = D("5000")
    risk: RiskSettings = RiskSettings()

    def validate(self) -> None:
        instrument = self.instrument
        if (not isinstance(instrument, SpotInstrument) or instrument.instrument_id != "BTC-USDT"
                or instrument.base_currency not in (None, "BTC")
                or instrument.quote_currency not in (None, "USDT")):
            raise ResearchError("Требуются подтверждённые ограничения BTC-USDT SPOT")
        for value in (instrument.min_size, instrument.lot_size, instrument.tick_size,
                      instrument.max_market_size, self.initial_usdt, self.max_position_usdt):
            if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
                raise ResearchError("Ограничение инструмента или капитала неизвестно")
        if instrument.max_market_size < instrument.min_size:
            raise ResearchError("maxMktSz меньше minSz")
        for value in (self.taker_fee_rate, self.slippage_rate, self.max_position_fraction):
            if not isinstance(value, Decimal) or not value.is_finite() or value < 0 or value >= 1:
                raise ResearchError("Комиссия, проскальзывание или доля позиции некорректны")
        if self.max_position_fraction == 0 or not isinstance(self.risk, RiskSettings) or self.risk != RiskSettings():
            raise ResearchError("Лимиты риска должны совпадать с утверждёнными")


@dataclass(frozen=True)
class Metrics:
    variant: str
    period: str
    bars: int
    final_usdt: Decimal
    return_pct: Decimal
    max_drawdown_pct: Decimal
    profit_factor: Decimal | None
    trades: int
    entries: int
    wins: int
    buy_hold_return_pct: Decimal | None
    relative_buy_hold_pct: Decimal | None
    open_position_at_end: bool
    entry_indices: tuple[int, ...]
    entry_quantities: tuple[Decimal, ...]
    max_entry_notional_usdt: Decimal
    skipped_min_size: int
    blocked_daily_limit: int
    blocked_loss_limit: int
    blocked_entry_limit: int


@dataclass(frozen=True)
class StressResult:
    name: str
    return_pct: Decimal | None
    max_drawdown_pct: Decimal | None
    note: str


@dataclass(frozen=True)
class ResearchReport:
    split_indices: tuple[int, int, int, int]
    training: tuple[Metrics, ...]
    validation: tuple[Metrics, ...]
    selected_variant: str
    final_test: Metrics
    stresses: tuple[StressResult, ...]
    recommendation: str
    warnings: tuple[str, ...]


def _decimal(value: str) -> Decimal:
    try:
        result = D(value)
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ResearchError("CSV содержит некорректное число") from exc
    if not result.is_finite() or result <= 0:
        raise ResearchError("CSV содержит некорректную цену")
    return result


def validate_candles(candles: tuple[Candle, ...]) -> None:
    if len(candles) < MIN_BARS:
        raise ResearchError(f"Нужно минимум {MIN_BARS} закрытых 15m свечей для расчёта")
    previous = None
    for candle in candles:
        if not isinstance(candle, Candle) or candle.confirmed is not True:
            raise ResearchError("Историческая свеча не подтверждена")
        at = candle.opened_at
        if (not isinstance(at, datetime) or at.tzinfo is None or at.utcoffset() != timedelta(0)
                or at.minute % 15 or at.second or at.microsecond
                or (previous is not None and at - previous != timedelta(minutes=15))):
            raise ResearchError("15m свечи должны быть непрерывными, UTC и отсортированными")
        prices = (candle.open, candle.high, candle.low, candle.close)
        if any(not isinstance(p, Decimal) or not p.is_finite() or p <= 0 for p in prices):
            raise ResearchError("Свеча содержит некорректную цену")
        if candle.high < max(candle.open, candle.close) or candle.low > min(candle.open, candle.close) or candle.low > candle.high:
            raise ResearchError("OHLC свечи противоречив")
        previous = at


def load_candles_csv(path: Path) -> tuple[Candle, ...]:
    """CSV: opened_at (ISO UTC), open, high, low, close, confirm=1."""
    try:
        if not path.is_file() or path.stat().st_size > MAX_CSV_BYTES:
            raise ResearchError("Исторический CSV отсутствует или слишком велик")
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["opened_at", "open", "high", "low", "close", "confirm"]:
                raise ResearchError("CSV требует столбцы opened_at,open,high,low,close,confirm")
            candles = []
            for row in reader:
                if row.get("confirm") != "1" or None in row or any(value is None for value in row.values()):
                    raise ResearchError("CSV содержит открытую или неполную свечу")
                try:
                    at = datetime.fromisoformat(row["opened_at"].replace("Z", "+00:00"))
                except (TypeError, ValueError) as exc:
                    raise ResearchError("CSV содержит некорректное UTC-время") from exc
                candles.append(Candle(at, *(_decimal(row[key]) for key in ("open", "high", "low", "close")), True))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ResearchError("Не удалось безопасно прочитать исторический CSV") from exc
    result = tuple(candles)
    validate_candles(result)
    return result


@dataclass(frozen=True)
class _Indicators:
    ema9: tuple[Decimal | None, ...]
    ema21: tuple[Decimal | None, ...]
    ema50: tuple[Decimal | None, ...]
    atr14: tuple[Decimal | None, ...]


def _ema(values: tuple[Decimal, ...], period: int) -> tuple[Decimal | None, ...]:
    result: list[Decimal | None] = [None] * len(values)
    ema = sum(values[:period], ZERO) / D(period)
    result[period - 1] = ema
    multiplier = D(2) / D(period + 1)
    for index in range(period, len(values)):
        ema += multiplier * (values[index] - ema)
        result[index] = ema
    return tuple(result)


def _indicators(candles: tuple[Candle, ...]) -> _Indicators:
    closes = tuple(c.close for c in candles)
    tr = tuple(max(c.high - c.low, abs(c.high - candles[i - 1].close),
                   abs(c.low - candles[i - 1].close)) if i else c.high - c.low
               for i, c in enumerate(candles))
    atr: list[Decimal | None] = [None] * len(candles)
    value = sum(tr[:14], ZERO) / D(14)
    atr[13] = value
    for index in range(14, len(candles)):
        value = (value * 13 + tr[index]) / 14
        atr[index] = value
    return _Indicators(_ema(closes, 9), _ema(closes, 21), _ema(closes, 50), tuple(atr))


def _signal_at(index: int, candles: tuple[Candle, ...], ind: _Indicators, variant: str) -> str:
    if index < 21:
        return "HOLD"
    short, long = ind.ema9[index], ind.ema21[index]
    previous_short, previous_long = ind.ema9[index - 1], ind.ema21[index - 1]
    assert short is not None and long is not None and previous_short is not None and previous_long is not None
    if previous_short >= previous_long and short < long:
        return "SELL"
    if not (previous_short <= previous_long and short > long):
        return "HOLD"
    if variant not in VARIANTS:
        raise ResearchError("Неизвестный вариант EMA")
    if variant in ("EMA_TREND", "EMA_ALL_FILTERS"):
        trend = ind.ema50[index]
        old_trend = ind.ema50[index - 4] if index >= 53 else None
        if trend is None or old_trend is None or not (candles[index].close > trend > old_trend):
            return "HOLD"
    if variant in ("EMA_VOLATILITY", "EMA_ALL_FILTERS"):
        atr = ind.atr14[index]
        if atr is None or not D("0.0015") <= atr / candles[index].close <= D("0.03"):
            return "HOLD"
    if variant in ("EMA_STRENGTH", "EMA_ALL_FILTERS"):
        atr = ind.atr14[index]
        if atr is None or atr <= 0 or (short - long) / atr < D("0.1"):
            return "HOLD"
    return "BUY"


def _floor(value: Decimal, step: Decimal) -> Decimal:
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def _buy_hold(candles: tuple[Candle, ...], start: int, end: int, config: ResearchConfig) -> Decimal | None:
    entry = candles[start + 1].open * (ONE + config.slippage_rate)
    quantity = _floor(config.initial_usdt / (entry * (ONE + config.taker_fee_rate)),
                      config.instrument.lot_size)
    if quantity < config.instrument.min_size:
        return None
    remainder = config.initial_usdt - quantity * entry * (ONE + config.taker_fee_rate)
    final = remainder + quantity * candles[end - 1].close * (ONE - config.slippage_rate) * (ONE - config.taker_fee_rate)
    return (final / config.initial_usdt - ONE) * 100


def run_backtest(candles: tuple[Candle, ...], start: int, end: int, variant: str,
                 config: ResearchConfig, indicators: _Indicators | None = None,
                 period: str = "offline") -> Metrics:
    """A close-i signal can only fill at open-(i+1), inside this period."""
    config.validate()
    validate_candles(candles)
    if variant not in VARIANTS or not 0 <= start < end <= len(candles) or end - start < 2:
        raise ResearchError("Некорректный вариант или период исследования")
    ind = indicators or _indicators(candles)
    fee, slip = config.taker_fee_rate, config.slippage_rate
    cash = config.initial_usdt
    quantity = ZERO
    entry_cost = ZERO
    stop = take = ZERO
    gross_wins = gross_losses = ZERO
    entries = wins = trades = skipped = blocked_daily = blocked_loss = blocked_entries = 0
    entry_indices: list[int] = []
    entry_quantities: list[Decimal] = []
    max_entry_notional = ZERO
    pending = "HOLD"
    current_day = None
    day_start = config.initial_usdt
    realized_losses_today = ZERO
    entries_today = 0
    loss_locked = False
    peak = config.initial_usdt
    max_drawdown = ZERO

    def sell(price: Decimal) -> None:
        nonlocal cash, quantity, entry_cost, gross_wins, gross_losses, wins, trades, realized_losses_today
        proceeds = quantity * price * (ONE - fee)
        pnl = proceeds - entry_cost
        cash += proceeds
        gross_wins += max(ZERO, pnl)
        gross_losses += max(ZERO, -pnl)
        realized_losses_today += max(ZERO, -pnl)
        wins += int(pnl > 0)
        trades += 1
        quantity = entry_cost = ZERO

    for index in range(start, end):
        candle = candles[index]
        if candle.opened_at.date() != current_day:
            current_day = candle.opened_at.date()
            day_start = cash + quantity * candle.open * (ONE - slip) * (ONE - fee)
            realized_losses_today = ZERO
            entries_today = 0
            loss_locked = False
        if pending == "SELL" and quantity:
            sell(candle.open * (ONE - slip))
        if pending == "BUY" and not quantity:
            if loss_locked or realized_losses_today >= day_start * config.risk.daily_loss_limit_pct:
                blocked_loss += 1
                blocked_daily += 1
            elif entries_today >= config.risk.max_new_entries_per_day:
                blocked_entries += 1
                blocked_daily += 1
            else:
                entry = candle.open * (ONE + slip)
                planned_stop = _floor(entry * (ONE - config.risk.stop_loss_pct), config.instrument.tick_size)
                planned_take = _floor(entry * (ONE + config.risk.take_profit_pct), config.instrument.tick_size)
                stop_fill = planned_stop * (ONE - slip)
                risk_per_btc = entry * (ONE + fee) - stop_fill * (ONE - fee)
                risk_budget = min(cash * config.risk.risk_per_trade,
                                  day_start * config.risk.daily_loss_limit_pct - realized_losses_today)
                if risk_per_btc > 0 and planned_stop > 0 and planned_take > entry and risk_budget > 0:
                    raw = min(risk_budget / risk_per_btc, cash / (entry * (ONE + fee)),
                              cash * config.max_position_fraction / entry,
                              config.max_position_usdt / entry, config.instrument.max_market_size)
                    planned = _floor(raw, config.instrument.lot_size)
                    if planned < config.instrument.min_size:
                        skipped += 1
                    else:
                        quantity = planned
                        entry_cost = quantity * entry * (ONE + fee)
                        cash -= entry_cost
                        stop, take = planned_stop, planned_take
                        entries += 1
                        entries_today += 1
                        entry_indices.append(index)
                        entry_quantities.append(quantity)
                        max_entry_notional = max(max_entry_notional, quantity * entry)
                else:
                    skipped += 1
        pending = "HOLD"
        if quantity:
            # A bar containing both triggers exits at the stop; a gap below it
            # fills at the worse open. This can exceed the planned risk budget.
            if candle.open <= stop or candle.low <= stop:
                sell(min(candle.open, stop) * (ONE - slip))
            elif candle.high >= take:
                sell(take * (ONE - slip))
        marked = cash + quantity * candle.close * (ONE - slip) * (ONE - fee)
        unrealized_loss = max(ZERO, entry_cost - quantity * candle.close * (ONE - slip) * (ONE - fee)) if quantity else ZERO
        if realized_losses_today + unrealized_loss >= day_start * config.risk.daily_loss_limit_pct:
            loss_locked = True
        peak = max(peak, marked)
        if peak > 0:
            max_drawdown = max(max_drawdown, (ONE - marked / peak) * 100)
        if index + 1 < end:
            signal = _signal_at(index, candles, ind, variant)
            if signal == "SELL" and quantity:
                pending = "SELL"
            elif signal == "BUY" and not quantity:
                pending = "BUY"
    final = cash + quantity * candles[end - 1].close * (ONE - slip) * (ONE - fee)
    buy_hold = _buy_hold(candles, start, end, config)
    pct = (final / config.initial_usdt - ONE) * 100
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else None
    return Metrics(variant, period, end - start, final, pct, max_drawdown,
                   profit_factor, trades, entries, wins, buy_hold,
                   pct - buy_hold if buy_hold is not None else None,
                   quantity > 0, tuple(entry_indices), tuple(entry_quantities),
                   max_entry_notional, skipped, blocked_daily, blocked_loss, blocked_entries)


def _loss_streak(config: ResearchConfig) -> StressResult:
    equity = config.initial_usdt
    day_start = equity
    losses = ZERO
    count = 0
    for _ in range(10):
        if losses >= day_start * config.risk.daily_loss_limit_pct or count >= config.risk.max_new_entries_per_day:
            break
        loss = min(equity * config.risk.risk_per_trade,
                   day_start * config.risk.daily_loss_limit_pct - losses)
        equity -= loss
        losses += loss
        count += 1
    return StressResult("10 последовательных стопов за сутки (модель)",
                        (equity / day_start - ONE) * 100, None,
                        f"Разрешено входов: {count}; затем дневной лимит запрещает новые входы. "
                        "Реальный гэп может превысить модельный стоп.")


def research(candles: tuple[Candle, ...], config: ResearchConfig) -> ResearchReport:
    config.validate()
    validate_candles(candles)
    n = len(candles)
    train_end, validation_end = n * 3 // 5, n * 4 // 5
    ind = _indicators(candles)
    training = tuple(run_backtest(candles, 0, train_end, name, config, ind, "TRAIN") for name in VARIANTS)
    validation = tuple(run_backtest(candles, train_end, validation_end, name, config, ind, "VALIDATION") for name in VARIANTS)
    # Select with TRAIN/VALIDATION only. Final holdout is evaluated once below.
    ranked = sorted(zip(training, validation),
                    key=lambda pair: (D("0.4") * (pair[0].return_pct - pair[0].max_drawdown_pct)
                                      + D("0.6") * (pair[1].return_pct - pair[1].max_drawdown_pct),
                                      pair[1].return_pct, -VARIANTS.index(pair[0].variant)), reverse=True)
    selected = ranked[0][0].variant
    final = run_backtest(candles, validation_end, n, selected, config, ind, "FINAL_TEST")
    warnings = []
    if n < 17280 or train_end < 8640 or validation_end - train_end < 2880 or n - validation_end < 2880:
        warnings.append("История короче примерно шести месяцев 15m свечей; статистической мощности недостаточно")
    chosen_train = next(item for item in training if item.variant == selected)
    chosen_val = next(item for item in validation if item.variant == selected)
    if chosen_train.trades < 50 or chosen_val.trades < 20 or final.trades < 20:
        warnings.append("Слишком мало закрытых сделок (нужно минимум 50/20/20 TRAIN/VALIDATION/FINAL_TEST)")
    if final.open_position_at_end:
        warnings.append("В финале осталась открытая модельная позиция; доходность помечена к рынку")
    if final.profit_factor is None:
        warnings.append("Profit factor финального периода не определён без убыточных закрытых сделок")
    stresses = []
    for name, fee, slip in (("Комиссия x2", config.taker_fee_rate * 2, config.slippage_rate),
                            ("Проскальзывание x3", config.taker_fee_rate, config.slippage_rate * 3)):
        if fee >= 1 or slip >= 1:
            stresses.append(StressResult(name, None, None, "Шоковый параметр вне допустимого диапазона"))
        else:
            stressed = run_backtest(candles, validation_end, n, selected,
                                    replace(config, taker_fee_rate=fee, slippage_rate=slip), ind, name)
            stresses.append(StressResult(name, stressed.return_pct, stressed.max_drawdown_pct,
                                         "Только финальный период; выбор стратегии не менялся"))
    if final.entry_indices and final.entry_indices[0] + 1 < n:
        shock_at = final.entry_indices[0] + 1
        changed = list(candles)
        original = changed[shock_at]
        gap_open = original.open * D("0.9")
        changed[shock_at] = replace(original, open=gap_open, low=min(original.low, gap_open))
        shocked = run_backtest(tuple(changed), validation_end, n, selected, config, period="GAP_DOWN_10PCT")
        stresses.append(StressResult("Гэп вниз 10% после первого входа", shocked.return_pct,
                                     shocked.max_drawdown_pct, "Стоп исполнен по худшему open; путь цены синтетический"))
    else:
        stresses.append(StressResult("Гэп вниз 10%", None, None, "На финальном периоде нет входа для проверки гэпа"))
    stresses.append(_loss_streak(config))
    gap_stress = stresses[-2]
    if gap_stress.return_pct is not None and gap_stress.return_pct < -D("2"):
        warnings.append("Сценарий гэпа превышает дневной лимит потерь 2%; стоп не гарантирует предел убытка")
    credible = (not warnings and final.return_pct > 0 and final.relative_buy_hold_pct is not None
                and final.relative_buy_hold_pct > 0 and final.profit_factor is not None
                and final.profit_factor > 1)
    recommendation = (f"ИССЛЕДОВАТЕЛЬСКИЙ КАНДИДАТ: {selected}" if credible else
                      "НЕТ ДОСТАТОЧНЫХ ОСНОВАНИЙ МЕНЯТЬ EMA 9/21 ИЛИ ВКЛЮЧАТЬ ТОРГОВЛЮ")
    if not credible and not warnings:
        warnings.append("Финальный период не подтвердил рост после затрат и превосходство Buy & Hold")
    return ResearchReport((0, train_end, validation_end, n), training, validation, selected,
                          final, tuple(stresses), recommendation, tuple(warnings))
