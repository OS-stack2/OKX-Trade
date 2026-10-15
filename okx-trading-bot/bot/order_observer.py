"""Read-only OKX Demo order observation; every unresolved exposure stays HALTED."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from bot.demo_client import DemoClient, DemoError, DemoFill, DemoOrder
from bot.execution_engine import ExecutionDiagnostic, ExecutionEngine, ExecutionError


class DemoOrderObserver:
    """Match a persisted clientOrderId against GET order/fills, never submit."""

    def __init__(self, *, client: DemoClient, engine: ExecutionEngine,
                 clock=lambda: datetime.now(timezone.utc)) -> None:
        self.client = client
        self.engine = engine
        self.clock = clock

    def observe(self, client_order_id: str) -> ExecutionDiagnostic:
        diagnostic = self.engine.diagnose()
        matches = [item for item in diagnostic.intents if item.client_order_id == client_order_id]
        if len(matches) != 1:
            raise ExecutionError("Нет ровно одного сохранённого Demo-намерения для сверки")
        intent = matches[0]
        try:
            order = self.client.get_order_by_client_id(client_order_id)
            if not isinstance(order, DemoOrder):
                raise DemoError("Demo-ордер не подтверждён")
            now = self.clock()
            if (not isinstance(now, datetime) or now.tzinfo is None
                    or now.utcoffset() != timedelta(0)
                    or not isinstance(order.checked_at_utc, datetime)
                    or order.checked_at_utc.tzinfo is None
                    or order.checked_at_utc.utcoffset() != timedelta(0)
                    or order.checked_at_utc > now
                    or now - order.checked_at_utc > timedelta(seconds=60)
                    or order.client_order_id != intent.client_order_id
                    or order.size_btc != intent.quantity
                    or order.limit_price_usdt != intent.entry_price
                    or order.state not in ("live", "partially_filled", "filled", "canceled")
                    or not isinstance(order.filled_btc, Decimal)
                    or not order.filled_btc.is_finite() or order.filled_btc < 0
                    or order.filled_btc > intent.quantity):
                raise DemoError("Demo-ордер не совпадает с локальным намерением")
            fills = self.client.get_order_fills(order.order_id)
            if not isinstance(fills, tuple) or len(fills) >= 100:
                raise DemoError("История исполнений ордера неполна")
            ids: set[str] = set()
            total = Decimal(0)
            for fill in fills:
                if (not isinstance(fill, DemoFill) or fill.order_id != order.order_id
                        or not isinstance(fill.trade_id, str) or not fill.trade_id
                        or fill.trade_id in ids or not isinstance(fill.size_btc, Decimal)
                        or not fill.size_btc.is_finite() or fill.size_btc <= 0
                        or not isinstance(fill.price_usdt, Decimal)
                        or not fill.price_usdt.is_finite() or fill.price_usdt <= 0
                        or not isinstance(fill.fee, Decimal) or not fill.fee.is_finite()
                        or fill.fee > 0 or fill.fee_currency not in ("BTC", "USDT")
                        or not isinstance(fill.observed_at_utc, datetime)
                        or fill.observed_at_utc.tzinfo is None
                        or fill.observed_at_utc.utcoffset() != timedelta(0)
                        or fill.observed_at_utc > now):
                    raise DemoError("Исполнение Demo-ордера противоречиво")
                ids.add(fill.trade_id)
                total += fill.size_btc
            if total != order.filled_btc:
                raise DemoError("Исполненный размер не подтверждён полной историей")
            if ((order.state == "filled" and order.filled_btc != order.size_btc)
                    or (order.state == "partially_filled"
                        and not Decimal(0) < order.filled_btc < order.size_btc)
                    or (order.state == "live" and order.filled_btc != 0)
                    or (order.state == "canceled" and order.filled_btc == order.size_btc)):
                raise DemoError("Статус BUY противоречит фактическому исполнению")
            if not diagnostic.halted:
                self.engine.record_reconciled_ack(client_order_id, order.order_id)
            for fill in fills:
                self.engine.record_fill(client_order_id, trade_id=fill.trade_id,
                                        quantity=fill.size_btc, price=fill.price_usdt,
                                        fee=fill.fee, fee_ccy=fill.fee_currency,
                                        exchange_order_id=order.order_id)
            # Even a live unfilled order can fill without atomically proven SL/TP.
            reason = ("PARTIAL_BUY_ACTIVE" if order.state == "partially_filled" else
                      "PARTIAL_BUY_CANCELED" if order.state == "canceled" and order.filled_btc > 0 else
                      "FILLED_BUY_UNPROTECTED" if order.state == "filled" else
                      "UNPROTECTED_DEMO_ORDER")
            self.engine.emergency_halt(reason)
        except (DemoError, ExecutionError, OSError, TypeError, ValueError):
            self.engine.emergency_halt("ORDER_STATUS_UNCERTAIN")
        return self.engine.diagnose()
