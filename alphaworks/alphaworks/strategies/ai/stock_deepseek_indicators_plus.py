"""带指标 Plus 版 DeepSeek 股票策略（可用于回测，也可复用实盘执行器）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .stock_deepseek_strategy import StockDeepseekStrategy
from ... import plotting
from ...backtest import Trade
from ...config import Settings
from ...longport_client import build_config

try:  # pragma: no cover - optional dependency
    from longport.openapi import (
        OrderSide,
        OrderType,
        OutsideRTH,
        PushOrderChanged,
        TimeInForceType,
        TopicType,
        TradeContext,
    )
except ImportError:  # pragma: no cover - runtime fallback
    OrderSide = OrderType = OutsideRTH = PushOrderChanged = TimeInForceType = TopicType = TradeContext = None  # type: ignore[assignment]


class StockDeepseekOkPlusStrategy(StockDeepseekStrategy):
    """复刻 OK 版指标提示词的股票回测策略"""

    name = "stock_deepseek_ok_plus"

    def _kline_records(self, df_slice: pd.DataFrame) -> List[Dict[str, Any]]:  # type: ignore[override]
        subset = df_slice.tail(5).reset_index()
        time_col = subset.columns[0]
        subset["timestamp"] = pd.to_datetime(subset[time_col]).dt.strftime("%Y-%m-%d %H:%M:%S")
        return subset[["timestamp", "open", "high", "low", "close", "volume"]].to_dict("records")

    def _build_prompt(  # type: ignore[override]
        self,
        symbol: str,
        timeframe: str,
        price_data: Dict[str, Any],
        trend: Dict[str, Any],
        levels: Dict[str, Any],
        sentiment_text: str,  # Unused, for compatibility with base signature
        kline_text: str,
        technical_text: str,
        position_text: str,
        last_signal_desc: str,
    ) -> str:
        return f"""
你是一个专业的美股交易分析师。标的 {symbol}，周期 {timeframe}。

{kline_text}

{technical_text}

【趋势诊断】
- 短期: {trend['short_term']}
- 中期: {trend['medium_term']}
- 整体: {trend['overall']}
- MACD: {trend['macd']}

【关键位置】
- 静态阻力: {levels.get('static_resistance', float('nan')):.2f}
- 静态支撑: {levels.get('static_support', float('nan')):.2f}
- 动态阻力: {levels.get('dynamic_resistance', float('nan')):.2f}
- 动态支撑: {levels.get('dynamic_support', float('nan')):.2f}

【当前行情】
- 最新价格: {price_data['price']:.2f}
- 当根范围: {price_data.get('low', 0):.2f} - {price_data.get('high', 0):.2f}
- 涨跌幅: {price_data.get('price_change', 0):+.2f}%
- 持仓状态: {position_text}
- 上次信号: {last_signal_desc}

请严格按 JSON 输出：
{{
  "signal": "BUY|SELL|HOLD",
  "reason": "简短分析",
  "stop_loss": 数值,
  "take_profit": 数值,
  "confidence": "HIGH|MEDIUM|LOW"
}}
"""

    def _sentiment_from_returns(self, returns: pd.Series) -> Dict[str, Any] | None:  # type: ignore[override]
        """Plus 版不使用市场情绪，直接返回 None."""
        return None

    def plot(  # type: ignore[override]
        self,
        data: pd.DataFrame,
        trades: Iterable[Trade],
        output_path: Path,
        equity_curve: pd.Series | None = None,
        initial_capital: float | None = None,
    ) -> Path:
        """复用超级趋势的资金曲线图生成逻辑。"""
        return plotting.plot_supertrend(
            data=data,
            trades=trades,
            equity_curve=equity_curve,
            initial_capital=initial_capital,
            output_path=output_path,
            title=f"{self.name} 资金曲线",
        )


__all__ = ["StockDeepseekOkPlusStrategy"]


class LongbridgeTradeExecutor:
    """封装 Longbridge OpenAPI 交易流程，可被脚本或实时策略复用。"""

    def __init__(
        self,
        settings: Settings,
        *,
        symbol: str,
        allow_short: bool,
        test_mode: bool = True,
        default_order_type: OrderType | None = None,
        default_time_in_force: TimeInForceType | None = None,
        outside_rth: bool = False,
        remark_prefix: str = "DeepSeek AI",
    ) -> None:
        self.settings = settings
        self.symbol = symbol
        self.allow_short = allow_short
        self.test_mode = test_mode
        self.order_type = default_order_type or (OrderType.MO if OrderType else None)
        self.time_in_force = default_time_in_force or (
            TimeInForceType.Day if TimeInForceType else None
        )
        self.outside_rth = outside_rth
        self.remark_prefix = remark_prefix
        self._trade_ctx: TradeContext | None = None
        self._subscribed = False

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _ensure_ctx(self) -> TradeContext:
        if TradeContext is None:
            raise RuntimeError("未安装 longport SDK，无法实盘交易。")
        if self._trade_ctx is None:
            cfg = build_config(self.settings)
            self._trade_ctx = TradeContext(cfg)
        if not self._subscribed and TopicType is not None:
            try:
                self._trade_ctx.set_on_order_changed(self._handle_order_event)
                topic = getattr(TopicType, "Private", TopicType)
                self._trade_ctx.subscribe([topic])
                self._subscribed = True
            except Exception as exc:  # pragma: no cover - SDK 运行时异常
                print(f"⚠️ Longbridge 订单推送订阅失败: {exc}")
        return self._trade_ctx

    def _handle_order_event(self, ctx: TradeContext, event: PushOrderChanged) -> None:
        """打印订单状态更新，便于实盘排查。"""
        status = getattr(event.status, "name", str(event.status))
        side = getattr(event.side, "name", str(event.side))
        qty = getattr(event, "executed_quantity", None)
        price = getattr(event, "executed_price", None)
        print(
            f"📥 Longbridge 推送: id={event.order_id} status={status} "
            f"side={side} exec_qty={qty} exec_price={price}"
        )

    def _submit_order(self, side: str, quantity: int, price: float, reason: str) -> None:
        """将标准化指令转换为 Longbridge 委托。"""
        if quantity <= 0:
            return
        reason = reason.strip() or "signal"
        remark = f"{self.remark_prefix} {reason}".strip()[:40]
        price = float(max(price, 0))

        if self.test_mode:
            print(f"[PaperTrading] {side} {quantity} @ {price:.2f} | {remark}")
            return

        ctx = self._ensure_ctx()
        order_side = OrderSide.Buy if side.upper() == "BUY" else OrderSide.Sell
        order_type = self.order_type or OrderType.LO
        tif = self.time_in_force or TimeInForceType.Day
        outside_flag = None
        if OutsideRTH is not None:
            outside_flag = OutsideRTH.AnyTime if self.outside_rth else OutsideRTH.RTHOnly
        submit_kwargs: Dict[str, Any] = {
            "submitted_price": price,
            "remark": remark,
        }
        if outside_flag is not None:
            submit_kwargs["outside_rth"] = outside_flag
        try:
            order_id = ctx.submit_order(
                self.symbol,
                order_type,
                order_side,
                quantity,
                tif,
                **submit_kwargs,
            )
            print(
                f"📤 Longbridge 下单成功 id={order_id} "
                f"{getattr(order_side, 'name', str(order_side))} "
                f"{quantity} @ {price:.2f}"
            )
        except Exception as exc:  # pragma: no cover - 网络/账户异常
            print(f"❌ Longbridge 下单失败: {exc}")

    # ------------------------------------------------------------------ #
    # 对外接口
    # ------------------------------------------------------------------ #
    def sync_position(
        self,
        *,
        signal: str,
        target_side: str | None,
        target_size: int,
        price: float,
        current_position: Optional[Dict[str, Any]],
        reason: str = "",
    ) -> None:
        """根据 AI 信号与当前持仓，提交必要委托实现目标仓位。"""
        target_side = (target_side or "").lower() or None
        if target_side == "short" and not self.allow_short:
            print("⚠️ 当前账户未启用做空，忽略开空指令。")
            target_side = None
        current_side = (current_position or {}).get("side")
        current_size = int((current_position or {}).get("size", 0))

        actions: List[Tuple[str, int, str]] = []

        def queue(order_side: str, qty: int, action_reason: str) -> None:
            if qty > 0:
                actions.append((order_side, qty, action_reason))

        if target_side == "long":
            if current_side == "short" and current_size > 0:
                queue("BUY", current_size, "cover short→flat")
                current_side = None
                current_size = 0
            delta = target_size - current_size
            if delta > 0:
                queue("BUY", delta, "increase long")
            elif delta < 0:
                queue("SELL", -delta, "trim long")
        elif target_side == "short":
            if current_side == "long" and current_size > 0:
                queue("SELL", current_size, "close long→flat")
                current_side = None
                current_size = 0
            delta = target_size - current_size
            if delta > 0:
                queue("SELL", delta, "increase short")
            elif delta < 0:
                queue("BUY", -delta, "trim short")
        else:  # 清仓
            if current_side == "long" and current_size > 0:
                queue("SELL", current_size, "flat long")
            elif current_side == "short" and current_size > 0:
                queue("BUY", current_size, "flat short")

        if not actions:
            print("ℹ️ 目标仓位与当前一致，无需下单。")
            return

        for order_side, qty, action_reason in actions:
            desc = f"{signal}: {action_reason}"
            self._submit_order(order_side, qty, price, f"{desc} | {reason}")

    @staticmethod
    def _extract_value(record: Any, keys: Tuple[str, ...]) -> float | None:
        for key in keys:
            value = None
            if isinstance(record, dict):
                value = record.get(key)
            else:
                value = getattr(record, key, None)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    def fetch_available_cash(self) -> float | None:
        if self.test_mode:
            return None
        try:
            ctx = self._ensure_ctx()
            balances = ctx.account_balance()
        except Exception as exc:
            print(f"⚠️ 获取账户资金失败: {exc}")
            return None
        items = balances if isinstance(balances, (list, tuple)) else [balances]
        if not items:
            return None
        record = items[0]
        value = self._extract_value(
            record,
            (
                "buy_power",
                "total_cash",
                "net_assets",
                "remaining_finance_amount",
            ),
        )
        if value is None and getattr(record, "cash_infos", None):
            for info in record.cash_infos:
                value = self._extract_value(
                    info,
                    (
                        "available_cash",
                        "withdraw_cash",
                        "cash",
                        "balance",
                    ),
                )
                if value is not None:
                    break
        return value

    def fetch_position_snapshot(self) -> Optional[Dict[str, Any]]:
        if self.test_mode:
            return None
        try:
            ctx = self._ensure_ctx()
            positions = ctx.stock_positions(symbols=[self.symbol])
        except Exception as exc:
            print(f"⚠️ 获取实盘持仓失败: {exc}")
            return None
        channels = getattr(positions, "channels", None)
        if not channels:
            return None
        for channel in channels:
            for record in getattr(channel, "positions", []) or []:
                qty = self._extract_value(record, ("quantity", "qty", "position"))
                if qty is None or abs(qty) < 1e-6:
                    continue
                side = "long" if qty > 0 else "short"
                entry = self._extract_value(
                    record,
                    ("average_price", "avg_price", "cost_price", "price"),
                )
                return {
                    "side": side,
                    "size": int(abs(round(qty))),
                    "entry_price": float(entry or 0.0),
                }
        return None

    def estimate_max_quantity(
        self,
        *,
        price: float,
        side: str,
        order_type: OrderType | None = None,
    ) -> Dict[str, float | int] | None:
        if self.test_mode:
            return None
        try:
            ctx = self._ensure_ctx()
            response = ctx.estimate_max_purchase_quantity(
                self.symbol,
                order_type or self.order_type or OrderType.MO,
                OrderSide.Buy if side.lower() == "buy" else OrderSide.Sell,
                price,
            )
        except Exception as exc:
            print(f"⚠️ 估算最大下单量失败: {exc}")
            return None
        result: Dict[str, float | int] = {}
        result["cash"] = int(getattr(response, "cash_max_qty", 0) or 0)
        result["margin"] = int(getattr(response, "margin_max_qty", 0) or 0)
        buying_power = getattr(response, "buying_power", None)
        if buying_power is not None:
            try:
                result["buying_power"] = float(buying_power)
            except (TypeError, ValueError):
                pass
        return result

    def close(self) -> None:
        """释放 TradeContext，避免资源泄露。"""
        if self._trade_ctx is not None:
            try:
                if self._subscribed:
                    self._trade_ctx.unsubscribe(["order"])
            except Exception:  # pragma: no cover - 连接已断开
                pass
            try:
                if hasattr(self._trade_ctx, "close"):
                    self._trade_ctx.close()  # type: ignore[attr-defined]
            finally:
                self._trade_ctx = None
                self._subscribed = False

    def __del__(self) -> None:  # pragma: no cover - 解释器退出时调用
        self.close()


__all__.append("LongbridgeTradeExecutor")
