"""超级趋势策略的回测工具。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from .config import Settings


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    return_pct: float
    bars_held: int
    direction: str
    exit_reason: str


@dataclass
class BacktestStats:
    total_return: float
    annualized_return: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    trade_count: int


@dataclass
class BacktestResult:
    stats: BacktestStats
    trades: List[Trade]
    equity_curve: pd.Series
    trade_log: pd.DataFrame


PERIODS_PER_YEAR = 252 * 26  # 交易日数量 * 每日 15 分钟线数量（约数）


def run_backtest(data: pd.DataFrame, settings: Settings) -> BacktestResult:
    """执行矢量化回测。"""
    df = data.copy()
    cash = settings.initial_capital
    position = 0
    entry_price = 0.0
    entry_time = None
    entry_atr: float | None = None
    stop_price: float | None = None
    take_profit_price: float | None = None
    trades: List[Trade] = []
    equity_points: List[Tuple[pd.Timestamp, float]] = []
    cooldown_long = 0
    cooldown_short = 0

    def bars_between(start: pd.Timestamp | None, end: pd.Timestamp) -> int:
        if start is None:
            return 0
        return len(df.loc[start:end])

    def exit_long(ts: pd.Timestamp, exec_price: float, reason: str) -> None:
        nonlocal cash, position, entry_price, entry_time, stop_price, take_profit_price, entry_atr, cooldown_long
        qty = position
        if qty <= 0:
            return
        proceeds = exec_price * qty - settings.commission
        cash += proceeds
        pnl = (exec_price - entry_price) * qty - 2 * settings.commission
        denom = entry_price * qty if entry_price else 0.0
        return_pct = pnl / denom if denom else 0.0
        trades.append(
            Trade(
                entry_time=entry_time,
                exit_time=ts,
                entry_price=entry_price,
                exit_price=exec_price,
                quantity=qty,
                pnl=pnl,
                return_pct=return_pct,
                bars_held=bars_between(entry_time, ts),
                direction="做多",
                exit_reason=reason,
            )
        )
        position = 0
        entry_price = 0.0
        entry_time = None
        stop_price = None
        take_profit_price = None
        entry_atr = None
        if settings.cooldown_bars > 0:
            cooldown_long = settings.cooldown_bars

    def exit_short(ts: pd.Timestamp, exec_price: float, reason: str) -> None:
        nonlocal cash, position, entry_price, entry_time, stop_price, take_profit_price, entry_atr, cooldown_short
        qty = abs(position)
        if qty <= 0:
            return
        cost = exec_price * qty + settings.commission
        cash -= cost
        pnl = (entry_price - exec_price) * qty - 2 * settings.commission
        denom = entry_price * qty if entry_price else 0.0
        return_pct = pnl / denom if denom else 0.0
        trades.append(
            Trade(
                entry_time=entry_time,
                exit_time=ts,
                entry_price=entry_price,
                exit_price=exec_price,
                quantity=qty,
                pnl=pnl,
                return_pct=return_pct,
                bars_held=bars_between(entry_time, ts),
                direction="做空",
                exit_reason=reason,
            )
        )
        position = 0
        entry_price = 0.0
        entry_time = None
        stop_price = None
        take_profit_price = None
        entry_atr = None
        if settings.cooldown_bars > 0:
            cooldown_short = settings.cooldown_bars

    def determine_position_size(
        direction: str,
        equity_value: float,
        price_for_entry: float,
        atr_value: float | None,
    ) -> int:
        qty = 0
        if (
            settings.risk_per_trade_pct > 0
            and atr_value is not None
            and atr_value > 0
        ):
            atr_multiplier = (
                settings.risk_atr_multiplier
                if settings.risk_atr_multiplier and settings.risk_atr_multiplier > 0
                else (
                    settings.atr_stop_multiplier
                    if settings.atr_stop_multiplier > 0
                    else 2.0
                )
            )
            risk_per_share = atr_value * atr_multiplier
            if risk_per_share > 0:
                risk_budget = equity_value * settings.risk_per_trade_pct
                qty = int(risk_budget // risk_per_share)
        if qty <= 0:
            qty = int(cash // price_for_entry)
        elif direction == "long":
            qty = min(qty, int(cash // price_for_entry))
        elif direction == "short":
            qty = min(qty, int(cash // price_for_entry))
        return qty

    for ts, row in df.iterrows():
        price = float(row["close"])
        high_price = float(row.get("high", price))
        low_price = float(row.get("low", price))
        atr_raw = row.get("atr", np.nan)
        current_atr = float(atr_raw) if pd.notna(atr_raw) else None
        equity = cash + position * price
        equity_points.append((ts, equity))

        if cooldown_long > 0:
            cooldown_long -= 1
        if cooldown_short > 0:
            cooldown_short -= 1

        long_entry_signal = (
            settings.enable_long
            and cooldown_long == 0
            and row.get("long_entry", 0) == 1
        )
        long_exit_signal = row.get("long_exit", 0) == 1
        short_entry_signal = (
            settings.enable_short
            and cooldown_short == 0
            and row.get("short_entry", 0) == 1
        )
        short_exit_signal = settings.enable_short and row.get("short_exit", 0) == 1

        if position > 0:
            if (
                settings.atr_stop_multiplier > 0
                and stop_price is not None
                and low_price <= stop_price
            ):
                exec_price = max(stop_price - settings.slippage, 0.0)
                exit_long(ts, exec_price, "ATR 止损")
            elif (
                settings.atr_take_profit_multiplier > 0
                and take_profit_price is not None
                and high_price >= take_profit_price
            ):
                exec_price = max(take_profit_price - settings.slippage, 0.0)
                exit_long(ts, exec_price, "ATR 止盈")

        if position < 0:
            if (
                settings.atr_stop_multiplier > 0
                and stop_price is not None
                and high_price >= stop_price
            ):
                exec_price = stop_price + settings.slippage
                exit_short(ts, exec_price, "ATR 止损")
            elif (
                settings.atr_take_profit_multiplier > 0
                and take_profit_price is not None
                and low_price <= take_profit_price
            ):
                exec_price = max(take_profit_price + settings.slippage, 0.0)
                exit_short(ts, exec_price, "ATR 止盈")

        if position > 0 and long_exit_signal:
            exec_price = max(price - settings.slippage, 0.0)
            exit_long(ts, exec_price, "信号平仓")

        if position < 0 and short_exit_signal:
            exec_price = price + settings.slippage
            exit_short(ts, exec_price, "信号平仓")

        if position == 0:
            if long_entry_signal:
                qty = determine_position_size("long", equity, price, current_atr)
                if qty == 0:
                    continue
                exec_price = price + settings.slippage
                cost = exec_price * qty + settings.commission
                cash -= cost
                position = qty
                entry_price = exec_price
                entry_time = ts
                entry_atr = current_atr if current_atr and current_atr > 0 else None
                stop_price = (
                    max(exec_price - entry_atr * settings.atr_stop_multiplier, 0.0)
                    if entry_atr and settings.atr_stop_multiplier > 0
                    else None
                )
                take_profit_price = (
                    exec_price + entry_atr * settings.atr_take_profit_multiplier
                    if entry_atr and settings.atr_take_profit_multiplier > 0
                    else None
                )
            elif short_entry_signal:
                qty = determine_position_size("short", equity, price, current_atr)
                if qty == 0:
                    continue
                exec_price = price - settings.slippage
                proceeds = exec_price * qty - settings.commission
                cash += proceeds
                position = -qty
                entry_price = exec_price
                entry_time = ts
                entry_atr = current_atr if current_atr and current_atr > 0 else None
                stop_price = (
                    exec_price + entry_atr * settings.atr_stop_multiplier
                    if entry_atr and settings.atr_stop_multiplier > 0
                    else None
                )
                take_profit_price = (
                    max(exec_price - entry_atr * settings.atr_take_profit_multiplier, 0.0)
                    if entry_atr and settings.atr_take_profit_multiplier > 0
                    else None
                )

    # 在最后一根 K 线处清算剩余仓位
    if position > 0:
        last_ts = df.index[-1]
        last_price = max(float(df["close"].iat[-1]) - settings.slippage, 0.0)
        exit_long(last_ts, last_price, "到期强制平仓")
    elif position < 0:
        last_ts = df.index[-1]
        last_price = max(float(df["close"].iat[-1]) + settings.slippage, 0.0)
        exit_short(last_ts, last_price, "到期强制平仓")

    equity_series = pd.Series(
        [point[1] for point in equity_points],
        index=[point[0] for point in equity_points],
        name="equity",
    )

    total_return = (equity_series.iloc[-1] / settings.initial_capital) - 1.0
    periods = max(len(df), 1)
    compounded_growth = (1 + total_return) ** (PERIODS_PER_YEAR / periods) - 1

    drawdowns = equity_series / equity_series.cummax() - 1.0
    max_drawdown = drawdowns.min() if not drawdowns.empty else 0.0

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl < 0]
    win_rate = len(wins) / len(trades) if trades else 0.0
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss else np.inf

    stats = BacktestStats(
        total_return=total_return,
        annualized_return=compounded_growth,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        profit_factor=profit_factor,
        trade_count=len(trades),
    )

    log_records = [t.__dict__ for t in trades]
    trade_log = (
        pd.DataFrame(log_records)
        if log_records
        else pd.DataFrame(
            columns=[
                "entry_time",
                "exit_time",
                "entry_price",
                "exit_price",
                "quantity",
                "pnl",
                "return_pct",
                "bars_held",
                "direction",
                "exit_reason",
            ]
        )
    )

    return BacktestResult(
        stats=stats,
        trades=trades,
        equity_curve=equity_series,
        trade_log=trade_log,
    )
