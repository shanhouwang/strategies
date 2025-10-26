"""Backtest engine for the Lorentzian KNN strategy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import PartialStopLeg, PartialTrailingLeg, Settings


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


@dataclass
class TrailingLegState:
    qty: int
    trigger_pct: float
    offset_pct: float
    trigger_price: float
    peak_price: float
    leg_index: int
    stop_price: float | None = None
    active: bool = False
    exited: bool = False


@dataclass
class StopLegState:
    qty: int
    stop_loss_pct: float
    stop_price: float
    leg_index: int
    exited: bool = False


def _select_platform_tier_fee(
    qty: int, tiers: List[Tuple[int, Optional[int], float]]
) -> float:
    """Return the platform fee contribution based on share quantity tiers."""
    shares = max(qty, 0)
    if shares == 0:
        return 0.0
    for lower, upper, fee_per_share in tiers:
        if shares >= lower and (upper is None or shares <= upper):
            return fee_per_share * shares
    return 0.0


def _calculate_transaction_cost(
    qty: int,
    price: float,
    settings: Settings,
    *,
    include_transfer_fee: bool = False,
) -> float:
    """Calculate all fees for a single trade leg."""
    shares = abs(qty)
    if shares == 0:
        return 0.0
    notional = shares * price
    commission_component = settings.commission + settings.commission_rate * notional
    commission_component = max(commission_component, settings.commission_min)
    platform_component = (
        settings.platform_fee_fixed_per_share * shares
        + _select_platform_tier_fee(shares, settings.platform_fee_tiers)
    )
    transfer_component = settings.stock_transfer_in_fee if include_transfer_fee else 0.0
    return commission_component + platform_component + transfer_component


def _distribute_quantities(total_qty: int, ratios: List[float]) -> List[int]:
    allocations: List[int] = []
    remaining = total_qty
    if total_qty <= 0:
        return [0 for _ in ratios]
    for idx, ratio in enumerate(ratios):
        if remaining <= 0:
            allocations.append(0)
            continue
        if idx == len(ratios) - 1:
            qty = remaining
        else:
            qty = int(round(total_qty * ratio))
            qty = max(0, min(qty, remaining))
        allocations.append(qty)
        remaining -= qty
    if remaining > 0 and allocations:
        allocations[-1] += remaining
    return allocations


def _build_trailing_leg_states(
    total_qty: int,
    entry_price: float,
    legs_def: List[PartialTrailingLeg],
    direction: int,
) -> List[TrailingLegState]:
    ratios = [leg.ratio for leg in legs_def]
    allocations = _distribute_quantities(total_qty, ratios)
    states: List[TrailingLegState] = []
    for idx, (leg_def, qty) in enumerate(zip(legs_def, allocations), start=1):
        if qty <= 0:
            continue
        trigger_pct = leg_def.trigger_pct
        if direction == 1:
            trigger_price = entry_price * (1 + trigger_pct / 100.0)
        else:
            trigger_price = entry_price * (1 - trigger_pct / 100.0)
        states.append(
            TrailingLegState(
                qty=qty,
                trigger_pct=trigger_pct,
                offset_pct=leg_def.offset_pct,
                trigger_price=trigger_price,
                peak_price=entry_price,
                leg_index=idx,
            )
        )
    return states


def _build_stop_leg_states(
    total_qty: int,
    entry_price: float,
    legs_def: List[PartialStopLeg],
    direction: int,
) -> List[StopLegState]:
    ratios = [leg.ratio for leg in legs_def]
    allocations = _distribute_quantities(total_qty, ratios)
    states: List[StopLegState] = []
    for idx, (leg_def, qty) in enumerate(zip(legs_def, allocations), start=1):
        if qty <= 0:
            continue
        if direction == 1:
            stop_price = entry_price * (1 - leg_def.stop_loss_pct / 100.0)
        else:
            stop_price = entry_price * (1 + leg_def.stop_loss_pct / 100.0)
        states.append(
            StopLegState(
                qty=qty,
                stop_loss_pct=leg_def.stop_loss_pct,
                stop_price=stop_price,
                leg_index=idx,
            )
        )
    return states


@dataclass
class PositionState:
    qty: int = 0
    direction: int = 0  # 1 long, -1 short
    entry_price: float = 0.0
    entry_time: pd.Timestamp | None = None
    stop_price: float | None = None
    limit_price: float | None = None
    trailing_active: bool = False
    max_price: float | None = None
    min_price: float | None = None
    entry_fees: float = 0.0
    trailing_legs: List[TrailingLegState] = field(default_factory=list)
    stop_legs: List[StopLegState] = field(default_factory=list)

    def reset(self) -> None:
        self.qty = 0
        self.direction = 0
        self.entry_price = 0.0
        self.entry_time = None
        self.stop_price = None
        self.limit_price = None
        self.trailing_active = False
        self.max_price = None
        self.min_price = None
        self.entry_fees = 0.0
        self.trailing_legs.clear()
        self.stop_legs.clear()


def _bars_between(df: pd.DataFrame, start: pd.Timestamp | None, end: pd.Timestamp) -> int:
    if start is None:
        return 0
    return len(df.loc[start:end])


def _compute_long_risk(
    state: PositionState, row: pd.Series, settings: Settings
) -> tuple[float | None, float | None, str | None]:
    reason = None
    entry = state.entry_price
    high_price = float(row.get("high", entry))
    if settings.enable_moving_profit and settings.enable_partial_trailing and state.trailing_legs:
        state.max_price = max(state.max_price or high_price, high_price)
        for leg in state.trailing_legs:
            if leg.exited or leg.qty <= 0:
                continue
            leg.peak_price = max(leg.peak_price, high_price)
            if not leg.active and leg.peak_price >= leg.trigger_price - 1e-9:
                leg.active = True
            if leg.active:
                profit = leg.peak_price - entry
                stop_candidate = entry + profit * (1 - leg.offset_pct / 100.0)
                leg.stop_price = max(leg.stop_price or 0.0, stop_candidate)
        state.stop_price = None
        state.limit_price = None
        return state.stop_price, state.limit_price, None
    if settings.enable_moving_profit:
        trigger_price = entry * (1 + settings.long_trailing_trigger_pct / 100.0)
        state.max_price = max(state.max_price or high_price, high_price)
        if state.max_price >= trigger_price:
            state.trailing_active = True
        if state.trailing_active:
            profit = state.max_price - entry
            stop_candidate = entry + profit * (1 - settings.long_trailing_offset_pct / 100.0)
            state.stop_price = max(state.stop_price or 0.0, stop_candidate)
            reason = "移动止损"
        else:
            state.stop_price = entry * (1 - settings.long_stop_loss_pct / 100.0)
            reason = "固定止损"
        state.limit_price = None
    elif settings.enable_stop_loss_only:
        state.stop_price = entry * (1 - settings.long_stop_loss_pct / 100.0)
        state.limit_price = None
        reason = "固定止损"
    elif settings.enable_fixed_profit:
        state.stop_price = entry * (1 - settings.long_stop_loss_pct / 100.0)
        state.limit_price = entry + entry * (settings.long_stop_loss_pct / 100.0) * settings.long_profit_loss_ratio
        reason = None
    elif settings.enable_atr_profit:
        atr_value = row.get("atr")
        if pd.notna(atr_value):
            atr_target = float(atr_value) * settings.atr_multiplier
            state.stop_price = max(entry - atr_target, 0.0)
            state.limit_price = entry + atr_target * settings.long_profit_loss_ratio
            reason = None
        else:
            state.stop_price = None
            state.limit_price = None
    else:
        state.stop_price = None
        state.limit_price = None
    return state.stop_price, state.limit_price, reason


def _compute_short_risk(
    state: PositionState, row: pd.Series, settings: Settings
) -> tuple[float | None, float | None, str | None]:
    reason = None
    entry = state.entry_price
    low_price = float(row.get("low", entry))
    if settings.enable_moving_profit and settings.enable_partial_trailing and state.trailing_legs:
        state.min_price = min(state.min_price or low_price, low_price)
        for leg in state.trailing_legs:
            if leg.exited or leg.qty <= 0:
                continue
            leg.peak_price = min(leg.peak_price, low_price)
            if not leg.active and leg.peak_price <= leg.trigger_price + 1e-9:
                leg.active = True
            if leg.active:
                profit = entry - leg.peak_price
                stop_candidate = entry - profit * (1 - leg.offset_pct / 100.0)
                leg.stop_price = min(leg.stop_price or float("inf"), stop_candidate)
        state.stop_price = None
        state.limit_price = None
        return state.stop_price, state.limit_price, None
    if settings.enable_moving_profit:
        trigger_price = entry * (1 - settings.short_trailing_trigger_pct / 100.0)
        state.min_price = min(state.min_price or low_price, low_price)
        if state.min_price <= trigger_price:
            state.trailing_active = True
        if state.trailing_active:
            profit = entry - state.min_price
            stop_candidate = entry - profit * (1 - settings.short_trailing_offset_pct / 100.0)
            state.stop_price = min(state.stop_price or float("inf"), stop_candidate)
            reason = "移动止损"
        else:
            state.stop_price = entry * (1 + settings.short_stop_loss_pct / 100.0)
            reason = "固定止损"
        state.limit_price = None
    elif settings.enable_stop_loss_only:
        state.stop_price = entry * (1 + settings.short_stop_loss_pct / 100.0)
        state.limit_price = None
        reason = "固定止损"
    elif settings.enable_fixed_profit:
        state.stop_price = entry * (1 + settings.short_stop_loss_pct / 100.0)
        state.limit_price = entry - entry * (settings.short_stop_loss_pct / 100.0) * settings.short_profit_loss_ratio
        reason = None
    elif settings.enable_atr_profit:
        atr_value = row.get("atr")
        if pd.notna(atr_value):
            atr_target = float(atr_value) * settings.atr_multiplier
            state.stop_price = entry + atr_target
            state.limit_price = entry - atr_target * settings.short_profit_loss_ratio
            reason = None
        else:
            state.stop_price = None
            state.limit_price = None
    else:
        state.stop_price = None
        state.limit_price = None
    return state.stop_price, state.limit_price, reason


def _collect_long_partial_exits(
    position: PositionState, settings: Settings, low_price: float
) -> List[tuple[StopLegState | TrailingLegState, float, str]]:
    events: List[tuple[StopLegState | TrailingLegState, float, str]] = []
    for leg in position.stop_legs:
        if leg.qty <= 0 or leg.exited:
            continue
        if low_price <= leg.stop_price:
            exec_price = max(leg.stop_price - settings.slippage, 0.0)
            events.append((leg, exec_price, f"分批止损-{leg.leg_index}"))
    if settings.enable_moving_profit and settings.enable_partial_trailing:
        for leg in position.trailing_legs:
            if leg.qty <= 0 or leg.exited:
                continue
            if leg.stop_price is not None and low_price <= leg.stop_price:
                exec_price = max(leg.stop_price - settings.slippage, 0.0)
                events.append((leg, exec_price, f"分批移动止盈-{leg.leg_index}"))
    return events


def _collect_short_partial_exits(
    position: PositionState, settings: Settings, high_price: float
) -> List[tuple[StopLegState | TrailingLegState, float, str]]:
    events: List[tuple[StopLegState | TrailingLegState, float, str]] = []
    for leg in position.stop_legs:
        if leg.qty <= 0 or leg.exited:
            continue
        if high_price >= leg.stop_price:
            exec_price = leg.stop_price + settings.slippage
            events.append((leg, exec_price, f"分批止损-{leg.leg_index}"))
    if settings.enable_moving_profit and settings.enable_partial_trailing:
        for leg in position.trailing_legs:
            if leg.qty <= 0 or leg.exited:
                continue
            if leg.stop_price is not None and high_price >= leg.stop_price:
                exec_price = leg.stop_price + settings.slippage
                events.append((leg, exec_price, f"分批移动止盈-{leg.leg_index}"))
    return events


def _prune_legs(legs: List[TrailingLegState | StopLegState]) -> List[TrailingLegState | StopLegState]:
    return [leg for leg in legs if leg.qty > 0 and not getattr(leg, "exited", False)]


def _cap_leg_quantities(
    legs: List[TrailingLegState | StopLegState], available: int
) -> None:
    if available <= 0:
        for leg in legs:
            leg.qty = 0
            leg.exited = True
        return
    total = sum(leg.qty for leg in legs if not leg.exited)
    overflow = total - available
    if overflow <= 0:
        return
    for leg in reversed(legs):
        if leg.exited or leg.qty <= 0:
            continue
        reduction = min(leg.qty, overflow)
        leg.qty -= reduction
        overflow -= reduction
        if leg.qty <= 0:
            leg.exited = True
        if overflow <= 0:
            break


def run_backtest(data: pd.DataFrame, settings: Settings) -> BacktestResult:
    """Execute a sequential backtest using generated signals."""
    df = data.copy()
    cash = settings.initial_capital
    position = PositionState()
    trades: List[Trade] = []
    equity_points: List[Tuple[pd.Timestamp, float]] = []

    def exit_long(
        ts: pd.Timestamp, exec_price: float, reason: str, qty: int | None = None
    ) -> None:
        nonlocal cash
        current_qty = position.qty
        if current_qty <= 0:
            return
        exit_qty = current_qty if qty is None else min(qty, current_qty)
        if exit_qty <= 0:
            return
        exit_fee = _calculate_transaction_cost(exit_qty, exec_price, settings)
        proceeds = exec_price * exit_qty - exit_fee
        cash += proceeds
        entry_fee_share = (
            position.entry_fees * (exit_qty / current_qty) if current_qty else 0.0
        )
        pnl = (exec_price - position.entry_price) * exit_qty - entry_fee_share - exit_fee
        denom = position.entry_price * exit_qty if position.entry_price else 0.0
        return_pct = pnl / denom if denom else 0.0
        trades.append(
            Trade(
                entry_time=position.entry_time,
                exit_time=ts,
                entry_price=position.entry_price,
                exit_price=exec_price,
                quantity=exit_qty,
                pnl=pnl,
                return_pct=return_pct,
                bars_held=_bars_between(df, position.entry_time, ts),
                direction="做多",
                exit_reason=reason,
            )
        )
        position.entry_fees = max(position.entry_fees - entry_fee_share, 0.0)
        position.qty -= exit_qty
        if position.qty <= 0:
            position.reset()

    def exit_short(
        ts: pd.Timestamp, exec_price: float, reason: str, qty: int | None = None
    ) -> None:
        nonlocal cash
        current_qty = abs(position.qty)
        if current_qty <= 0:
            return
        exit_qty = current_qty if qty is None else min(qty, current_qty)
        if exit_qty <= 0:
            return
        exit_fee = _calculate_transaction_cost(exit_qty, exec_price, settings)
        cost = exec_price * exit_qty + exit_fee
        cash -= cost
        entry_fee_share = (
            position.entry_fees * (exit_qty / current_qty) if current_qty else 0.0
        )
        pnl = (position.entry_price - exec_price) * exit_qty - entry_fee_share - exit_fee
        denom = position.entry_price * exit_qty if position.entry_price else 0.0
        return_pct = pnl / denom if denom else 0.0
        trades.append(
            Trade(
                entry_time=position.entry_time,
                exit_time=ts,
                entry_price=position.entry_price,
                exit_price=exec_price,
                quantity=exit_qty,
                pnl=pnl,
                return_pct=return_pct,
                bars_held=_bars_between(df, position.entry_time, ts),
                direction="做空",
                exit_reason=reason,
            )
        )
        position.entry_fees = max(position.entry_fees - entry_fee_share, 0.0)
        position.qty += exit_qty
        if position.qty >= 0:
            position.reset()

    for ts, row in df.iterrows():
        price = float(row["close"])
        high_price = float(row.get("high", price))
        low_price = float(row.get("low", price))
        adx_value = float(row.get("adx", 0.0))

        equity = cash + position.qty * price
        equity_points.append((ts, equity))

        # Update trailing references
        if position.direction == 1:
            position.max_price = max(position.max_price or high_price, high_price)
        elif position.direction == -1:
            position.min_price = min(position.min_price or low_price, low_price)

        # Recalculate stop/limit
        if position.direction == 1:
            stop_price, limit_price, risk_reason = _compute_long_risk(position, row, settings)
        elif position.direction == -1:
            stop_price, limit_price, risk_reason = _compute_short_risk(position, row, settings)
        else:
            stop_price = limit_price = None
            risk_reason = None

        # Partial exits
        if position.direction == 1 and (position.stop_legs or position.trailing_legs):
            partial_events = _collect_long_partial_exits(position, settings, low_price)
            for leg, exec_price, reason in partial_events:
                if position.direction != 1 or position.qty <= 0:
                    break
                exit_qty = min(leg.qty, position.qty)
                if exit_qty <= 0:
                    leg.exited = True
                    continue
                exit_long(ts, exec_price, reason, exit_qty)
                leg.qty -= exit_qty
                if leg.qty <= 0:
                    leg.exited = True
            if partial_events:
                if position.direction == 1 and position.qty > 0:
                    available = position.qty
                    _cap_leg_quantities(position.stop_legs, available)
                    _cap_leg_quantities(position.trailing_legs, available)
                    position.stop_legs = _prune_legs(position.stop_legs)
                    position.trailing_legs = _prune_legs(position.trailing_legs)
                else:
                    position.stop_legs = []
                    position.trailing_legs = []
                    continue
        elif position.direction == -1 and (position.stop_legs or position.trailing_legs):
            partial_events = _collect_short_partial_exits(position, settings, high_price)
            for leg, exec_price, reason in partial_events:
                remaining = abs(position.qty)
                if position.direction != -1 or remaining <= 0:
                    break
                exit_qty = min(leg.qty, remaining)
                if exit_qty <= 0:
                    leg.exited = True
                    continue
                exit_short(ts, exec_price, reason, exit_qty)
                leg.qty -= exit_qty
                if leg.qty <= 0:
                    leg.exited = True
            if partial_events:
                if position.direction == -1 and position.qty < 0:
                    available = abs(position.qty)
                    _cap_leg_quantities(position.stop_legs, available)
                    _cap_leg_quantities(position.trailing_legs, available)
                    position.stop_legs = _prune_legs(position.stop_legs)
                    position.trailing_legs = _prune_legs(position.trailing_legs)
                else:
                    position.stop_legs = []
                    position.trailing_legs = []
                    continue

        # Risk exits
        if position.direction == 1 and position.stop_price is not None and low_price <= position.stop_price:
            exec_price = max(position.stop_price - settings.slippage, 0.0)
            exit_long(ts, exec_price, risk_reason or "止损")
        elif position.direction == 1 and position.limit_price is not None and high_price >= position.limit_price:
            exec_price = max(position.limit_price - settings.slippage, 0.0)
            exit_long(ts, exec_price, "止盈")

        if position.direction == -1 and position.stop_price is not None and high_price >= position.stop_price:
            exec_price = position.stop_price + settings.slippage
            exit_short(ts, exec_price, risk_reason or "止损")
        elif position.direction == -1 and position.limit_price is not None and low_price <= position.limit_price:
            exec_price = position.limit_price + settings.slippage
            exit_short(ts, exec_price, "止盈")

        # Entry signals (after risk exits to avoid double counting)
        long_signal = bool(row.get("start_long", False)) and adx_value >= settings.adx_threshold
        short_signal = bool(row.get("start_short", False)) and adx_value >= settings.adx_threshold

        if position.direction == 1 and short_signal:
            exec_price = max(price - settings.slippage, 0.0)
            exit_long(ts, exec_price, "信号反转")
        if position.direction == -1 and long_signal:
            exec_price = price + settings.slippage
            exit_short(ts, exec_price, "信号反转")

        if position.direction == 0:
            if long_signal:
                exec_price = price + settings.slippage
                if exec_price <= 0:
                    continue
                qty = int(cash // max(exec_price, 1e-9))
                if qty > 0:
                    while qty > 0:
                        entry_fee = _calculate_transaction_cost(
                            qty, exec_price, settings, include_transfer_fee=True
                        )
                        cost = exec_price * qty + entry_fee
                        if cost <= cash + 1e-9:
                            break
                        qty -= 1
                    if qty <= 0:
                        continue
                    entry_fee = _calculate_transaction_cost(
                        qty, exec_price, settings, include_transfer_fee=True
                    )
                    cost = exec_price * qty + entry_fee
                    cash -= cost
                    position.qty = qty
                    position.direction = 1
                    position.entry_price = exec_price
                    position.entry_time = ts
                    position.max_price = exec_price
                    position.stop_price = None
                    position.limit_price = None
                    position.trailing_active = False
                    position.min_price = None
                    position.entry_fees = entry_fee
                    position.trailing_legs = (
                        _build_trailing_leg_states(
                            qty,
                            exec_price,
                            settings.long_partial_trailing_legs,
                            direction=1,
                        )
                        if settings.enable_partial_trailing and settings.enable_moving_profit
                        else []
                    )
                    position.stop_legs = (
                        _build_stop_leg_states(
                            qty,
                            exec_price,
                            settings.long_partial_stop_legs,
                            direction=1,
                        )
                        if settings.enable_partial_stop_loss
                        else []
                    )
            elif short_signal and settings.open_short:
                exec_price = price - settings.slippage
                if exec_price <= 0:
                    continue
                qty = int(cash // max(exec_price, 1e-9))
                if qty > 0:
                    while qty > 0:
                        entry_fee = _calculate_transaction_cost(
                            qty, exec_price, settings, include_transfer_fee=True
                        )
                        proceeds = exec_price * qty - entry_fee
                        if proceeds > 0:
                            break
                        qty -= 1
                    if qty <= 0:
                        continue
                    entry_fee = _calculate_transaction_cost(
                        qty, exec_price, settings, include_transfer_fee=True
                    )
                    proceeds = exec_price * qty - entry_fee
                    cash += proceeds
                    position.qty = -qty
                    position.direction = -1
                    position.entry_price = exec_price
                    position.entry_time = ts
                    position.min_price = exec_price
                    position.stop_price = None
                    position.limit_price = None
                    position.trailing_active = False
                    position.max_price = None
                    position.entry_fees = entry_fee
                    position.trailing_legs = (
                        _build_trailing_leg_states(
                            qty,
                            exec_price,
                            settings.short_partial_trailing_legs,
                            direction=-1,
                        )
                        if settings.enable_partial_trailing and settings.enable_moving_profit
                        else []
                    )
                    position.stop_legs = (
                        _build_stop_leg_states(
                            qty,
                            exec_price,
                            settings.short_partial_stop_legs,
                            direction=-1,
                        )
                        if settings.enable_partial_stop_loss
                        else []
                    )

    # Liquidate at end
    if position.direction == 1:
        last_ts = df.index[-1]
        last_price = max(float(df["close"].iat[-1]) - settings.slippage, 0.0)
        exit_long(last_ts, last_price, "到期强制平仓")
    elif position.direction == -1:
        last_ts = df.index[-1]
        last_price = float(df["close"].iat[-1]) + settings.slippage
        exit_short(last_ts, last_price, "到期强制平仓")

    equity_series = pd.Series(
        [point[1] for point in equity_points],
        index=[point[0] for point in equity_points],
        name="equity",
    )

    final_equity = equity_series.iloc[-1]
    total_return = (final_equity / settings.initial_capital) - 1.0
    first_ts = equity_series.index[0]
    last_ts = equity_series.index[-1]
    elapsed_seconds = max((last_ts - first_ts).total_seconds(), 0.0)
    seconds_per_year = 365.25 * 24 * 60 * 60
    if elapsed_seconds <= 0 or final_equity <= 0:
        compounded_growth = total_return
    else:
        years = elapsed_seconds / seconds_per_year
        if years <= 0:
            compounded_growth = total_return
        else:
            compounded_growth = (final_equity / settings.initial_capital) ** (1 / years) - 1

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
