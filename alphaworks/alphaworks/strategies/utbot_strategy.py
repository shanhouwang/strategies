"""UT Bot 多空引擎的 Python 版本。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .. import plotting
from ..backtest import Trade
from .base import Strategy


def ema(series: pd.Series, length: int) -> pd.Series:
    if length <= 0:
        raise ValueError("EMA 长度必须大于 0")
    return series.ewm(span=length, adjust=False).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    )
    tr = tr_components.max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def rsi(series: pd.Series, length: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    return rsi_series.fillna(50.0)


def compute_stc(
    close: pd.Series,
    cycle_length: int,
    fast_length: int,
    slow_length: int,
    smooth_factor: float,
) -> pd.Series:
    macd = ema(close, fast_length) - ema(close, slow_length)
    lowest_macd = macd.rolling(cycle_length).min()
    range_macd = macd.rolling(cycle_length).max() - lowest_macd
    percent_k = ((macd - lowest_macd) / range_macd * 100).where(range_macd > 0)
    percent_k = percent_k.ffill().fillna(50.0)
    smooth_k = percent_k.ewm(alpha=smooth_factor, adjust=False).mean()
    lowest_d = smooth_k.rolling(cycle_length).min()
    range_d = smooth_k.rolling(cycle_length).max() - lowest_d
    percent_d = ((smooth_k - lowest_d) / range_d * 100).where(range_d > 0)
    percent_d = percent_d.ffill().fillna(50.0)
    stc = percent_d.ewm(alpha=smooth_factor, adjust=False).mean()
    return stc


def compute_qqe(
    source: pd.Series,
    rsi_length: int,
    smoothing: int,
    qqe_factor: float,
) -> tuple[pd.Series, pd.Series]:
    rsi_series = rsi(source, rsi_length)
    smoothed_rsi = ema(rsi_series, smoothing)
    atr_rsi = smoothed_rsi.diff().abs()
    wilders_length = rsi_length * 2 - 1
    smoothed_atr = atr_rsi.ewm(alpha=1 / wilders_length, adjust=False).mean()
    delta = smoothed_atr * qqe_factor

    long_band = np.zeros(len(smoothed_rsi))
    short_band = np.zeros(len(smoothed_rsi))

    for i, (sm, dl) in enumerate(zip(smoothed_rsi.fillna(50.0), delta.fillna(0.0))):
        new_long = sm - dl
        new_short = sm + dl
        if i == 0:
            long_band[i] = new_long
            short_band[i] = new_short
            continue
        prev_long = long_band[i - 1]
        prev_short = short_band[i - 1]
        prev_sm = smoothed_rsi.iloc[i - 1]

        if prev_sm > prev_long and sm > prev_long:
            long_band[i] = max(prev_long, new_long)
        else:
            long_band[i] = new_long

        if prev_sm < prev_short and sm < prev_short:
            short_band[i] = min(prev_short, new_short)
        else:
            short_band[i] = new_short

    long_band_series = pd.Series(long_band, index=smoothed_rsi.index)
    short_band_series = pd.Series(short_band, index=smoothed_rsi.index)

    prev_short = short_band_series.shift(1)
    prev_long = long_band_series.shift(1)
    prev_sm = smoothed_rsi.shift(1)

    cross_up = (prev_sm < prev_short) & (smoothed_rsi > prev_short)
    long_band_cross = (prev_sm > prev_long) & (smoothed_rsi < prev_long)

    trend_direction = pd.Series(0, index=smoothed_rsi.index, dtype=int)
    trend_direction.iloc[0] = 1
    for i in range(1, len(trend_direction)):
        if cross_up.iloc[i]:
            trend_direction.iloc[i] = 1
        elif long_band_cross.iloc[i]:
            trend_direction.iloc[i] = -1
        else:
            trend_direction.iloc[i] = trend_direction.iloc[i - 1]

    trend_line = pd.Series(
        np.where(trend_direction == 1, long_band_series, short_band_series),
        index=smoothed_rsi.index,
    )
    return trend_line, smoothed_rsi


def crossover(series: pd.Series, threshold: float) -> pd.Series:
    prev = series.shift(1)
    return (prev <= threshold) & (series > threshold)


def crossunder(series: pd.Series, threshold: float) -> pd.Series:
    prev = series.shift(1)
    return (prev >= threshold) & (series < threshold)


def crossover_series(left: pd.Series, right: pd.Series) -> pd.Series:
    return (left.shift(1) <= right.shift(1)) & (left > right)


def crossunder_series(left: pd.Series, right: pd.Series) -> pd.Series:
    return (left.shift(1) >= right.shift(1)) & (left < right)


def ut_trailing_stop(price: pd.Series, atr_series: pd.Series, atr_mult: float) -> pd.Series:
    stop = pd.Series(index=price.index, dtype=float)
    prev_stop = price.iloc[0]
    prev_price = price.iloc[0]
    for i, (p, atr_val) in enumerate(zip(price, atr_series.ffill())):
        atr_val = atr_val if np.isfinite(atr_val) else 0.0
        prev_stop_nz = prev_stop if np.isfinite(prev_stop) else 0.0
        iff_1 = p - atr_mult * atr_val if p > prev_stop_nz else p + atr_mult * atr_val
        if p < prev_stop_nz and prev_price < prev_stop_nz:
            iff_2 = min(prev_stop_nz, p + atr_mult * atr_val)
        else:
            iff_2 = iff_1
        if p > prev_stop_nz and prev_price > prev_stop_nz:
            current = max(prev_stop_nz, p - atr_mult * atr_val)
        else:
            current = iff_2
        stop.iloc[i] = current
        prev_stop = current
        prev_price = p
    return stop


class UTBotStrategy(Strategy):
    """Alpha UT 多空引擎 v3.0 的 Python 实现。"""

    name = "utbot"

    def prepare_data(self, price_df: pd.DataFrame) -> pd.DataFrame:
        if price_df.empty:
            raise ValueError("输入行情数据为空。")
        required_cols = {"high", "low", "close"}
        if not required_cols.issubset(price_df.columns):
            raise ValueError(f"行情数据必须包含 {required_cols} 列。")

        params = self.params
        df = price_df.copy().sort_index()

        stc_length = int(params.get("stc_length", 80))
        stc_fast = int(params.get("stc_fast", 26))
        stc_slow = int(params.get("stc_slow", 50))
        stc_smooth = float(params.get("stc_smooth", 0.5))
        stc_bull = float(params.get("stc_bull_threshold", 30.0))
        stc_bear = float(params.get("stc_bear_threshold", 70.0))
        use_stc = bool(params.get("use_stc_filter", True))
        use_qqe = bool(params.get("use_qqe_filter", True))
        require_both = bool(params.get("filter_require_both", False))
        enable_short = bool(params.get("enable_short", False))

        rsi_len_primary = int(params.get("rsi_primary_length", 6))
        rsi_smooth_primary = int(params.get("rsi_primary_smoothing", 5))
        qqe_factor_primary = float(params.get("qqe_primary_factor", 3.0))

        rsi_len_secondary = int(params.get("rsi_secondary_length", 6))
        rsi_smooth_secondary = int(params.get("rsi_secondary_smoothing", 5))
        qqe_factor_secondary = float(params.get("qqe_secondary_factor", 1.61))
        threshold_secondary = float(params.get("qqe_threshold", 3.0))

        boll_length = int(params.get("boll_length", 50))
        boll_mult = float(params.get("boll_multiplier", 0.35))

        atr_period = int(params.get("atr_period", 1))
        atr_mult = float(params.get("atr_multiplier", 3.0))

        close = df["close"]
        high = df["high"]
        low = df["low"]

        atr_series = atr(high, low, close, atr_period).ffill()
        df["atr"] = atr_series

        trailing_stop = ut_trailing_stop(close, atr_series, atr_mult)
        ema1 = ema(close, 1)
        above = crossover_series(ema1, trailing_stop)
        below = crossunder_series(ema1, trailing_stop)

        ut_up = (close > trailing_stop) & above
        ut_down = (close < trailing_stop) & below

        stc_series = compute_stc(close, stc_length, stc_fast, stc_slow, stc_smooth)
        stc_up = crossover(stc_series, stc_bull)
        stc_down = crossunder(stc_series, stc_bear)

        primary_trend, primary_rsi = compute_qqe(
            close,
            rsi_len_primary,
            rsi_smooth_primary,
            qqe_factor_primary,
        )
        secondary_trend, secondary_rsi = compute_qqe(
            close,
            rsi_len_secondary,
            rsi_smooth_secondary,
            qqe_factor_secondary,
        )

        basis_input = primary_trend - 50.0
        boll_basis = basis_input.rolling(boll_length).mean()
        boll_std = basis_input.rolling(boll_length).std(ddof=0)
        boll_upper = boll_basis + boll_std * boll_mult
        boll_lower = boll_basis - boll_std * boll_mult

        qqe_up = (secondary_rsi - 50.0 > threshold_secondary) & (
            primary_rsi - 50.0 > boll_upper
        )
        qqe_down = (secondary_rsi - 50.0 < -threshold_secondary) & (
            primary_rsi - 50.0 < boll_lower
        )

        filters_disabled = (not use_stc) and (not use_qqe)
        index = df.index
        if filters_disabled:
            long_filter_pass = pd.Series(True, index=df.index)
            short_filter_pass = pd.Series(True, index=df.index)
        elif require_both:
            stc_pass_long = stc_up if use_stc else pd.Series(True, index=index)
            qqe_pass_long = qqe_up if use_qqe else pd.Series(True, index=index)
            long_filter_pass = stc_pass_long & qqe_pass_long

            stc_pass_short = stc_down if use_stc else pd.Series(True, index=index)
            qqe_pass_short = qqe_down if use_qqe else pd.Series(True, index=index)
            short_filter_pass = stc_pass_short & qqe_pass_short
        else:
            stc_pass_long = stc_up if use_stc else pd.Series(False, index=index)
            qqe_pass_long = qqe_up if use_qqe else pd.Series(False, index=index)
            long_filter_pass = stc_pass_long | qqe_pass_long

            stc_pass_short = stc_down if use_stc else pd.Series(False, index=index)
            qqe_pass_short = qqe_down if use_qqe else pd.Series(False, index=index)
            short_filter_pass = stc_pass_short | qqe_pass_short

        if filters_disabled:
            long_filter_exit = pd.Series(False, index=index)
            short_filter_exit = pd.Series(False, index=index)
        else:
            long_filter_exit = short_filter_pass
            short_filter_exit = long_filter_pass

        long_entry = ut_up & long_filter_pass
        long_exit = ut_down | long_filter_exit
        if enable_short:
            short_entry = ut_down & short_filter_pass
            short_exit = ut_up | short_filter_exit
        else:
            short_entry = pd.Series(False, index=index)
            short_exit = ut_up

        df["long_entry"] = long_entry.astype(int)
        df["long_exit"] = long_exit.astype(int)
        df["short_entry"] = short_entry.astype(int)
        df["short_exit"] = short_exit.astype(int)
        df["ut_trailing_stop"] = trailing_stop
        df["stc"] = stc_series
        df["primary_qqe"] = primary_trend
        df["secondary_rsi"] = secondary_rsi

        return df

    def plot(
        self,
        data: pd.DataFrame,
        trades: Iterable[Trade],
        output_path: Path,
        equity_curve: pd.Series | None = None,
        initial_capital: float | None = None,
    ) -> Path:
        """沿用统一的资金曲线绘图逻辑。"""
        return plotting.plot_supertrend(
            data=data,
            trades=trades,
            equity_curve=equity_curve,
            initial_capital=initial_capital,
            output_path=output_path,
            title="UT Bot 策略资金曲线",
        )


__all__ = ["UTBotStrategy"]
