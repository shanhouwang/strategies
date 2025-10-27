"""海龟交易法则策略实现（基于 QuantConnect/Lean 思路）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    import pandas_ta as ta
except ImportError as exc:  # pragma: no cover - runtime guard
    raise RuntimeError("未检测到 pandas-ta，请先安装后再运行海龟策略。") from exc

from .. import plotting
from ..backtest import Trade
from .base import Strategy


class TurtleStrategy(Strategy):
    """基于唐奇安通道突破的海龟策略（支持双系统参数）。"""

    name = "turtle"

    def prepare_data(self, price_df: pd.DataFrame) -> pd.DataFrame:
        if price_df is None or price_df.empty:
            raise ValueError("输入数据为空，无法计算海龟策略指标。")

        required_cols = {"high", "low", "close"}
        if not required_cols.issubset(price_df.columns):
            raise ValueError(f"海龟策略需要包含 {required_cols} 列的数据。")

        params = self.params

        def _coerce_bool(value: Any, *, default: bool | None = None) -> bool:
            if value is None:
                if default is None:
                    raise ValueError("布尔参数缺少默认值。")
                return default
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "1", "yes", "y", "t"}:
                    return True
                if lowered in {"false", "0", "no", "n", "f"}:
                    return False
            raise ValueError("use_secondary_system 参数必须是布尔值。")

        entry_window_primary = int(params.get("entry_window", 20))
        exit_window_primary = int(params.get("exit_window", 10))
        entry_window_secondary = int(params.get("secondary_entry_window", 55))
        exit_window_secondary = int(params.get("secondary_exit_window", 20))
        use_secondary = _coerce_bool(params.get("use_secondary_system"), default=True)
        enable_long = _coerce_bool(params.get("enable_long"), default=self.settings.enable_long)
        enable_short = _coerce_bool(
            params.get("enable_short"), default=self.settings.enable_short
        )
        atr_period = int(params.get("atr_period", 20))

        atr_multiplier_param = params.get("atr_multiplier")
        if atr_multiplier_param is not None:
            atr_multiplier = float(atr_multiplier_param)
        elif self.settings.atr_stop_multiplier > 0:
            atr_multiplier = float(self.settings.atr_stop_multiplier)
        else:
            atr_multiplier = 2.0

        if entry_window_primary <= 1 or exit_window_primary <= 1:
            raise ValueError("entry_window 与 exit_window 必须大于 1。")
        if use_secondary and (
            entry_window_secondary <= 1 or exit_window_secondary <= 1
        ):
            raise ValueError("secondary_entry_window 与 secondary_exit_window 必须大于 1。")
        if atr_period <= 1:
            raise ValueError("atr_period 必须大于 1。")
        if atr_multiplier <= 0:
            raise ValueError("atr_multiplier 必须大于 0。")

        trend_period = int(params.get("trend_filter_period", self.settings.trend_filter_period))
        min_volatility_pct = float(
            params.get(
                "volatility_filter_atr_pct",
                self.settings.volatility_filter_atr_pct,
            )
            or 0.0
        )

        if atr_multiplier_param is not None:
            self.settings.atr_stop_multiplier = atr_multiplier

        result = price_df.copy()

        def rolling_high(series: pd.Series, window: int) -> pd.Series:
            return series.rolling(window=window, min_periods=window).max().shift(1)

        def rolling_low(series: pd.Series, window: int) -> pd.Series:
            return series.rolling(window=window, min_periods=window).min().shift(1)

        high = result["high"]
        low = result["low"]
        close = result["close"]

        entry_high_primary = rolling_high(high, entry_window_primary)
        entry_low_primary = rolling_low(low, entry_window_primary)
        exit_high_primary = rolling_high(high, exit_window_primary)
        exit_low_primary = rolling_low(low, exit_window_primary)

        result["turtle_entry_high_primary"] = entry_high_primary
        result["turtle_entry_low_primary"] = entry_low_primary
        result["turtle_exit_high_primary"] = exit_high_primary
        result["turtle_exit_low_primary"] = exit_low_primary

        long_entry_primary = (
            (close > entry_high_primary) & entry_high_primary.notna()
        ).astype(int)
        short_entry_primary = (
            (close < entry_low_primary) & entry_low_primary.notna()
        ).astype(int)
        long_exit_primary = (
            (close < exit_low_primary) & exit_low_primary.notna()
        ).astype(int)
        short_exit_primary = (
            (close > exit_high_primary) & exit_high_primary.notna()
        ).astype(int)

        long_entry = long_entry_primary.copy()
        short_entry = short_entry_primary.copy()
        long_exit = long_exit_primary.copy()
        short_exit = short_exit_primary.copy()

        if use_secondary:
            entry_high_secondary = rolling_high(high, entry_window_secondary)
            entry_low_secondary = rolling_low(low, entry_window_secondary)
            exit_high_secondary = rolling_high(high, exit_window_secondary)
            exit_low_secondary = rolling_low(low, exit_window_secondary)

            result["turtle_entry_high_secondary"] = entry_high_secondary
            result["turtle_entry_low_secondary"] = entry_low_secondary
            result["turtle_exit_high_secondary"] = exit_high_secondary
            result["turtle_exit_low_secondary"] = exit_low_secondary

            long_entry_secondary = (
                (close > entry_high_secondary) & entry_high_secondary.notna()
            ).astype(int)
            short_entry_secondary = (
                (close < entry_low_secondary) & entry_low_secondary.notna()
            ).astype(int)
            long_exit_secondary = (
                (close < exit_low_secondary) & exit_low_secondary.notna()
            ).astype(int)
            short_exit_secondary = (
                (close > exit_high_secondary) & exit_high_secondary.notna()
            ).astype(int)

            long_entry = ((long_entry_primary == 1) | (long_entry_secondary == 1)).astype(
                int
            )
            short_entry = (
                (short_entry_primary == 1) | (short_entry_secondary == 1)
            ).astype(int)
            long_exit = ((long_exit_primary == 1) | (long_exit_secondary == 1)).astype(
                int
            )
            short_exit = (
                (short_exit_primary == 1) | (short_exit_secondary == 1)
            ).astype(int)

        if trend_period > 0:
            trend_ma = close.ewm(span=trend_period, adjust=False).mean()
            long_trend_mask = close > trend_ma
            short_trend_mask = close < trend_ma
            result["turtle_trend_ma"] = trend_ma
        else:
            long_trend_mask = short_trend_mask = pd.Series(
                True, index=result.index, dtype=bool
            )

        result["long_entry"] = long_entry
        result["short_entry"] = short_entry
        result["long_exit"] = long_exit
        result["short_exit"] = short_exit

        atr = ta.atr(high=high, low=low, close=close, length=atr_period)
        result["atr"] = atr
        result["turtle_atr_multiplier"] = atr_multiplier
        result["turtle_long_stop"] = close - atr * atr_multiplier
        result["turtle_short_stop"] = close + atr * atr_multiplier

        if min_volatility_pct > 0:
            with pd.option_context("mode.use_inf_as_na", True):
                vol_ratio = atr / close.replace(0, pd.NA)
            vol_mask = vol_ratio.fillna(0.0) >= min_volatility_pct
            result["turtle_volatility_ratio"] = vol_ratio
        else:
            vol_mask = pd.Series(True, index=result.index, dtype=bool)

        final_long_mask = long_trend_mask & vol_mask & enable_long
        final_short_mask = short_trend_mask & vol_mask & enable_short

        result["long_entry"] = (result["long_entry"] & final_long_mask).astype(int)
        result["short_entry"] = (result["short_entry"] & final_short_mask).astype(int)

        if not enable_long:
            result["long_exit"] = 0
        else:
            result["long_exit"] = result["long_exit"].astype(int)
        if not enable_short:
            result["short_exit"] = 0
        else:
            result["short_exit"] = result["short_exit"].astype(int)

        return result

    def plot(
        self,
        data: pd.DataFrame,
        trades: Iterable[Trade],
        output_path: Path,
        equity_curve: pd.Series | None = None,
        initial_capital: float | None = None,
    ) -> Path:
        """输出与超级趋势相同的资金曲线图。"""
        return plotting.plot_supertrend(
            data=data,
            trades=trades,
            equity_curve=equity_curve,
            initial_capital=initial_capital,
            output_path=output_path,
        )


__all__ = ["TurtleStrategy"]
