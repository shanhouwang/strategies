"""Translate market data into trading signals following the Pine script logic."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("未检测到 pandas-ta，请先安装后再运行策略。") from exc

from .config import Settings
from .features import compute_atr, compute_feature_matrix
from .filters import (
    compute_adx_filter,
    compute_regime_filter,
    compute_trend_filters,
    compute_volatility_filter,
)
from .knn import compute_knn_predictions
from .utils import crossover, crossunder


def _rational_quadratic_kernel(
    series: pd.Series, window: int, relative_weight: float, regression_level: int
) -> pd.Series:
    values = series.to_numpy(dtype=float)
    result = np.full_like(values, fill_value=np.nan)
    r = relative_weight
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        weights = []
        items = []
        for j in range(start, idx + 1):
            dist = (idx - j) / max(regression_level, 1)
            weight = (1.0 + (dist**2) / (2.0 * r)) ** (-r)
            weights.append(weight)
            items.append(values[j])
        denom = float(np.sum(weights))
        if denom == 0:
            result[idx] = np.nan
        else:
            result[idx] = float(np.dot(weights, items) / denom)
    return pd.Series(result, index=series.index, name="kernel_rq")


def _gaussian_kernel(series: pd.Series, window: int) -> pd.Series:
    if window <= 0:
        window = 2
    values = series.to_numpy(dtype=float)
    result = np.full_like(values, fill_value=np.nan)
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        weights = []
        items = []
        for j in range(start, idx + 1):
            dist = idx - j
            weight = math.exp(-0.5 * (dist / window) ** 2)
            weights.append(weight)
            items.append(values[j])
        denom = float(np.sum(weights))
        if denom == 0:
            result[idx] = np.nan
        else:
            result[idx] = float(np.dot(weights, items) / denom)
    return pd.Series(result, index=series.index, name="kernel_gaussian")


def build_indicator_dataframe(
    price_df: pd.DataFrame, settings: Settings
) -> pd.DataFrame:
    """Create the enriched dataframe that feeds the backtest."""
    if price_df.empty:
        raise ValueError("输入数据为空，无法计算策略逻辑。")

    data = price_df.copy()
    feature_df = compute_feature_matrix(data, settings)
    atr_series = compute_atr(data, length=settings.atr_period)

    future_close = data["close"].shift(-4)
    label_series = pd.Series(
        np.where(
            future_close > data["close"],
            1.0,
            np.where(future_close < data["close"], -1.0, 0.0),
        ),
        index=data.index,
        name="label",
    )

    prediction_series = compute_knn_predictions(feature_df, label_series, settings)
    prediction_series = prediction_series.fillna(0.0)

    vol_filter = (
        compute_volatility_filter(data) if settings.use_volatility_filter else pd.Series(True, index=data.index)
    )
    regime_filter = (
        compute_regime_filter(data, settings.regime_threshold)
        if settings.use_regime_filter
        else pd.Series(True, index=data.index)
    )
    adx_filter = (
        compute_adx_filter(data, settings.adx_threshold)
        if settings.use_adx_filter
        else pd.Series(True, index=data.index)
    )
    filter_all = (vol_filter & regime_filter & adx_filter).rename("filter_all")

    # Sequential signal similar to Pine's assignment
    signal_values = []
    prev_signal = 0.0
    for idx in range(len(prediction_series)):
        if filter_all.iat[idx]:
            if prediction_series.iat[idx] > 0:
                current = 1.0
            elif prediction_series.iat[idx] < 0:
                current = -1.0
            else:
                current = prev_signal
        else:
            current = prev_signal
        signal_values.append(current)
        prev_signal = current
    signal_series = pd.Series(signal_values, index=data.index, name="signal")

    ema_uptrend, ema_downtrend = compute_trend_filters(data, settings)
    is_buy_signal = (signal_series == 1) & ema_uptrend
    is_sell_signal = (signal_series == -1) & ema_downtrend
    is_different_signal = signal_series.ne(signal_series.shift(1).fillna(0))
    is_new_buy = (is_buy_signal & is_different_signal).rename("is_new_buy")
    is_new_sell = (is_sell_signal & is_different_signal).rename("is_new_sell")

    # Kernel regression filters
    if settings.use_kernel_filter:
        rq_kernel = _rational_quadratic_kernel(
            data["close"],
            settings.kernel_window,
            settings.kernel_relative_weight,
            settings.kernel_regression_level,
        )
        gaussian_kernel = _gaussian_kernel(
            data["close"], max(2, settings.kernel_window - settings.kernel_lag)
        )
        was_bearish_rate = rq_kernel.shift(2) > rq_kernel.shift(1)
        was_bullish_rate = rq_kernel.shift(2) < rq_kernel.shift(1)
        is_bearish_rate = rq_kernel.shift(1) > rq_kernel
        is_bullish_rate = rq_kernel.shift(1) < rq_kernel
        is_bearish_change = is_bearish_rate & was_bullish_rate
        is_bullish_change = is_bullish_rate & was_bearish_rate
        is_bullish_smooth = gaussian_kernel >= rq_kernel
        is_bearish_smooth = gaussian_kernel <= rq_kernel
        bullish_cross = crossover(gaussian_kernel, rq_kernel)
        bearish_cross = crossunder(gaussian_kernel, rq_kernel)
    else:
        rq_kernel = pd.Series(np.nan, index=data.index, name="kernel_rq")
        gaussian_kernel = pd.Series(np.nan, index=data.index, name="kernel_gaussian")
        is_bearish_change = pd.Series(False, index=data.index)
        is_bullish_change = pd.Series(False, index=data.index)
        is_bullish_smooth = pd.Series(True, index=data.index)
        is_bearish_smooth = pd.Series(True, index=data.index)
        is_bearish_rate = pd.Series(True, index=data.index)
        is_bullish_rate = pd.Series(True, index=data.index)
        bullish_cross = pd.Series(False, index=data.index)
        bearish_cross = pd.Series(False, index=data.index)

    if settings.use_kernel_filter:
        if settings.use_kernel_smoothing:
            bullish_filter = is_bullish_smooth
            bearish_filter = is_bearish_smooth
        else:
            bullish_filter = is_bullish_rate
            bearish_filter = is_bearish_rate
    else:
        bullish_filter = pd.Series(True, index=data.index)
        bearish_filter = pd.Series(True, index=data.index)

    start_long = (
        is_new_buy & bullish_filter & ema_uptrend
    ).rename("start_long")
    start_short = (
        is_new_sell & bearish_filter & ema_downtrend
    ).rename("start_short")

    # ADX for entry condition (using pandas-ta)
    adx_df = ta.adx(
        high=data["high"], low=data["low"], close=data["close"], length=14
    )
    adx_col = [col for col in adx_df.columns if col.startswith("ADX_")]
    adx_series = adx_df[adx_col[0]].rename("adx") if adx_col else pd.Series(np.nan, index=data.index, name="adx")

    enriched = data.copy()
    enriched = enriched.join(feature_df, how="left")
    enriched["prediction"] = prediction_series
    enriched["signal"] = signal_series
    enriched["filter_volatility"] = vol_filter.astype(bool)
    enriched["filter_regime"] = regime_filter.astype(bool)
    enriched["filter_adx"] = adx_filter.astype(bool)
    enriched["filter_all"] = filter_all.astype(bool)
    enriched["is_buy_signal"] = is_buy_signal.astype(bool)
    enriched["is_sell_signal"] = is_sell_signal.astype(bool)
    enriched["start_long"] = start_long.astype(bool)
    enriched["start_short"] = start_short.astype(bool)
    enriched["kernel_rq"] = rq_kernel
    enriched["kernel_gaussian"] = gaussian_kernel
    enriched["kernel_bullish_change"] = is_bullish_change.astype(bool)
    enriched["kernel_bearish_change"] = is_bearish_change.astype(bool)
    enriched["kernel_bullish_cross"] = bullish_cross.astype(bool)
    enriched["kernel_bearish_cross"] = bearish_cross.astype(bool)
    enriched["atr"] = atr_series
    enriched["adx"] = adx_series
    enriched["label"] = label_series
    return enriched
