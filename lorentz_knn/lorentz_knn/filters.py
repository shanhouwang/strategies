"""Filter utilities for refining KNN predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("未检测到 pandas-ta，请先安装后再运行策略。") from exc

from .config import Settings


def compute_volatility_filter(data: pd.DataFrame, window: int = 10) -> pd.Series:
    returns = data["close"].pct_change()
    short_vol = returns.rolling(window).std()
    long_vol = returns.rolling(window * 3).std()
    # 当短期波动率高于长期波动率时认为波动充分
    return (short_vol > long_vol).fillna(False)


def compute_regime_filter(
    data: pd.DataFrame, threshold_pct: float = -0.1, lookback: int = 4
) -> pd.Series:
    change = data["close"].pct_change(lookback) * 100
    return (change > threshold_pct).fillna(False)


def compute_adx_filter(
    data: pd.DataFrame, threshold: float = 20.0, length: int = 14
) -> pd.Series:
    adx_df = ta.adx(high=data["high"], low=data["low"], close=data["close"], length=length)
    if adx_df is None or adx_df.empty:
        raise RuntimeError("pandas-ta.adx 返回空结果，请检查输入数据。")
    adx_col = [col for col in adx_df.columns if col.startswith("ADX_")]
    adx_series = adx_df[adx_col[0]]
    return (adx_series > threshold).fillna(False)


def compute_trend_filters(data: pd.DataFrame, settings: Settings) -> tuple[pd.Series, pd.Series]:
    if settings.use_ema_filter:
        ema_series = ta.ema(data["close"], length=settings.ema_period)
        is_up = (data["close"] > ema_series).fillna(False)
        is_down = (data["close"] < ema_series).fillna(False)
    else:
        is_up = pd.Series(True, index=data.index)
        is_down = pd.Series(True, index=data.index)

    if settings.use_sma_filter:
        sma_series = ta.sma(data["close"], length=settings.sma_period)
        is_up = is_up & (data["close"] > sma_series).fillna(False)
        is_down = is_down & (data["close"] < sma_series).fillna(False)

    return is_up, is_down
