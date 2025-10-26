"""Feature engineering helpers mirroring the TradingView indicator options."""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

try:
    import pandas_ta as ta
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("未检测到 pandas-ta，请先安装后再运行策略。") from exc

from .config import FeatureConfig, Settings


def _rsi(close: pd.Series, length: int) -> pd.Series:
    return ta.rsi(close=close, length=length)


def _wt(
    high: pd.Series, low: pd.Series, close: pd.Series, channel_length: int, avg_length: int
) -> pd.Series:
    # Prefer pandas-ta built-in if available
    if hasattr(ta, "wt"):
        wt_df = ta.wt(high=high, low=low, close=close, channel_length=channel_length, average_length=avg_length)
        if wt_df is None or wt_df.empty:
            raise RuntimeError("pandas-ta.wt 返回空结果，请检查输入数据。")
        return wt_df.iloc[:, 0]

    # Fallback WaveTrend (WT) implementation compatible with pandas-ta 0.4.x
    hlc3 = (high + low + close) / 3.0
    esa = ta.ema(hlc3, length=channel_length)
    de = ta.ema((hlc3 - esa).abs(), length=channel_length)
    denom = 0.015 * de.replace(0, np.nan)
    ci = (hlc3 - esa) / denom
    wt1 = ta.ema(ci, length=avg_length)
    return wt1.rename("WT")


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    return ta.cci(high=high, low=low, close=close, length=length)


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    adx_df = ta.adx(high=high, low=low, close=close, length=length)
    if adx_df is None or adx_df.empty:
        raise RuntimeError("pandas-ta.adx 返回空结果，请检查输入数据。")
    # ADX column name pattern: ADX_{length}
    adx_col = [col for col in adx_df.columns if col.startswith("ADX_")]
    if not adx_col:
        raise RuntimeError("pandas-ta.adx 输出缺少 ADX 列。")
    return adx_df[adx_col[0]]


FEATURE_BUILDERS = {
    "RSI": lambda data, cfg: _rsi(data["close"], cfg.param_a),
    "WT": lambda data, cfg: _wt(data["high"], data["low"], data["close"], cfg.param_a, cfg.param_b),
    "CCI": lambda data, cfg: _cci(data["high"], data["low"], data["close"], cfg.param_a),
    "ADX": lambda data, cfg: _adx(data["high"], data["low"], data["close"], cfg.param_a),
}


def compute_feature_matrix(data: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Compute the selected feature series and assemble into a matrix."""
    rows = {}
    for idx, cfg in enumerate(settings.features[: settings.feature_count], start=1):
        key = f"f{idx}"
        builder = FEATURE_BUILDERS[cfg.name]
        series = builder(data, cfg).astype(float)
        rows[key] = series

    feature_df = pd.DataFrame(rows, index=data.index)
    return feature_df


def compute_atr(data: pd.DataFrame, length: int) -> pd.Series:
    atr_series = ta.atr(high=data["high"], low=data["low"], close=data["close"], length=length)
    return atr_series.rename("atr")
