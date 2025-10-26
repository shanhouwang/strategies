"""技术指标计算（依赖 pandas-ta）。"""

from __future__ import annotations

import pandas as pd

try:
    import pandas_ta as ta
except ImportError as exc:  # pragma: no cover - 运行时导入检查
    raise RuntimeError("未检测到 pandas-ta，请先安装后再运行策略。") from exc


def supertrend(
    data: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
    atr_length: int | None = None,
) -> pd.DataFrame:
    """通过 pandas-ta 计算超级趋势指标并生成交易信号。"""
    required_cols = {"high", "low", "close"}
    if not required_cols.issubset(data.columns):
        raise ValueError(f"输入数据必须包含 {required_cols} 列。")

    if data.empty:
        raise ValueError("输入数据为空，无法计算超级趋势。")

    base = data.copy()
    st_df = ta.supertrend(
        high=base["high"],
        low=base["low"],
        close=base["close"],
        length=period,
        multiplier=multiplier,
    )

    if st_df is None or st_df.empty:
        raise RuntimeError("pandas-ta 返回空的超级趋势结果，请检查输入数据。")

    rename_map = {}
    for col in st_df.columns:
        lower_col = col.lower()
        if "supertd" in lower_col:
            rename_map[col] = "supertrend_direction"
        elif "supertl" in lower_col:
            rename_map[col] = "supertrend_lower"
        elif "superts" in lower_col:
            rename_map[col] = "supertrend_upper"
        elif lower_col.startswith("supert_") or "supertr" in lower_col:
            rename_map[col] = "supertrend"
        else:
            rename_map[col] = col
    st_df = st_df.rename(columns=rename_map)

    required_outputs = {
        "supertrend",
        "supertrend_direction",
        "supertrend_lower",
        "supertrend_upper",
    }
    missing_outputs = required_outputs - set(st_df.columns)
    if missing_outputs:
        raise RuntimeError(f"超级趋势输出缺少列：{missing_outputs}")

    result = base.join(st_df, how="left")
    atr_len = atr_length or period
    if atr_len <= 0:
        raise ValueError("ATR 周期必须为正整数。")
    atr_series = ta.atr(
        high=base["high"],
        low=base["low"],
        close=base["close"],
        length=atr_len,
    )
    result["atr"] = atr_series
    dir_series = result["supertrend_direction"].fillna(method="ffill").fillna(1.0)
    direction = (dir_series > 0).astype(int)

    if not direction.empty:
        prev_dir = direction.shift(1, fill_value=direction.iloc[0])
    else:
        prev_dir = direction

    result["supertrend_direction"] = direction
    result["long_entry"] = ((direction == 1) & (prev_dir == 0)).astype(int)
    result["long_exit"] = ((direction == 0) & (prev_dir == 1)).astype(int)
    result["short_entry"] = ((direction == 0) & (prev_dir == 1)).astype(int)
    result["short_exit"] = ((direction == 1) & (prev_dir == 0)).astype(int)

    return result
