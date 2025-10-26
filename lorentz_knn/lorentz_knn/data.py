"""Data conversion utilities for the Lorentzian KNN strategy."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from pathlib import Path

import pandas as pd


def _get_attr(obj: Any, candidates: list[str]):
    for name in candidates:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(f"{obj} 缺少属性 {candidates}")


def candles_to_dataframe(candles: Iterable) -> pd.DataFrame:
    """Convert LongPort candle objects into a pandas DataFrame."""
    rows = []
    for candle in candles:
        timestamp = _get_attr(candle, ["timestamp", "time", "ts"])
        open_price = _get_attr(candle, ["open", "open_price"])
        high_price = _get_attr(candle, ["high", "high_price"])
        low_price = _get_attr(candle, ["low", "low_price"])
        close_price = _get_attr(candle, ["close", "close_price"])
        volume = _get_attr(candle, ["volume"])
        turnover = getattr(candle, "turnover", None)

        ts = (
            pd.to_datetime(timestamp, unit="s", utc=True).tz_convert("America/New_York")
            if isinstance(timestamp, (int, float))
            else pd.to_datetime(timestamp)
        )
        rows.append(
            {
                "timestamp": ts,
                "open": float(open_price),
                "high": float(high_price),
                "low": float(low_price),
                "close": float(close_price),
                "volume": float(volume),
                "turnover": float(turnover) if turnover is not None else None,
            }
        )

    frame = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    frame.set_index("timestamp", inplace=True)
    return frame

# 追加：本地CSV读写工具

def dataframe_to_csv(frame: pd.DataFrame, path: str | Path) -> None:
    """Save price dataframe to CSV including timestamp index."""
    frame.to_csv(path, index=True, index_label="timestamp")


def dataframe_from_csv(path: str | Path) -> pd.DataFrame:
    """Load price dataframe from CSV and restore timestamp index."""
    df = pd.read_csv(path, parse_dates=["timestamp"])  # tz-aware strings preserved
    df = df.sort_values("timestamp").reset_index(drop=True)
    df.set_index("timestamp", inplace=True)
    return df
