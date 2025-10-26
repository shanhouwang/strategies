"""Utility helpers for signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def crossover(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """Return True where series1 crosses above series2."""
    prev1 = series1.shift(1)
    prev2 = series2.shift(1)
    return (series1 > series2) & (prev1 <= prev2)


def crossunder(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """Return True where series1 crosses below series2."""
    prev1 = series1.shift(1)
    prev2 = series2.shift(1)
    return (series1 < series2) & (prev1 >= prev2)


def bars_since(condition: pd.Series) -> pd.Series:
    """Equivalent to Pine's ta.barssince for boolean series."""
    result = []
    last = None
    for idx, flag in enumerate(condition.astype(bool)):
        if flag:
            last = idx
            result.append(0)
        elif last is None:
            result.append(np.nan)
        else:
            result.append(idx - last)
    return pd.Series(result, index=condition.index, dtype=float)
