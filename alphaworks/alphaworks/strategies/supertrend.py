"""超级趋势策略实现。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from .. import indicators, plotting
from ..backtest import Trade
from .base import Strategy


class SupertrendStrategy(Strategy):
    """沿用现有超级趋势逻辑的策略实现。"""

    name = "supertrend"

    def prepare_data(self, price_df: pd.DataFrame) -> pd.DataFrame:
        params = self.params
        period = int(params.get("period", self.settings.supertrend_period))
        multiplier = float(params.get("multiplier", self.settings.supertrend_multiplier))

        atr_value = params.get("atr_length", params.get("atr_period"))
        if atr_value is not None:
            atr_length = int(atr_value)
        else:
            atr_length = self.settings.atr_period or period

        return indicators.supertrend(
            price_df,
            period=period,
            multiplier=multiplier,
            atr_length=atr_length,
        )

    def plot(
        self,
        data: pd.DataFrame,
        trades: Iterable[Trade],
        output_path: Path,
        equity_curve: pd.Series | None = None,
        initial_capital: float | None = None,
    ) -> Path:
        return plotting.plot_supertrend(
            data=data,
            trades=trades,
            equity_curve=equity_curve,
            initial_capital=initial_capital,
            output_path=output_path,
        )


__all__ = ["SupertrendStrategy"]
