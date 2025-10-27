"""策略接口定义。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, TYPE_CHECKING

import pandas as pd

from ..backtest import Trade

if TYPE_CHECKING:
    from ..config import Settings


class Strategy(ABC):
    """策略抽象基类。"""

    name: str = "base"

    def __init__(self, *, settings: "Settings", **params: Any) -> None:
        self.settings = settings
        self.params: Dict[str, Any] = params

    @abstractmethod
    def prepare_data(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """根据行情数据生成回测所需指标与信号。"""

    def plot(
        self,
        data: pd.DataFrame,
        trades: Iterable[Trade],
        output_path: Path,
        equity_curve: pd.Series | None = None,
        initial_capital: float | None = None,
    ) -> Path | None:
        """绘制回测结果，默认不输出图像。"""
        return None

    def default_chart_path(self) -> Path:
        """默认图表输出路径，可被具体策略覆盖。"""
        return Path(self.settings.artifacts_dir) / f"{self.name}.png"


__all__ = ["Strategy"]
