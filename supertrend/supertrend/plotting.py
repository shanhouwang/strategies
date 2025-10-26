"""绘图工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from .backtest import Trade


def plot_supertrend(
    data: pd.DataFrame,
    trades: Iterable[Trade],
    output_path: Path,
) -> Path:
    """绘制价格、超级趋势通道与交易点。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    trade_list = list(trades)

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(data.index, data["close"], label="收盘价", color="black", linewidth=1.2)
    ax.plot(
        data.index,
        data["supertrend_upper"],
        label="超级趋势上轨",
        color="red",
        linestyle="--",
    )
    ax.plot(
        data.index,
        data["supertrend_lower"],
        label="超级趋势下轨",
        color="green",
        linestyle="--",
    )
    ax.plot(data.index, data["supertrend"], label="超级趋势", color="blue", linewidth=1.0)

    for idx, trade in enumerate(trade_list):
        ax.scatter(
            trade.entry_time,
            trade.entry_price,
            color="green",
            marker="^",
            s=80,
            label="开仓" if idx == 0 else "",
        )
        ax.scatter(
            trade.exit_time,
            trade.exit_price,
            color="red",
            marker="v",
            s=80,
            label="平仓" if idx == 0 else "",
        )

    ax.set_title("超级趋势策略回测")
    ax.set_xlabel("时间")
    ax.set_ylabel("价格")
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.5)
    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path
