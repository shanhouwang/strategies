"""绘图工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import pandas as pd

from .backtest import Trade


_FONT_CONFIGURED = False


def _configure_chinese_font() -> None:
    """确保 Matplotlib 使用支持中文的字体。"""
    global _FONT_CONFIGURED
    if _FONT_CONFIGURED:
        return
    preferred_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "PingFang HK",
        "PingFang SC",
        "Hiragino Sans GB",
        "WenQuanYi Micro Hei",
        "Sarasa Gothic SC",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in available:
            plt.rcParams["font.family"] = [font_name]
            plt.rcParams["font.sans-serif"] = [font_name]
            break
    plt.rcParams["axes.unicode_minus"] = False
    _FONT_CONFIGURED = True


def plot_supertrend(
    data: pd.DataFrame,
    trades: Iterable[Trade],  # 保留参数以兼容调用方，当前未使用
    equity_curve: pd.Series | None,
    initial_capital: float | None,
    output_path: Path,
    title: str | None = None,
) -> Path:
    """绘制策略资金曲线与买入持有基准。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _configure_chinese_font()

    fig, ax = plt.subplots(figsize=(14, 6))

    curves_plotted = 0
    if equity_curve is not None and not equity_curve.empty:
        ax.plot(
            equity_curve.index,
            equity_curve.values,
            color="teal",
            linewidth=1.4,
            label="策略资金曲线",
        )
        curves_plotted += 1

    if (
        initial_capital is not None
        and initial_capital > 0
        and not data.empty
        and "close" in data.columns
        and float(data["close"].iloc[0]) > 0
    ):
        buy_hold_curve = initial_capital * (
            data["close"] / data["close"].iloc[0]
        )
        ax.plot(
            data.index,
            buy_hold_curve,
            color="purple",
            linewidth=1.1,
            linestyle="--",
            label="买入持有曲线",
        )
        curves_plotted += 1

    if curves_plotted == 0:
        ax.text(
            0.5,
            0.5,
            "无可绘制的资金曲线数据",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
        )

    ax.set_title(title or "超级趋势策略资金曲线")
    ax.set_xlabel("时间")
    ax.set_ylabel("资金")
    if curves_plotted > 0:
        ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path
