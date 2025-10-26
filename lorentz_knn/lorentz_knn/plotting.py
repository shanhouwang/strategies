"""Plotting helpers for the Lorentzian KNN backtest pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import pandas as pd

from .backtest import Trade


_FONT_CONFIGURED = False


def _configure_chinese_font() -> None:
    """Set a Chinese-capable font for Matplotlib if available."""
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


def plot_strategy(
    data: pd.DataFrame,
    _trades: Iterable[Trade],
    equity_curve: Optional[pd.Series],
    settings_initial_capital: float,
    output_path: Path,
) -> None:
    """Plot strategy equity curve with buy-and-hold benchmark."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _configure_chinese_font()

    fig, ax = plt.subplots(figsize=(14, 6))

    has_equity = equity_curve is not None and not equity_curve.empty
    curves_plotted = 0

    if has_equity:
        ax.plot(
            equity_curve.index,
            equity_curve.values,
            color="teal",
            linewidth=1.4,
            label="策略资金曲线",
        )
        curves_plotted += 1

    if not data.empty and float(data["close"].iloc[0]) > 0:
        buy_hold_curve = settings_initial_capital * (data["close"] / data["close"].iloc[0])
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

    ax.set_title("Lorentzian KNN 资金曲线对比")
    ax.set_xlabel("时间")
    ax.set_ylabel("资金")
    if curves_plotted > 0:
        ax.legend(loc="best")
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
