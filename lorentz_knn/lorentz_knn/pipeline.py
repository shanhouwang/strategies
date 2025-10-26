"""Backtest pipeline orchestrator for the Lorentzian KNN strategy."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from . import backtest, data as data_utils
from .config import Settings
from .indicator import build_indicator_dataframe
from .plotting import plot_strategy
from .longport_client import fetch_candles, quote_context


TRADE_LOG_HEADERS: List[Tuple[str, str]] = [
    ("entry_time", "开仓时间"),
    ("exit_time", "平仓时间"),
    ("entry_price", "开仓价格"),
    ("exit_price", "平仓价格"),
    ("quantity", "成交数量"),
    ("pnl", "净利润"),
    ("return_pct", "收益率"),
    ("bars_held", "持仓K线数"),
    ("direction", "方向"),
    ("exit_reason", "离场原因"),
]


def run_backtest_pipeline(settings: Settings) -> backtest.BacktestResult:
    """Load data, compute signals, execute backtest, and export artifacts."""
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 统一数据加载：本地CSV优先，否则调用 LongPort 并缓存到CSV
    price_df = None
    if settings.use_local_data and settings.local_data_csv and Path(settings.local_data_csv).exists():
        print("=== 使用本地CSV数据 ===")
        price_df = data_utils.dataframe_from_csv(Path(settings.local_data_csv))
        print(f"从本地加载 {len(price_df)} 条K线：{settings.local_data_csv}")
    else:
        print("=== 尝试连接 LongPort API 获取真实数据 ===")
        with quote_context(settings) as ctx:
            candles = fetch_candles(
                ctx,
                settings.symbol,
                settings.interval,
                settings.start,
                settings.end,
            )
        print("✅ 成功从 LongPort API 获取真实K线数据")

        if not candles:
            raise RuntimeError("LongPort 未返回任何 K 线，请检查标的或时间区间。")

        print("=== K 线数据信息 ===")
        print(f"获取到 {len(candles)} 条 K 线数据")
        if candles:
            print(f"数据类型: {type(candles[0])}")
            print(f"第一条数据: {candles[0]}")
            print(f"最后一条数据: {candles[-1]}")

        price_df = data_utils.candles_to_dataframe(candles)
        # 写入本地CSV缓存，便于后续 --use_local_data 复用
        try:
            Path(settings.local_data_csv).parent.mkdir(parents=True, exist_ok=True)
            data_utils.dataframe_to_csv(price_df, Path(settings.local_data_csv))
            print(f"已保存本地数据CSV：{settings.local_data_csv}")
        except Exception as e:
            print(f"保存本地CSV失败：{e}")

    # 后续处理沿用原逻辑
    print("=== DataFrame 总览 ===")
    print(price_df.head(3))
    print(price_df.tail(3))

    indicator_df = build_indicator_dataframe(price_df, settings)

    result = backtest.run_backtest(indicator_df, settings)

    if not result.trade_log.empty:
        trade_log = result.trade_log.copy()
        english_order = [col for col, _ in TRADE_LOG_HEADERS if col in trade_log.columns]
        if english_order:
            trade_log = trade_log[english_order]
        header_map = {eng: zh for eng, zh in TRADE_LOG_HEADERS}
        trade_log = trade_log.rename(columns=header_map)
        trade_log.to_csv(settings.trades_csv, index=False)
    else:
        settings.trades_csv.write_text("未产生任何交易。\n")

    plot_strategy(
        indicator_df,
        result.trades,
        result.equity_curve,
        settings.initial_capital,
        Path(settings.chart_path),
    )
    return result
