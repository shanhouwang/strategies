"""超级趋势回测流程编排。"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

from . import backtest, data as data_utils, indicators, plotting
from .config import Settings
from .longport_client import fetch_candles, quote_context



def run_backtest_pipeline(settings: Settings) -> backtest.BacktestResult:
    """拉取数据、执行超级趋势回测并生成输出。"""
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)

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

    print(f"=== K 线数据信息 ===")
    print(f"获取到 {len(candles)} 条 K 线数据")
    if candles:
        print(f"数据类型: {type(candles[0])}")
        print(f"第一条数据: {candles[0]}")
        print(f"最后一条数据: {candles[-1]}")

    price_df = data_utils.candles_to_dataframe(candles)
    
    print(f"\n=== DataFrame 信息 ===")
    print(f"DataFrame 形状: {price_df.shape}")
    print(f"列名: {list(price_df.columns)}")
    print(f"数据类型:\n{price_df.dtypes}")
    print(f"时间范围: {price_df.index.min()} 到 {price_df.index.max()}")
    print(f"\n前 5 行数据:")
    print(price_df.head())
    print(f"\n后 5 行数据:")
    print(price_df.tail())
    print(f"\n基本统计信息:")
    print(price_df.describe())
    
    indicator_df = indicators.supertrend(
        price_df,
        period=settings.supertrend_period,
        multiplier=settings.supertrend_multiplier,
        atr_length=settings.atr_period or settings.supertrend_period,
    )

    result = backtest.run_backtest(indicator_df, settings)

    if not result.trade_log.empty:
        def translate_exit_mode(reason: str) -> str:
            if isinstance(reason, str):
                if "止损" in reason:
                    return "止损出场"
                if "止盈" in reason:
                    return "止盈出场"
            return "正常方式"

        enriched_log = result.trade_log.copy()
        enriched_log["exit_mode"] = enriched_log["exit_reason"].apply(
            translate_exit_mode
        )
        rename_map = {
            "entry_time": "开仓时间",
            "exit_time": "平仓时间",
            "entry_price": "开仓价格",
            "exit_price": "平仓价格",
            "quantity": "成交数量",
            "pnl": "净利润",
            "return_pct": "收益率",
            "bars_held": "持仓K线数",
            "direction": "方向",
            "exit_reason": "离场原因",
            "exit_mode": "出场方式",
        }
        export_columns = [
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "quantity",
            "pnl",
            "return_pct",
            "bars_held",
            "direction",
            "exit_reason",
            "exit_mode",
        ]
        trade_log = enriched_log[export_columns].rename(columns=rename_map)
        trade_log.to_csv(settings.trades_csv, index=False)
        total_profit = enriched_log["pnl"].sum()
        print(f"总净利润：{total_profit:.2f}")
    else:
        settings.trades_csv.write_text("未产生任何交易。\n")

    plotting.plot_supertrend(indicator_df, result.trades, Path(settings.chart_path))

    return result
