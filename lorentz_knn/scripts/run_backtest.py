#!/usr/bin/env python3
"""Run the Lorentzian KNN strategy backtest."""

from __future__ import annotations

import argparse
from datetime import datetime

from lorentz_knn.config import Settings
from lorentz_knn.pipeline import run_backtest_pipeline


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lorentzian KNN 策略回测脚本。")
    parser.add_argument("--symbol", help="标的代码，例如 AAPL.US。")
    parser.add_argument("--interval", help="K 线周期，例如 15m。")
    parser.add_argument("--start", help="起始时间（ISO8601）。")
    parser.add_argument("--end", help="结束时间（ISO8601）。")
    parser.add_argument("--neighbors", type=int, help="KNN 邻居数量。")
    parser.add_argument("--max-bars", type=int, help="历史最大回看长度。")
    parser.add_argument("--feature-count", type=int, help="启用的特征数量。")
    parser.add_argument("--initial-capital", type=float, help="初始资金。")
    parser.add_argument("--commission", type=float, help="单笔佣金。")
    parser.add_argument("--slippage", type=float, help="滑点。")
    parser.add_argument("--disable-short", action="store_true", help="禁止做空。")
    parser.add_argument("--moving-profit", action="store_true", help="启用移动止盈。")
    parser.add_argument("--partial-trailing", action="store_true", help="启用分批移动止盈。")
    parser.add_argument("--partial-stop", action="store_true", help="启用分批止损。")
    parser.add_argument("--stop-loss-only", action="store_true", help="仅使用固定止损。")
    parser.add_argument("--fixed-profit", action="store_true", help="使用固定盈亏比。")
    parser.add_argument("--atr-profit", action="store_true", help="使用 ATR 止盈止损。")
    # 新增：移动止盈参数（多头）
    parser.add_argument("--long_trailing_trigger_pct", type=float, help="多头移动止盈触发百分比。")
    parser.add_argument("--long_trailing_offset_pct", type=float, help="多头盈利回调幅度（百分比）。")
    # 新增：ADX 过滤参数
    parser.add_argument("--use_adx_filter", action="store_true", help="启用 ADX 过滤。")
    parser.add_argument("--adx_threshold", type=float, help="ADX 阈值。")
    # 新增：本地数据开关
    parser.add_argument("--use_local_data", action="store_true", help="使用本地CSV数据（若存在）。")
    parser.add_argument("--local_data_csv", help="指定本地CSV路径，默认 artifacts/data.csv")

    args = parser.parse_args()

    overrides = {}
    if args.symbol:
        overrides["symbol"] = args.symbol
    if args.interval:
        overrides["interval"] = args.interval
    if args.start:
        overrides["start"] = parse_datetime(args.start)
    if args.end:
        overrides["end"] = parse_datetime(args.end)
    if args.neighbors is not None:
        overrides["neighbors_count"] = args.neighbors
    if args.max_bars is not None:
        overrides["max_bars_back"] = args.max_bars
    if args.feature_count is not None:
        overrides["feature_count"] = args.feature_count
    if args.initial_capital is not None:
        overrides["initial_capital"] = args.initial_capital
    if args.commission is not None:
        overrides["commission"] = args.commission
    if args.slippage is not None:
        overrides["slippage"] = args.slippage
    if args.disable_short:
        overrides["open_short"] = False
    if args.moving_profit:
        overrides["enable_moving_profit"] = True
    if args.partial_trailing:
        overrides["enable_partial_trailing"] = True
    if args.partial_stop:
        overrides["enable_partial_stop_loss"] = True
    if args.stop_loss_only:
        overrides["enable_stop_loss_only"] = True
    if args.fixed_profit:
        overrides["enable_fixed_profit"] = True
    if args.atr_profit:
        overrides["enable_atr_profit"] = True
    # 新增：将 CLI 参数写入设置
    if getattr(args, "long_trailing_trigger_pct", None) is not None:
        overrides["long_trailing_trigger_pct"] = args.long_trailing_trigger_pct
    if getattr(args, "long_trailing_offset_pct", None) is not None:
        overrides["long_trailing_offset_pct"] = args.long_trailing_offset_pct
    # 新增：ADX 过滤设置
    if args.use_adx_filter:
        overrides["use_adx_filter"] = True
    if getattr(args, "adx_threshold", None) is not None:
        overrides["adx_threshold"] = args.adx_threshold
    # 新增：本地数据设置
    if args.use_local_data:
        overrides["use_local_data"] = True
    if getattr(args, "local_data_csv", None):
        overrides["local_data_csv"] = args.local_data_csv

    settings = Settings(**overrides)
    result = run_backtest_pipeline(settings)

    stats = result.stats
    print("回测完成。")
    print(f"成交笔数：{stats.trade_count}")
    print(f"总收益率：{stats.total_return:.2%}")
    print(f"年化收益率：{stats.annualized_return:.2%}")
    print(f"最大回撤：{stats.max_drawdown:.2%}")
    print(f"胜率：{stats.win_rate:.2%}")
    print(f"盈利因子：{stats.profit_factor:.2f}")
    final_profit = float(result.trade_log["pnl"].sum()) if not result.trade_log.empty else 0.0
    print(f"初始资金：{settings.initial_capital:.2f}")
    print(f"最终净利润：{final_profit:.2f}")
    print(f"净值曲线输出目录：{settings.artifacts_dir}")
    print(f"交易记录 CSV：{settings.trades_csv}")
    print(f"图表路径：{settings.chart_path}")


if __name__ == "__main__":
    main()
