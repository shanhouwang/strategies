#!/usr/bin/env python3
"""基于 LongPort OpenAPI 运行 SOXL 超级趋势回测。"""

from __future__ import annotations

import argparse
from datetime import datetime

from supertrend.config import Settings
from supertrend.pipeline import run_backtest_pipeline


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def build_settings(args: argparse.Namespace) -> Settings:
    kwargs = {
        "symbol": args.symbol,
        "interval": args.interval,
        "start": parse_datetime(args.start),
        "end": parse_datetime(args.end),
        "supertrend_period": args.period,
        "supertrend_multiplier": args.multiplier,
        "initial_capital": args.initial_capital,
        "commission": args.commission,
        "slippage": args.slippage,
    }
    if args.enable_short is not None:
        kwargs["enable_short"] = args.enable_short
    if args.atr_period is not None:
        kwargs["atr_period"] = args.atr_period
    if args.atr_stop_multiplier is not None:
        kwargs["atr_stop_multiplier"] = args.atr_stop_multiplier
    if args.atr_take_profit_multiplier is not None:
        kwargs["atr_take_profit_multiplier"] = args.atr_take_profit_multiplier
    return Settings(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="超级趋势回测运行脚本。")
    parser.add_argument("--symbol", default="SOXL.US", help="目标标的代码。")
    parser.add_argument("--interval", default="15m", help="K 线周期。")
    parser.add_argument("--start", help="ISO8601 起始时间（本地时区）。")
    parser.add_argument("--end", help="ISO8601 结束时间（本地时区）。")
    parser.add_argument("--period", type=int, default=10, help="超级趋势 ATR 回看周期。")
    parser.add_argument("--multiplier", type=float, default=3.0, help="超级趋势乘数。")
    parser.add_argument(
        "--initial-capital", type=float, default=10_000.0, help="回测初始资金。"
    )
    parser.add_argument("--commission", type=float, default=0.0, help="单笔佣金。")
    parser.add_argument("--slippage", type=float, default=0.0, help="每股滑点。")
    parser.add_argument(
        "--atr-period",
        type=int,
        help="ATR 计算周期（默认与超级趋势周期一致）。",
    )
    parser.add_argument(
        "--atr-stop-multiplier",
        type=float,
        help="ATR 止损乘数（0 表示关闭）。",
    )
    parser.add_argument(
        "--atr-take-profit-multiplier",
        type=float,
        help="ATR 止盈乘数（0 表示关闭）。",
    )
    parser.add_argument(
        "--enable-short",
        dest="enable_short",
        action="store_true",
        help="开启超级趋势做空逻辑。",
    )
    parser.add_argument(
        "--disable-short",
        dest="enable_short",
        action="store_false",
        help="关闭超级趋势做空逻辑。",
    )
    parser.set_defaults(enable_short=None)
    args = parser.parse_args()

    settings = build_settings(args)
    result = run_backtest_pipeline(settings)

    stats = result.stats
    print("回测完成。")
    print(f"成交笔数：{stats.trade_count}")
    print(f"总收益率：{stats.total_return:.2%}")
    print(f"年化收益率：{stats.annualized_return:.2%}")
    print(f"最大回撤：{stats.max_drawdown:.2%}")
    print(f"胜率：{stats.win_rate:.2%}")
    print(f"盈利因子：{stats.profit_factor:.2f}")
    print(f"净值曲线输出目录：{settings.artifacts_dir}")
    print(f"交易记录 CSV：{settings.trades_csv}")
    print(f"图表路径：{settings.chart_path}")


if __name__ == "__main__":
    main()
