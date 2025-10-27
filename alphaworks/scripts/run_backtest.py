#!/usr/bin/env python3
"""基于 LongPort OpenAPI 运行回测。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from typing import Any, Dict

from alphaworks.config import Settings
from alphaworks.pipeline import run_backtest_pipeline


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def _coerce_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if raw.startswith("0") and raw != "0":
            raise ValueError
        return int(raw)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def parse_strategy_params(args: argparse.Namespace) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if args.strategy_params_json:
        parsed = json.loads(args.strategy_params_json)
        if not isinstance(parsed, dict):
            raise ValueError("--strategy-params-json 必须是 JSON 对象字符串。")
        params.update(parsed)
    if args.strategy_param:
        for item in args.strategy_param:
            if "=" not in item:
                raise ValueError(f"策略参数格式错误：{item!r}，应为 key=value。")
            key, value = item.split("=", 1)
            params[key] = _coerce_value(value)
    return params


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
    if args.enable_long is not None:
        kwargs["enable_long"] = args.enable_long
    if args.enable_short is not None:
        kwargs["enable_short"] = args.enable_short
    if args.atr_period is not None:
        kwargs["atr_period"] = args.atr_period
    if args.atr_stop_multiplier is not None:
        kwargs["atr_stop_multiplier"] = args.atr_stop_multiplier
    if args.atr_take_profit_multiplier is not None:
        kwargs["atr_take_profit_multiplier"] = args.atr_take_profit_multiplier
    if args.risk_per_trade_pct is not None:
        kwargs["risk_per_trade_pct"] = args.risk_per_trade_pct
    if args.risk_atr_multiplier is not None:
        kwargs["risk_atr_multiplier"] = args.risk_atr_multiplier
    if args.cooldown_bars is not None:
        kwargs["cooldown_bars"] = args.cooldown_bars
    if args.trend_filter_period is not None:
        kwargs["trend_filter_period"] = args.trend_filter_period
    if args.volatility_filter_atr_pct is not None:
        kwargs["volatility_filter_atr_pct"] = args.volatility_filter_atr_pct
    if args.strategy:
        kwargs["strategy"] = args.strategy
    strategy_params = parse_strategy_params(args)
    if strategy_params:
        kwargs["strategy_params"] = strategy_params
    return Settings(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="回测运行脚本。")
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
        help="开启做空逻辑。",
    )
    parser.add_argument(
        "--disable-short",
        dest="enable_short",
        action="store_false",
        help="关闭做空逻辑。",
    )
    parser.add_argument(
        "--enable-long",
        dest="enable_long",
        action="store_true",
        help="开启做多逻辑。",
    )
    parser.add_argument(
        "--disable-long",
        dest="enable_long",
        action="store_false",
        help="关闭做多逻辑。",
    )
    parser.add_argument(
        "--risk-per-trade-pct",
        type=float,
        help="每笔交易风险占权益百分比（0 表示按全仓处理）。",
    )
    parser.add_argument(
        "--risk-atr-multiplier",
        type=float,
        help="风险单位计算所用 ATR 乘数（默认使用策略或 2.0）。",
    )
    parser.add_argument(
        "--cooldown-bars",
        type=int,
        help="离场后需等待的冷却 K 线数量。",
    )
    parser.add_argument(
        "--trend-filter-period",
        type=int,
        help="趋势过滤均线周期（0 表示关闭）。",
    )
    parser.add_argument(
        "--volatility-filter-atr-pct",
        type=float,
        help="ATR/价格 波动率过滤阈值（0 表示关闭）。",
    )
    parser.add_argument(
        "--strategy",
        help="策略标识或类路径，默认 supertrend。",
    )
    parser.add_argument(
        "--strategy-param",
        action="append",
        dest="strategy_param",
        help="以 key=value 形式追加策略参数，可多次使用。",
    )
    parser.add_argument(
        "--strategy-params-json",
        help="以 JSON 对象传入的策略参数。",
    )
    parser.set_defaults(enable_short=None, enable_long=None)
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
    print(f"图表路径：{settings.chart_path_for(settings.strategy)}")


if __name__ == "__main__":
    main()
