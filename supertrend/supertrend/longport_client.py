"""LongPort OpenAPI 工具函数。"""

from __future__ import annotations

import contextlib
from datetime import datetime, timedelta
from typing import Iterator, Optional

from .config import Settings

try:
    from longport.openapi import AdjustType, Config, Period, QuoteContext, TradeContext
except ImportError as exc:  # pragma: no cover - 运行时导入异常
    raise RuntimeError("未检测到 longport，请先执行 `pip install -e .` 安装依赖后再运行流程。") from exc


PERIOD_MAP = {
    "1m": Period.Min_1,
    "5m": Period.Min_5,
    "15m": Period.Min_15,
    "30m": Period.Min_30,
    "60m": Period.Min_60,
    "1d": Period.Day,
}


def build_config(settings: Settings) -> Config:
    """根据配置创建 LongPort Config，遵循官方 SDK 标准调用方式。
    
    根据官方文档，SDK会自动处理连接配置，不需要手动设置URL。
    """
    return Config(
        settings.longport_app_key,
        settings.longport_app_secret,
        settings.longport_access_token,
    )


@contextlib.contextmanager
def quote_context(settings: Settings) -> Iterator[QuoteContext]:
    """生成官方 SDK 推荐的行情上下文。"""
    cfg = build_config(settings)
    ctx = QuoteContext(cfg)
    try:
        yield ctx
    finally:
        # QuoteContext可能需要手动关闭资源
        if hasattr(ctx, 'close'):
            ctx.close()


@contextlib.contextmanager
def trade_context(settings: Settings) -> Iterator[TradeContext]:
    """生成官方 SDK 推荐的交易上下文。"""
    cfg = build_config(settings)
    ctx = TradeContext(cfg)
    try:
        yield ctx
    finally:
        # TradeContext可能需要手动关闭资源
        if hasattr(ctx, 'close'):
            ctx.close()


def fetch_candles(
    ctx: QuoteContext,
    symbol: str,
    interval: str,
    start: Optional[datetime],
    end: Optional[datetime],
) -> list:
    """使用官方 SDK 的 history_candlesticks 查询历史数据。"""
    if interval not in PERIOD_MAP:
        raise ValueError(f"不支持的周期 {interval}")

    period = PERIOD_MAP[interval]
    start_date = start.date() if start else None
    end_date = end.date() if end else None

    # 如果无起止日期，直接请求一次
    if not start_date or not end_date:
        return ctx.history_candlesticks_by_date(
            symbol,
            period,
            AdjustType.NoAdjust,
            start=start_date,
            end=end_date,
        )

    # 计算每次分片的天数，避免超过单次最大条数限制
    bars_per_day = {
        "1m": 390,
        "5m": 78,
        "15m": 26,
        "30m": 13,
        "60m": 7,
        "1d": 1,
    }
    bpd = bars_per_day.get(interval, 26)
    max_bars_per_request = 800  # 留余量，避免触顶
    days_per_chunk = max(1, max_bars_per_request // bpd)

    all_candles = []
    cur = start_date
    while cur <= end_date:
        chunk_end = min(cur + timedelta(days=days_per_chunk - 1), end_date)
        part = ctx.history_candlesticks_by_date(
            symbol,
            period,
            AdjustType.NoAdjust,
            start=cur,
            end=chunk_end,
        )
        if part:
            all_candles.extend(part)
        cur = chunk_end + timedelta(days=1)

    return all_candles
