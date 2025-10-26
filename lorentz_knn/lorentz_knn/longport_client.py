"""LongPort OpenAPI helpers reused by the Lorentzian KNN pipeline."""

from __future__ import annotations

import contextlib
from datetime import datetime, timedelta
from typing import Iterator, Optional

from .config import Settings

try:
    from longport.openapi import AdjustType, Config, Period, QuoteContext
except ImportError as exc:  # pragma: no cover - runtime guard
    raise RuntimeError(
        "未检测到 longport，请先执行 `pip install -e .` 安装依赖后再运行流程。"
    ) from exc


PERIOD_MAP = {
    "1m": Period.Min_1,
    "5m": Period.Min_5,
    "15m": Period.Min_15,
    "30m": Period.Min_30,
    "60m": Period.Min_60,
    "1d": Period.Day,
}


def build_config(settings: Settings) -> Config:
    """Create SDK config using credentials from settings."""
    if not (
        settings.longport_app_key
        and settings.longport_app_secret
        and settings.longport_access_token
    ):
        raise RuntimeError("缺少 LongPort 凭证，无法通过 API 拉取数据。")
    return Config(
        settings.longport_app_key,
        settings.longport_app_secret,
        settings.longport_access_token,
    )


@contextlib.contextmanager
def quote_context(settings: Settings) -> Iterator[QuoteContext]:
    """Yield a quote context and ensure cleanup."""
    cfg = build_config(settings)
    ctx = QuoteContext(cfg)
    try:
        yield ctx
    finally:
        if hasattr(ctx, "close"):
            ctx.close()


def fetch_candles(
    ctx: QuoteContext,
    symbol: str,
    interval: str,
    start: Optional[datetime],
    end: Optional[datetime],
) -> list:
    """Fetch historical candles using LongPort SDK."""
    if interval not in PERIOD_MAP:
        raise ValueError(f"不支持的周期 {interval}")

    period = PERIOD_MAP[interval]
    start_date = start.date() if start else None
    end_date = end.date() if end else None

    if not start_date or not end_date:
        return ctx.history_candlesticks_by_date(
            symbol,
            period,
            AdjustType.NoAdjust,
            start=start_date,
            end=end_date,
        )

    bars_per_day = {
        "1m": 390,
        "5m": 78,
        "15m": 26,
        "30m": 13,
        "60m": 7,
        "1d": 1,
    }
    bpd = bars_per_day.get(interval, 26)
    max_bars = 800
    days_per_chunk = max(1, max_bars // bpd)

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
