"""UT Bot 策略的实盘执行脚本."""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List

import pandas as pd
from dotenv import load_dotenv

from .config import Settings
from .data import candles_to_dataframe
from .longport_client import fetch_candles, quote_context
from .utbot_strategy import UTBotStrategy
from .ai.stock_deepseek_indicators_plus import LongbridgeTradeExecutor

load_dotenv()


def env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


def env_float(key: str, default: float) -> float:
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def env_int(key: str, default: int) -> int:
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def timeframe_to_seconds(interval: str) -> int:
    mapping = {
        "1m": 60,
        "5m": 5 * 60,
        "15m": 15 * 60,
        "30m": 30 * 60,
        "60m": 60 * 60,
        "1d": int(6.5 * 60 * 60),
    }
    return mapping.get(interval.lower(), 60)


def fetch_price_dataframe(
    settings: Settings,
    symbol: str,
    interval: str,
    data_points: int,
    max_retry: int = 5,
) -> pd.DataFrame:
    lookback_bars = data_points + 5
    seconds_per_bar = timeframe_to_seconds(interval)
    lookback_seconds = lookback_bars * seconds_per_bar
    with quote_context(settings) as ctx:
        last_error: Exception | None = None
        for attempt in range(max_retry):
            end = datetime.now(UTC) - timedelta(seconds=attempt * seconds_per_bar)
            start = end - timedelta(seconds=lookback_seconds)
            try:
                candles = fetch_candles(ctx, symbol, interval, start, end)
            except Exception as exc:  # pragma: no cover - 网络异常
                last_error = exc
                time.sleep(1)
                continue
            if not candles:
                last_error = RuntimeError("LongPort 未返回任何 K 线")
                time.sleep(1)
                continue
            df = candles_to_dataframe(candles)
            if df.empty:
                last_error = RuntimeError("转换后的行情数据为空")
                time.sleep(1)
                continue
            return df.tail(data_points).copy()
    raise RuntimeError(f"无法获取 {symbol} {interval} 行情：{last_error}")  # pragma: no cover


def determine_target_size(
    price: float,
    cash: float | None,
    risk_pct: float,
    lot_size: int,
) -> int:
    if price <= 0:
        return 0
    capital = cash if cash is not None else 0.0
    budget = capital * risk_pct
    if budget <= 0:
        return lot_size
    qty = max(int(budget // price), lot_size)
    if lot_size > 1:
        qty = (qty // lot_size) * lot_size
    return max(qty, lot_size)


def main() -> None:
    settings = Settings()
    symbol = os.getenv("UTBOT_LIVE_SYMBOL", settings.symbol)
    interval = os.getenv("UTBOT_LIVE_INTERVAL", settings.interval or "15m")
    data_points = env_int("UTBOT_LIVE_DATA_POINTS", 300)
    risk_pct = env_float("UTBOT_LIVE_RISK_PCT", 0.02)
    allow_short = env_bool("UTBOT_LIVE_ENABLE_SHORT", False)

    strategy_params: Dict[str, Any] = {
        "stc_length": env_int("UTBOT_LIVE_STC_LENGTH", 80),
        "stc_fast": env_int("UTBOT_LIVE_STC_FAST", 26),
        "stc_slow": env_int("UTBOT_LIVE_STC_SLOW", 50),
        "stc_smooth": env_float("UTBOT_LIVE_STC_SMOOTH", 0.5),
        "stc_bull_threshold": env_float("UTBOT_LIVE_STC_BULL", 30.0),
        "stc_bear_threshold": env_float("UTBOT_LIVE_STC_BEAR", 70.0),
        "use_stc_filter": env_bool("UTBOT_LIVE_USE_STC_FILTER", True),
        "use_qqe_filter": env_bool("UTBOT_LIVE_USE_QQE_FILTER", True),
        "filter_require_both": env_bool("UTBOT_LIVE_FILTER_REQUIRE_BOTH", False),
        "enable_short": allow_short,
        "atr_period": env_int("UTBOT_LIVE_ATR_PERIOD", 1),
        "atr_multiplier": env_float("UTBOT_LIVE_ATR_MULTIPLIER", 3.0),
        "use_trend_ma_filter": env_bool("UTBOT_LIVE_USE_TREND_MA_FILTER", False),
        "trend_ma_length": env_int("UTBOT_LIVE_TREND_MA_LENGTH", 34),
        "use_adx_filter": env_bool("UTBOT_LIVE_USE_ADX_FILTER", False),
        "adx_length": env_int("UTBOT_LIVE_ADX_LENGTH", 34),
        "adx_threshold": env_float("UTBOT_LIVE_ADX_THRESHOLD", 20.0),
        "boll_length": env_int("UTBOT_LIVE_BOLL_LENGTH", 50),
        "boll_multiplier": env_float("UTBOT_LIVE_BOLL_MULTIPLIER", 0.35),
        "qqe_threshold": env_float("UTBOT_LIVE_QQE_THRESHOLD", 3.0),
    }
    strategy = UTBotStrategy(settings=settings, **strategy_params)

    trade_executor = LongbridgeTradeExecutor(
        settings,
        symbol=symbol,
        allow_short=allow_short,
        test_mode=env_bool("UTBOT_LIVE_PAPER", True),
    )

    seconds_per_bar = timeframe_to_seconds(interval)

    while True:
        try:
            price_df = fetch_price_dataframe(settings, symbol, interval, data_points)
            indicator_df = strategy.prepare_data(price_df)
            latest = indicator_df.iloc[-1]
        except Exception as exc:
            print(f"⚠️ 获取行情或计算信号失败：{exc}")
            time.sleep(seconds_per_bar // 2)
            continue

        price = float(latest["close"])
        long_entry = bool(latest.get("long_entry"))
        long_exit = bool(latest.get("long_exit"))
        short_entry = allow_short and bool(latest.get("short_entry"))
        short_exit = allow_short and bool(latest.get("short_exit"))

        cash = trade_executor.fetch_available_cash()
        pos_snapshot = trade_executor.fetch_position_snapshot()
        lot_size = env_int("UTBOT_LIVE_LOT_SIZE", 1)

        target_side: str | None = None
        target_size = 0
        reason = ""

        if long_exit:
            target_side = None
            reason = "long exit"
        elif long_entry:
            target_side = "long"
            target_size = determine_target_size(price, cash, risk_pct, lot_size)
            reason = "long entry"
        elif short_exit:
            target_side = None
            reason = "short exit"
        elif short_entry:
            target_side = "short"
            target_size = determine_target_size(price, cash, risk_pct, lot_size)
            reason = "short entry"
        else:
            reason = "hold"

        print(
            f"[{datetime.now().astimezone()}] close={price:.2f} "
            f"signal={reason} long_entry={long_entry} long_exit={long_exit}"
        )

        trade_executor.sync_position(
            signal=reason,
            target_side=target_side,
            target_size=target_size,
            price=price,
            current_position=pos_snapshot,
            reason=reason,
        )

        time.sleep(seconds_per_bar)


if __name__ == "__main__":
    main()
