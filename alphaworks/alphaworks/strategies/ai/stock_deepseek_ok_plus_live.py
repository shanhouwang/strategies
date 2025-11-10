"""实盘版 DeepSeek OK+ 策略，复用回测提示词并对接 Longbridge。"""

from __future__ import annotations

import atexit
import json
import os
import re
import time
from datetime import UTC, datetime, timedelta, time as dt_time
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

from ...config import Settings
from ...data import candles_to_dataframe
from ...longport_client import fetch_candles, quote_context
from .stock_deepseek_indicators_plus import (
    LongbridgeTradeExecutor,
    StockDeepseekOkPlusStrategy,
)

load_dotenv()

settings = Settings()
strategy = StockDeepseekOkPlusStrategy(settings=settings)


def _env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _env_float(key: str, default: float) -> float:
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_str(key: str, default: str) -> str:
    value = os.getenv(key)
    if value is None or not value.strip():
        return default
    return value.strip()


TRADE_CONFIG = {
    "symbol": os.getenv("STOCK_SYMBOL", settings.symbol),
    "timeframe": os.getenv("STOCK_INTERVAL", settings.interval),
    "data_points": int(os.getenv("DATA_POINTS", "96")),
    "allow_short": _env_bool("ALLOW_SHORT", False),
    "test_mode": _env_bool("TEST_MODE", True),
    "position_management": {
        "enable_intelligent_position": _env_bool("ENABLE_INTELLIGENT_POSITION", True),
        "base_cash_amount": _env_float("BASE_CASH_AMOUNT", 1_000.0),
        "max_position_ratio": _env_float("MAX_POSITION_RATIO", 0.25),
        "high_confidence_multiplier": _env_float("HIGH_CONFIDENCE_MULTIPLIER", 1.5),
        "medium_confidence_multiplier": _env_float("MEDIUM_CONFIDENCE_MULTIPLIER", 1.0),
        "low_confidence_multiplier": _env_float("LOW_CONFIDENCE_MULTIPLIER", 0.5),
        "trend_strength_multiplier": _env_float("TREND_STRENGTH_MULTIPLIER", 1.2),
        "account_capital": _env_float("ACCOUNT_CAPITAL", settings.initial_capital),
        "min_shares": int(os.getenv("MIN_SHARES", "1")),
    },
    "market_schedule": {
        "timezone": _env_str("MARKET_TIMEZONE", "America/New_York"),
        "regular_open": _env_str("MARKET_OPEN", "09:30"),
        "regular_close": _env_str("MARKET_CLOSE", "16:00"),
        "allow_pre_market": _env_bool("ALLOW_PRE_MARKET", False),
        "allow_after_hours": _env_bool("ALLOW_AFTER_HOURS", False),
    },
}

TIMEFRAME_TO_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "60m": 60,
    "1d": 60 * 6.5,
}

deepseek_key = os.getenv("DEEPSEEK_API_KEY")
if deepseek_key:
    from openai import OpenAI  # type: ignore

    deepseek_client = OpenAI(api_key=deepseek_key, base_url="https://api.deepseek.com")
else:
    deepseek_client = None

price_history: List[Dict[str, Any]] = []
signal_history: List[Dict[str, Any]] = []
portfolio_state: Dict[str, Any] | None = None


def _load_market_timezone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except Exception:
        fallback = "America/New_York"
        print(f"⚠️ 无法解析时区 {name}，默认 {fallback}")
        return ZoneInfo(fallback)


def _parse_time(value: str, default: dt_time) -> dt_time:
    try:
        hour, minute = value.split(":")
        return dt_time(hour=int(hour), minute=int(minute))
    except Exception:
        return default


MARKET_SCHEDULE = TRADE_CONFIG["market_schedule"]
MARKET_TIMEZONE = _load_market_timezone(MARKET_SCHEDULE["timezone"])
MARKET_OPEN_TIME = _parse_time(MARKET_SCHEDULE["regular_open"], dt_time(hour=9, minute=30))
MARKET_CLOSE_TIME = _parse_time(MARKET_SCHEDULE["regular_close"], dt_time(hour=16, minute=0))


def _describe_wait(seconds: int) -> str:
    seconds = max(seconds, 0)
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    parts = []
    if hours:
        parts.append(f"{hours} 小时")
    if minutes:
        parts.append(f"{minutes} 分")
    if sec and not hours:
        parts.append(f"{sec} 秒")
    return "".join(parts) or "片刻"


def is_regular_trading_session(now: datetime | None = None) -> tuple[bool, str, int | None]:
    tz = MARKET_TIMEZONE
    localized_now = (now or datetime.now(UTC)).astimezone(tz)
    weekday = localized_now.weekday()
    if weekday >= 5:
        return False, "当前为周末休市", None
    open_dt = localized_now.replace(
        hour=MARKET_OPEN_TIME.hour,
        minute=MARKET_OPEN_TIME.minute,
        second=0,
        microsecond=0,
    )
    close_dt = localized_now.replace(
        hour=MARKET_CLOSE_TIME.hour,
        minute=MARKET_CLOSE_TIME.minute,
        second=0,
        microsecond=0,
    )
    allow_pre = MARKET_SCHEDULE["allow_pre_market"]
    allow_after = MARKET_SCHEDULE["allow_after_hours"]
    tz_label = getattr(tz, "key", str(tz))
    if localized_now < open_dt:
        if allow_pre:
            return True, "", 0
        wait_seconds = int((open_dt - localized_now).total_seconds())
        return (
            False,
            f"尚未开盘（{MARKET_OPEN_TIME.strftime('%H:%M')} {tz_label}，约还有 {_describe_wait(wait_seconds)}）",
            wait_seconds,
        )
    if localized_now >= close_dt:
        if allow_after:
            return True, "", 0
        return False, f"已收盘（{MARKET_CLOSE_TIME.strftime('%H:%M')} {tz_label}），等待下一交易日", None
    return True, "", 0

try:
    trade_executor = LongbridgeTradeExecutor(
        settings,
        symbol=TRADE_CONFIG["symbol"],
        allow_short=TRADE_CONFIG["allow_short"],
        test_mode=TRADE_CONFIG["test_mode"],
    )
except Exception as exc:  # pragma: no cover - 凭证缺失
    print(f"⚠️ Longbridge 执行器初始化失败: {exc}")
    trade_executor = None
else:
    atexit.register(trade_executor.close)


def timeframe_to_minutes(timeframe: str) -> int:
    return int(TIMEFRAME_TO_MINUTES.get(timeframe, 15))


def fetch_price_dataframe(symbol: str, interval: str, points: int) -> pd.DataFrame:
    minutes_per_bar = timeframe_to_minutes(interval)
    lookback_minutes = minutes_per_bar * (points + 8)
    max_attempts = 5
    last_error: Exception | None = None

    with quote_context(settings) as ctx:
        for attempt in range(max_attempts):
            end_time = datetime.now(UTC) - timedelta(days=attempt)
            start_time = end_time - timedelta(minutes=lookback_minutes)
            try:
                candles = fetch_candles(ctx, symbol, interval, start_time, end_time)
            except Exception as exc:
                last_error = exc
                print(
                    f"⚠️ 第 {attempt + 1} 次获取 {symbol} {interval} K线失败：{exc}，尝试回退旧日期。"
                )
                continue
            if candles:
                df = candles_to_dataframe(candles)
                if not df.empty:
                    return df.tail(points).copy()
                last_error = RuntimeError("转换后的行情数据为空")
            else:
                last_error = RuntimeError("Longbridge 未返回任何 K 线")
            print(
                f"⚠️ 未获取到 {symbol} 在 {end_time.date()} 的 {interval} K线，"
                f"回退重试 ({attempt + 1}/{max_attempts})。"
            )

    if last_error:
        raise RuntimeError(f"多次重试后仍无法获取 {symbol} 的 {interval} K线数据。") from last_error
    raise RuntimeError(f"未能获取 {symbol} 的 {interval} K线数据，已尝试回退 {max_attempts} 天。")


def get_live_position_snapshot() -> Dict[str, Any] | None:
    global portfolio_state
    if trade_executor:
        snapshot = trade_executor.fetch_position_snapshot()
        if snapshot is not None:
            portfolio_state = snapshot
            return snapshot
        portfolio_state = None
    return portfolio_state


def get_available_cash() -> float | None:
    if trade_executor:
        return trade_executor.fetch_available_cash()
    return None


def estimate_max_qty(price: float, side: str) -> Dict[str, float | int] | None:
    if trade_executor:
        return trade_executor.estimate_max_quantity(price=price, side=side)
    return None


def _kline_text(df_slice: pd.DataFrame) -> str:
    records = strategy._kline_records(df_slice)  # type: ignore[attr-defined]
    lines = []
    for idx, rec in enumerate(records, start=1):
        change = (
            (rec["close"] - rec["open"]) / rec["open"] * 100 if rec["open"] else 0
        )
        lines.append(
            f"K{idx}: {'阳线' if rec['close'] >= rec['open'] else '阴线'} "
            f"开:{rec['open']:.2f} 收:{rec['close']:.2f} 涨跌:{change:+.2f}%"
        )
    return "【最近K线】\n" + "\n".join(lines)


def safe_json_parse(content: str) -> Dict[str, Any] | None:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.S)
        if not match:
            return None
        text = match.group()
        text = re.sub(r"(\w+):", r'"\1":', text)
        text = re.sub(r",\s*}", "}", text)
        text = re.sub(r",\s*]", "]", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None


def build_prompt(
    df_slice: pd.DataFrame,
    row: pd.Series,
    trend: Dict[str, Any],
    levels: Dict[str, Any],
    position_text: str,
    last_signal_desc: str,
) -> str:
    kline_text = _kline_text(df_slice)
    technical_text = strategy._technical_text(  # type: ignore[attr-defined]
        row["symbol"], row, trend, levels
    )
    return strategy._build_prompt(  # type: ignore[attr-defined]
        row["symbol"],
        TRADE_CONFIG["timeframe"],
        {
            "symbol": row["symbol"],
            "price": float(row["close"]),
            "timestamp": df_slice.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "volume": float(row["volume"]),
            "timeframe": TRADE_CONFIG["timeframe"],
            "price_change": float(
                (row["close"] - df_slice["close"].iloc[-2]) / df_slice["close"].iloc[-2] * 100
            )
            if len(df_slice) > 1 and df_slice["close"].iloc[-2]
            else 0.0,
        },
        trend,
        levels,
        "",
        kline_text,
        technical_text,
        position_text,
        last_signal_desc,
    )


def analyze_with_deepseek(df_slice: pd.DataFrame) -> Dict[str, Any]:
    if deepseek_client is None:
        raise RuntimeError("未设置 DEEPSEEK_API_KEY，无法调用模型。")

    row = df_slice.iloc[-1]
    levels = strategy._support_levels(df_slice)  # type: ignore[attr-defined]
    trend = strategy._trend_from_row(row)  # type: ignore[attr-defined]
    last_signal = signal_history[-1] if signal_history else None
    snapshot = get_live_position_snapshot()
    position_text = (
        f"{snapshot['side']} 仓位 {snapshot['size']} 股"
        if snapshot
        else "当前无持仓"
    )
    last_signal_desc = (
        f"{last_signal.get('signal')} / {last_signal.get('confidence')} / {last_signal.get('reason', '')}"
        if last_signal
        else "暂无历史信号"
    )
    prompt = build_prompt(df_slice, row, trend, levels, position_text, last_signal_desc)
    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": "你是一位专注于指标形态的美股交易分析师，输出需结构化且果断。",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.35,
    )
    content = response.choices[0].message.content  # type: ignore[index]
    parsed = safe_json_parse(content or "")
    if not parsed:
        print("⚠️ 模型输出无法解析，使用保守 HOLD。")
        return {
            "signal": "HOLD",
            "reason": "AI 输出解析失败。",
            "stop_loss": float(row["close"]) * 0.97,
            "take_profit": float(row["close"]) * 1.03,
            "confidence": "LOW",
            "is_fallback": True,
        }
    signal_history.append(parsed)
    return parsed


def calculate_intelligent_position(
    signal_data: Dict[str, Any],
    price: float,
    available_cash: float | None,
    max_qty_info: Dict[str, float | int] | None,
) -> int:
    config = TRADE_CONFIG["position_management"]
    if not config.get("enable_intelligent_position", True):
        return max(config.get("min_shares", 1), 1)
    confidence = (signal_data.get("confidence") or "MEDIUM").upper()
    multiplier = {
        "HIGH": config["high_confidence_multiplier"],
        "MEDIUM": config["medium_confidence_multiplier"],
        "LOW": config["low_confidence_multiplier"],
    }.get(confidence, config["medium_confidence_multiplier"])
    capital_source = (
        available_cash
        if available_cash is not None
        else config.get("account_capital", settings.initial_capital)
    )
    capital = max(float(capital_source), 0.0)
    max_cash = capital * config["max_position_ratio"]
    suggested_cash = max_cash * multiplier
    if available_cash is not None:
        suggested_cash = min(suggested_cash, available_cash)
    price = max(price, 1e-6)
    min_shares = max(config.get("min_shares", 1), 1)
    shares = max(int(suggested_cash // price), min_shares)

    # 如果券商返回了最大下单股数，则必须严格限制，避免出现“最多买 0 股却下单成功”的情况。
    if max_qty_info:
        limit_candidates: List[int] = []
        for key in ("cash", "margin"):
            raw_value = max_qty_info.get(key)
            if raw_value is None:
                continue
            try:
                limit_candidates.append(max(int(raw_value), 0))
            except (TypeError, ValueError):
                continue
        if limit_candidates:
            max_allowed_shares = max(limit_candidates)
            shares = min(shares, max_allowed_shares)
    return max(shares, 0)


def execute_trade(signal_data: Dict[str, Any], price_data: Dict[str, Any], df_slice: pd.DataFrame) -> None:
    global portfolio_state

    signal = signal_data.get("signal", "HOLD").upper()
    price = price_data["price"]
    available_cash = get_available_cash()
    max_qty_info: Dict[str, float | int] | None = None
    if signal == "BUY":
        max_qty_info = estimate_max_qty(price, "BUY")
    if available_cash is not None:
        print(f"💰 当前可用资金: ${available_cash:,.2f}")
    else:
        print("💰 当前可用资金: 未获取（测试模式或 API 失败）")
    position_size = calculate_intelligent_position(
        signal_data,
        price,
        available_cash,
        max_qty_info,
    )
    current_position = get_live_position_snapshot()

    def _set_position(side: str | None, size: int = 0) -> None:
        nonlocal price
        global portfolio_state
        if side is None or size <= 0:
            portfolio_state = None
        else:
            portfolio_state = {"side": side, "size": size, "entry_price": price}

    target_side: str | None = None
    target_size = 0

    if signal == "BUY":
        target_side = "long"
        target_size = position_size
        _set_position("long", position_size)
    elif signal == "SELL":
        target_side = "short" if TRADE_CONFIG["allow_short"] else None
        target_size = position_size if TRADE_CONFIG["allow_short"] else 0
        if TRADE_CONFIG["allow_short"]:
            _set_position("short", position_size)
        else:
            _set_position(None)
    else:
        target_side = None
        target_size = 0
        _set_position(None)

    if trade_executor:
        if max_qty_info:
            buying_power = max_qty_info.get("buying_power")
            if buying_power:
                print(f"🧾 Buying Power: ${float(buying_power):,.2f}")
            print(
                f"📊 最大下单量（现金/融资）: "
                f"{int(max_qty_info.get('cash', 0) or 0)} / "
                f"{int(max_qty_info.get('margin', 0) or 0)} 股"
            )
        try:
            trade_executor.sync_position(
                signal=signal,
                target_side=target_side,
                target_size=target_size,
                price=price,
                current_position=current_position,
                reason=signal_data.get("reason", ""),
            )
        except Exception as exc:
            print(f"❌ Longbridge 执行异常: {exc}")
        else:
            get_live_position_snapshot()

    if TRADE_CONFIG["test_mode"]:
        print("（测试模式）仅记录信号，未真实下单。")


def trading_cycle() -> int:
    is_open, reason, wait_hint = is_regular_trading_session()
    default_wait = timeframe_to_minutes(TRADE_CONFIG["timeframe"]) * 60
    if not is_open:
        current_us_time = datetime.now(UTC).astimezone(MARKET_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S %Z")
        print(f"当前非交易时段（{current_us_time}）：{reason}")
        if wait_hint and wait_hint < default_wait:
            return max(wait_hint, 30)
        return default_wait
    df = fetch_price_dataframe(
        TRADE_CONFIG["symbol"],
        TRADE_CONFIG["timeframe"],
        TRADE_CONFIG["data_points"],
    )
    df["symbol"] = TRADE_CONFIG["symbol"]
    df = strategy._calculate_indicators(df)  # type: ignore[attr-defined]
    signal = analyze_with_deepseek(df)
    price_data = {
        "price": float(df.iloc[-1]["close"]),
        "timestamp": df.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
    }
    execute_trade(signal, price_data, df)
    return default_wait


def main() -> None:
    print("DeepSeek OK+ 实盘脚本启动。")
    print(
        f"标的: {TRADE_CONFIG['symbol']} | 周期: {TRADE_CONFIG['timeframe']} | 测试模式: {TRADE_CONFIG['test_mode']}"
    )
    default_wait = timeframe_to_minutes(TRADE_CONFIG["timeframe"]) * 60
    while True:
        try:
            wait_seconds = trading_cycle()
        except Exception as exc:
            print(f"❌ 本轮执行异常: {exc}")
            wait_seconds = default_wait
        time.sleep(wait_seconds)


if __name__ == "__main__":
    main()
