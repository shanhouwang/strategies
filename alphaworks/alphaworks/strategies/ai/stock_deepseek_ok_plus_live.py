"""实盘版 DeepSeek OK+ 策略，复用回测提示词并对接 Longbridge。"""

from __future__ import annotations

import atexit
import json
import os
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List

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
            end_time = datetime.utcnow() - timedelta(days=attempt)
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
    position_text = (
        f"{portfolio_state['side']} 仓位 {portfolio_state['size']} 股"
        if portfolio_state
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


def calculate_intelligent_position(signal_data: Dict[str, Any], price: float) -> int:
    config = TRADE_CONFIG["position_management"]
    if not config.get("enable_intelligent_position", True):
        return max(config.get("min_shares", 1), 1)
    confidence = (signal_data.get("confidence") or "MEDIUM").upper()
    multiplier = {
        "HIGH": config["high_confidence_multiplier"],
        "MEDIUM": config["medium_confidence_multiplier"],
        "LOW": config["low_confidence_multiplier"],
    }.get(confidence, config["medium_confidence_multiplier"])
    suggested_cash = config["base_cash_amount"] * multiplier
    capital = config.get("account_capital", settings.initial_capital)
    max_cash = max(capital * config["max_position_ratio"], config["base_cash_amount"])
    final_cash = min(suggested_cash, max_cash)
    price = max(price, 1e-6)
    min_shares = max(config.get("min_shares", 1), 1)
    shares = max(int(final_cash // price), min_shares)
    return shares


def execute_trade(signal_data: Dict[str, Any], price_data: Dict[str, Any], df_slice: pd.DataFrame) -> None:
    global portfolio_state

    signal = signal_data.get("signal", "HOLD").upper()
    price = price_data["price"]
    position_size = calculate_intelligent_position(signal_data, price)
    current_position = {
        "side": portfolio_state.get("side"),
        "size": portfolio_state.get("size", 0),
        "entry_price": portfolio_state.get("entry_price"),
    } if portfolio_state else None

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

    if TRADE_CONFIG["test_mode"]:
        print("（测试模式）仅记录信号，未真实下单。")


def trading_cycle() -> None:
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


def main() -> None:
    print("DeepSeek OK+ 实盘脚本启动。")
    print(
        f"标的: {TRADE_CONFIG['symbol']} | 周期: {TRADE_CONFIG['timeframe']} | 测试模式: {TRADE_CONFIG['test_mode']}"
    )
    while True:
        try:
            trading_cycle()
        except Exception as exc:
            print(f"❌ 本轮执行异常: {exc}")
        wait_seconds = timeframe_to_minutes(TRADE_CONFIG["timeframe"]) * 60
        time.sleep(wait_seconds)


if __name__ == "__main__":
    main()
