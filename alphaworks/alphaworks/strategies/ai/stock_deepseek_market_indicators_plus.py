"""基于 LongPort 股票数据的 DeepSeek 智能信号脚本（含情绪/指标分析/实盘执行）。"""

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
try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore[assignment]

from ...config import Settings
from ...data import candles_to_dataframe
from ...longport_client import fetch_candles, quote_context
from .stock_deepseek_indicators_plus import LongbridgeTradeExecutor

load_dotenv()

settings = Settings()


def _env_bool(key: str, default: bool) -> bool:
    """读取布尔型环境变量，兼容多种 true/false 写法。"""
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _env_float(key: str, default: float) -> float:
    """读取浮点型环境变量，解析失败则退回默认值。"""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


# === 核心交易配置：优先读取环境变量，可随时覆盖 ===
TRADE_CONFIG = {
    "symbol": os.getenv("STOCK_SYMBOL", settings.symbol),
    "benchmark_symbol": os.getenv("BENCHMARK_SYMBOL", "SPY.US"),
    "timeframe": os.getenv("STOCK_INTERVAL", settings.interval),
    "data_points": int(os.getenv("DATA_POINTS", "96")),
    "test_mode": _env_bool("TEST_MODE", True),
    "allow_short": _env_bool("ALLOW_SHORT", False),
    "analysis_periods": {
        "short_term": 20,
        "medium_term": 50,
        "long_term": 120,
    },
    "position_management": {
        "enable_intelligent_position": _env_bool("ENABLE_INTELLIGENT_POSITION", True),
        "base_cash_amount": _env_float("BASE_CASH_AMOUNT", 1_000.0),
        "high_confidence_multiplier": _env_float("HIGH_CONFIDENCE_MULTIPLIER", 1.5),
        "medium_confidence_multiplier": _env_float("MEDIUM_CONFIDENCE_MULTIPLIER", 1.0),
        "low_confidence_multiplier": _env_float("LOW_CONFIDENCE_MULTIPLIER", 0.5),
        "max_position_ratio": _env_float("MAX_POSITION_RATIO", 0.25),
        "trend_strength_multiplier": _env_float("TREND_STRENGTH_MULTIPLIER", 1.2),
        "account_capital": _env_float("ACCOUNT_CAPITAL", settings.initial_capital),
        "min_shares": int(os.getenv("MIN_SHARES", "1")),
    },
}

# 统一的周期 -> 分钟映射，供等待与历史窗口计算
TIMEFRAME_TO_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "60m": 60,
    "1d": 60 * 6.5,
}

# DeepSeek API 客户端，用于生成自然语言信号
_deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
if OpenAI is None or not _deepseek_api_key:
    deepseek_client = None
else:
    deepseek_client = OpenAI(
        api_key=_deepseek_api_key,
        base_url="https://api.deepseek.com",
    )

# 运行时缓存：保存最近行情、AI 信号与模拟持仓
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
except Exception as exc:  # pragma: no cover - 配置缺失或 SDK 未装
    print(f"⚠️ Longbridge 执行器初始化失败: {exc}")
    trade_executor = None
else:
    atexit.register(trade_executor.close)


def timeframe_to_minutes(timeframe: str) -> int:
    """将 LongPort 周期标签转换为分钟数。"""
    return int(TIMEFRAME_TO_MINUTES.get(timeframe, 15))


def fetch_price_dataframe(symbol: str, interval: str, points: int) -> pd.DataFrame:
    """从 LongPort 拉取指定数量的历史 K 线并转为 DataFrame。"""
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
                if df.empty:
                    last_error = RuntimeError("转换后的行情数据为空")
                    print(
                        f"⚠️ {symbol} {interval} K线数据为空集，尝试回退旧日期 "
                        f"({attempt + 1}/{max_attempts})。"
                    )
                    continue
                return df.tail(points).copy()

            print(
                f"⚠️ 未获取到 {symbol} 在 {end_time.date()} 的 {interval} K线，"
                f"回退重试 ({attempt + 1}/{max_attempts})。"
            )

    if last_error:
        raise RuntimeError(f"多次重试后仍无法获取 {symbol} 的 {interval} K线数据。") from last_error
    raise RuntimeError(f"未能获取 {symbol} 的 {interval} K线数据，已尝试回退 {max_attempts} 天。")


def setup_market_environment() -> bool:
    """快速检查 LongPort 行情接口是否可用。"""
    try:
        df = fetch_price_dataframe(
            TRADE_CONFIG["symbol"],
            TRADE_CONFIG["timeframe"],
            min(TRADE_CONFIG["data_points"], 64),
        )
        print(
            f"✅ 已连接 LongPort，获取到 {TRADE_CONFIG['symbol']} "
            f"{len(df)} 条 {TRADE_CONFIG['timeframe']} K 线"
        )
        return True
    except Exception as exc:
        print(f"❌ 无法初始化行情环境: {exc}")
        return False


def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """在 K 线数据上补充均线、MACD、RSI、布林带等指标。"""
    try:
        df["sma_5"] = df["close"].rolling(window=5, min_periods=1).mean()
        df["sma_20"] = df["close"].rolling(window=20, min_periods=1).mean()
        df["sma_50"] = df["close"].rolling(window=50, min_periods=1).mean()

        df["ema_12"] = df["close"].ewm(span=12).mean()
        df["ema_26"] = df["close"].ewm(span=26).mean()
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        df["bb_middle"] = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
        df["bb_lower"] = df["bb_middle"] - (bb_std * 2)
        df["bb_position"] = (df["close"] - df["bb_lower"]) / (
            df["bb_upper"] - df["bb_lower"]
        )

        df["volume_ma"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"]

        df["resistance"] = df["high"].rolling(20).max()
        df["support"] = df["low"].rolling(20).min()

        return df.bfill().ffill()
    except Exception as exc:
        print(f"技术指标计算失败: {exc}")
        return df


def get_support_resistance_levels(df: pd.DataFrame, lookback: int = 20) -> Dict[str, float]:
    """计算静态/动态支撑阻力，用于生成提示词。"""
    try:
        recent = df.tail(lookback)
        current_price = recent["close"].iloc[-1]
        resistance_level = recent["high"].max()
        support_level = recent["low"].min()

        bb_upper = recent["bb_upper"].iloc[-1]
        bb_lower = recent["bb_lower"].iloc[-1]

        return {
            "static_resistance": resistance_level,
            "static_support": support_level,
            "dynamic_resistance": bb_upper,
            "dynamic_support": bb_lower,
            "price_vs_resistance": ((resistance_level - current_price) / current_price)
            * 100,
            "price_vs_support": ((current_price - support_level) / support_level)
            * 100,
        }
    except Exception as exc:
        print(f"支撑阻力计算失败: {exc}")
        return {}


def get_market_trend(df: pd.DataFrame) -> Dict[str, Any]:
    """基于均线与 MACD 判断短、中期及整体趋势。"""
    try:
        current_price = df["close"].iloc[-1]
        trend_short = "上涨" if current_price > df["sma_20"].iloc[-1] else "下跌"
        trend_medium = "上涨" if current_price > df["sma_50"].iloc[-1] else "下跌"
        macd_trend = "多头" if df["macd"].iloc[-1] > df["macd_signal"].iloc[-1] else "空头"

        if trend_short == "上涨" and trend_medium == "上涨":
            overall_trend = "强势上涨"
        elif trend_short == "下跌" and trend_medium == "下跌":
            overall_trend = "强势下跌"
        else:
            overall_trend = "震荡整理"

        return {
            "short_term": trend_short,
            "medium_term": trend_medium,
            "macd": macd_trend,
            "overall": overall_trend,
            "rsi_level": df["rsi"].iloc[-1],
        }
    except Exception as exc:
        print(f"趋势分析失败: {exc}")
        return {}


def get_stock_ohlcv_enhanced() -> Dict[str, Any] | None:
    """封装行情获取 + 指标处理，供策略主逻辑调用。"""
    try:
        df = fetch_price_dataframe(
            TRADE_CONFIG["symbol"],
            TRADE_CONFIG["timeframe"],
            TRADE_CONFIG["data_points"],
        )
        df = calculate_technical_indicators(df)
        if len(df) < 2:
            raise RuntimeError("K线数量不足以进行分析")

        current_data = df.iloc[-1]
        previous_data = df.iloc[-2]

        trend_analysis = get_market_trend(df)
        levels_analysis = get_support_resistance_levels(df)

        records = df.tail(10).reset_index()[
            ["timestamp", "open", "high", "low", "close", "volume"]
        ]
        records["timestamp"] = records["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

        return {
            "symbol": TRADE_CONFIG["symbol"],
            "price": float(current_data["close"]),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "high": float(current_data["high"]),
            "low": float(current_data["low"]),
            "volume": float(current_data["volume"]),
            "timeframe": TRADE_CONFIG["timeframe"],
            "price_change": float(
                (current_data["close"] - previous_data["close"]) / previous_data["close"]
                * 100
            ),
            "kline_data": records.to_dict("records"),
            "technical_data": {
                "sma_5": float(current_data.get("sma_5", 0)),
                "sma_20": float(current_data.get("sma_20", 0)),
                "sma_50": float(current_data.get("sma_50", 0)),
                "rsi": float(current_data.get("rsi", 0)),
                "macd": float(current_data.get("macd", 0)),
                "macd_signal": float(current_data.get("macd_signal", 0)),
                "macd_histogram": float(current_data.get("macd_histogram", 0)),
                "bb_upper": float(current_data.get("bb_upper", 0)),
                "bb_lower": float(current_data.get("bb_lower", 0)),
                "bb_position": float(current_data.get("bb_position", 0)),
                "volume_ratio": float(current_data.get("volume_ratio", 0)),
            },
            "trend_analysis": trend_analysis,
            "levels_analysis": levels_analysis,
            "full_data": df,
        }
    except Exception as exc:
        print(f"获取股票行情失败: {exc}")
        return None


def generate_technical_analysis_text(price_data: Dict[str, Any]) -> str:
    """将结构化指标转成自然语言，提升 DeepSeek 上下文质量。"""
    if "technical_data" not in price_data:
        return "技术指标数据不可用"

    tech = price_data["technical_data"]
    trend = price_data.get("trend_analysis", {})
    levels = price_data.get("levels_analysis", {})
    symbol = price_data.get("symbol", "标的")

    def safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    return f"""
    【{symbol} 技术指标分析】
    📈 移动平均线:
    - 5周期: {safe_float(tech['sma_5']):.2f}
    - 20周期: {safe_float(tech['sma_20']):.2f}
    - 50周期: {safe_float(tech['sma_50']):.2f}

    🎯 趋势: 短期 {trend.get('short_term', 'N/A')} | 中期 {trend.get('medium_term', 'N/A')} | 总体 {trend.get('overall', 'N/A')}
    📊 MACD: {safe_float(tech['macd']):.4f} / 信号线 {safe_float(tech['macd_signal']):.4f}
    💪 RSI: {safe_float(tech['rsi']):.2f}
    🎚️ 布林带位置: {safe_float(tech['bb_position']):.2%}
    💡 关键水平: 阻力 {safe_float(levels.get('static_resistance', 0)):.2f} / 支撑 {safe_float(levels.get('static_support', 0)):.2f}
    """


def get_sentiment_indicators() -> Dict[str, Any] | None:
    """用基准指数涨跌情况快速推导市场情绪。"""
    try:
        df = fetch_price_dataframe(TRADE_CONFIG["benchmark_symbol"], "1d", 60)
        df["returns"] = df["close"].pct_change()
        recent = df["returns"].dropna()
        if recent.empty:
            return None

        positive_ratio = float((recent > 0).mean())
        negative_ratio = float((recent < 0).mean())
        net_sentiment = positive_ratio - negative_ratio
        short_momentum = float(recent.tail(5).sum())
        long_momentum = float(recent.tail(21).sum())

        return {
            "benchmark": TRADE_CONFIG["benchmark_symbol"],
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio,
            "net_sentiment": net_sentiment,
            "short_momentum": short_momentum,
            "long_momentum": long_momentum,
            "data_points": len(recent),
        }
    except Exception as exc:
        print(f"情绪指标获取失败: {exc}")
        return None


def calculate_intelligent_position(
    signal_data: Dict[str, Any],
    price_data: Dict[str, Any],
    current_position: Dict[str, Any] | None,
) -> int:
    """根据信号置信度、趋势与 RSI 计算目标股数。"""
    config = TRADE_CONFIG["position_management"]

    if not config.get("enable_intelligent_position", True):
        return max(config.get("min_shares", 1), 1)

    confidence = signal_data.get("confidence", "MEDIUM").upper()
    multiplier = {
        "HIGH": config["high_confidence_multiplier"],
        "MEDIUM": config["medium_confidence_multiplier"],
        "LOW": config["low_confidence_multiplier"],
    }.get(confidence, config["medium_confidence_multiplier"])

    trend = price_data.get("trend_analysis", {}).get("overall", "震荡整理")
    trend_multiplier = config["trend_strength_multiplier"] if trend in {"强势上涨", "强势下跌"} else 1.0

    rsi = price_data.get("technical_data", {}).get("rsi", 50)
    rsi_multiplier = 0.7 if (rsi > 75 or rsi < 25) else 1.0

    suggested_cash = (
        config["base_cash_amount"] * multiplier * trend_multiplier * rsi_multiplier
    )

    capital = config.get("account_capital", settings.initial_capital)
    max_cash = max(capital * config["max_position_ratio"], config["base_cash_amount"])
    final_cash = min(suggested_cash, max_cash)

    price = max(price_data.get("price", 0), 1e-6)
    min_shares = max(config.get("min_shares", 1), 1)
    shares = max(int(final_cash // price), min_shares)
    return shares


def get_current_position() -> Dict[str, Any] | None:
    """返回当前模拟持仓，尚未接入真实下单。"""
    if portfolio_state is None:
        return None
    return {
        "side": portfolio_state.get("side"),
        "size": portfolio_state.get("size", 0),
        "entry_price": portfolio_state.get("entry_price"),
    }


def safe_json_parse(content: str) -> Dict[str, Any] | None:
    """尽量从模型输出中提取合法 JSON。"""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        try:
            json_str = re.search(r"\{.*\}", content, re.S)
            if not json_str:
                return None
            text = json_str.group()
            text = re.sub(r"(\w+):", r'"\1":', text)
            text = re.sub(r",\s*}", "}", text)
            text = re.sub(r",\s*]", "]", text)
            return json.loads(text)
        except Exception:
            return None


def create_fallback_signal(price_data: Dict[str, Any]) -> Dict[str, Any]:
    """在 DeepSeek 失联时提供保守的 HOLD 信号。"""
    price = price_data.get("price", 0)
    return {
        "signal": "HOLD",
        "reason": "AI 分析不可用，启用保守模式。",
        "stop_loss": price * 0.97,
        "take_profit": price * 1.03,
        "confidence": "LOW",
        "is_fallback": True,
    }


def analyze_with_deepseek(price_data: Dict[str, Any]) -> Dict[str, Any]:
    """整理提示词并调用 DeepSeek 生成交易建议。"""
    if deepseek_client is None:
        print("⚠️ 未安装 openai 库或未设置 DEEPSEEK_API_KEY，返回保守信号。")
        return create_fallback_signal(price_data)

    price_history.append(price_data)
    if len(price_history) > 50:
        price_history.pop(0)

    kline_text = f"【最近10根{TRADE_CONFIG['timeframe']}K线】\n"
    for idx, kline in enumerate(price_data.get("kline_data", []), start=1):
        open_price = kline["open"]
        close = kline["close"]
        change = ((close - open_price) / open_price) * 100 if open_price else 0
        kline_text += (
            f"K{idx}: {'阳线' if close >= open_price else '阴线'} 开:{open_price:.2f} 收:{close:.2f} 涨跌:{change:+.2f}%\n"
        )

    indicator_text = generate_technical_analysis_text(price_data)

    sentiment = get_sentiment_indicators()
    if sentiment:
        sign = "+" if sentiment["net_sentiment"] >= 0 else ""
        sentiment_text = (
            "【市场情绪】"
            f"参考标的 {sentiment['benchmark']}，上涨占比 {sentiment['positive_ratio']:.1%}，"
            f"下跌占比 {sentiment['negative_ratio']:.1%}，净值 {sign}{sentiment['net_sentiment']:.2f}。"
        )
    else:
        sentiment_text = "【市场情绪】暂不可用。"

    if signal_history:
        last_signal = signal_history[-1]
        signal_desc = (
            f"信号: {last_signal.get('signal')} / 信心: {last_signal.get('confidence')} / "
            f"理由: {last_signal.get('reason', 'N/A')}"
        )
    else:
        signal_desc = "暂无历史信号"

    current_pos = get_current_position()
    position_text = (
        "当前无持仓"
        if not current_pos
        else f"持有{current_pos['side']}仓 {current_pos['size']} 股，成本 {current_pos['entry_price']:.2f}"
    )

    prompt = f"""
    你是一名专业的股票量化分析师，目标标的是 {TRADE_CONFIG['symbol']}。请基于以下信息输出下一步操作建议：

    {kline_text}

    {indicator_text}

    {sentiment_text}

    【当前行情】
    - 最新价格: {price_data['price']:.2f}
    - 时间: {price_data['timestamp']}
    - 当根最高/最低: {price_data['high']:.2f} / {price_data['low']:.2f}
    - 成交量: {price_data['volume']:.2f}
    - 价格变化: {price_data['price_change']:+.2f}%
    - 当前持仓: {position_text}

    【上次信号】{signal_desc}

    请输出 JSON，格式如下：
    {{
        "signal": "BUY|SELL|HOLD",
        "reason": "简洁的交易理由",
        "stop_loss": 止损价,
        "take_profit": 止盈价,
        "confidence": "HIGH|MEDIUM|LOW"
    }}
    """

    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一位专注于中短线趋势的股票量化交易员，"
                    "请严格基于提供的数据给出客观、结构化的交易建议。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        stream=False,
        temperature=0.4,
    )

    content = response.choices[0].message.content  # type: ignore[index]
    parsed = safe_json_parse(content)
    if not parsed:
        print("⚠️ DeepSeek 返回无法解析，启用备用信号。原文:", content)
        return create_fallback_signal(price_data)

    signal_history.append(parsed)
    return parsed


def execute_intelligent_trade(signal_data: Dict[str, Any], price_data: Dict[str, Any]) -> None:
    """根据 AI 信号调整本地持仓状态，必要时触发 Longbridge 实盘委托。"""
    global portfolio_state

    current_position = get_current_position()
    position_size = calculate_intelligent_position(signal_data, price_data, current_position)
    signal = signal_data.get("signal", "HOLD").upper()
    target_side: str | None = None
    target_size = 0

    def _set_position(side: str | None, size: int = 0) -> None:
        global portfolio_state
        if side is None or size <= 0:
            portfolio_state = None
        else:
            portfolio_state = {
                "side": side,
                "size": size,
                "entry_price": price_data["price"],
            }

    print(
        f"信号: {signal} | 计划仓位: {position_size} 股 | 信心: {signal_data.get('confidence', 'N/A')}"
    )

    if signal == "BUY":
        target_side = "long"
        target_size = position_size
        if current_position and current_position["side"] == "short":
            print("➡️ 平空并转多，模拟平仓数量:", current_position["size"])
        elif current_position and current_position["side"] == "long":
            print(
                f"🔁 已有多头 {current_position['size']} 股，调整为 {position_size} 股"
            )
        else:
            print(f"🆕 开启多头仓位 {position_size} 股")
        _set_position("long", position_size)

    elif signal == "SELL":
        target_side = "short" if TRADE_CONFIG["allow_short"] else None
        target_size = position_size if TRADE_CONFIG["allow_short"] else 0
        if current_position and current_position["side"] == "long":
            print("➡️ 平掉现有多头，数量:", current_position["size"])
            if TRADE_CONFIG["allow_short"]:
                print(f"🔄 反手做空 {position_size} 股")
                _set_position("short", position_size)
            else:
                _set_position(None)
        elif current_position and current_position["side"] == "short":
            print(
                f"🔁 已有空头 {current_position['size']} 股，调整为 {position_size} 股"
            )
            if TRADE_CONFIG["allow_short"]:
                _set_position("short", position_size)
            else:
                _set_position(None)
        else:
            if TRADE_CONFIG["allow_short"]:
                print(f"🆕 开启空头仓位 {position_size} 股")
                _set_position("short", position_size)
            else:
                print("⚠️ 当前策略未启用做空，保持空仓")
                _set_position(None)

    else:
        if current_position:
            target_side = current_position["side"]
            target_size = current_position.get("size", 0)
        print("⏸ 观望，保持当前仓位不变。")

    if trade_executor:
        try:
            trade_executor.sync_position(
                signal=signal,
                target_side=target_side,
                target_size=target_size,
                price=price_data.get("price", 0),
                current_position=current_position,
                reason=signal_data.get("reason", ""),
            )
        except Exception as exc:
            print(f"❌ Longbridge 执行异常: {exc}")

    if TRADE_CONFIG["test_mode"]:
        print("（测试模式）以上操作仅为模拟记录，并未触发真实下单。")


def analyze_with_deepseek_with_retry(price_data: Dict[str, Any], max_retries: int = 2) -> Dict[str, Any]:
    """失败重试的包装器，提升调用稳定性。"""
    for attempt in range(max_retries):
        try:
            signal_data = analyze_with_deepseek(price_data)
            if signal_data and not signal_data.get("is_fallback"):
                return signal_data
            print(f"第 {attempt + 1} 次调用 DeepSeek 失败，重试中...")
        except Exception as exc:
            print(f"第 {attempt + 1} 次调用 DeepSeek 异常: {exc}")
        time.sleep(1)
    return create_fallback_signal(price_data)


def wait_for_next_period() -> int:
    """计算距离下一根 K 线开盘还剩多少秒。"""
    interval = timeframe_to_minutes(TRADE_CONFIG["timeframe"])
    now = datetime.now()
    remainder = now.minute % interval
    if remainder == 0 and now.second == 0:
        return 0
    minutes_to_wait = (interval - remainder) % interval
    seconds_to_wait = minutes_to_wait * 60 - now.second
    if seconds_to_wait <= 0:
        seconds_to_wait += interval * 60
    if minutes_to_wait > 0:
        print(f"🕒 等待 {minutes_to_wait} 分 {now.second if now.second else 0} 秒进入下一个周期...")
    else:
        print(f"🕒 等待 {seconds_to_wait} 秒进入下一个周期...")
    return seconds_to_wait


def trading_bot() -> None:
    """单次执行流程：等待 → 拉数据 → 调 AI → 执行信号。"""
    wait_seconds = wait_for_next_period()
    if wait_seconds > 0:
        time.sleep(wait_seconds)

    print("\n" + "=" * 60)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    price_data = get_stock_ohlcv_enhanced()
    if not price_data:
        return

    print(
        f"{price_data['symbol']} 当前价格: ${price_data['price']:.2f} | 周期 {price_data['timeframe']} | 涨跌 {price_data['price_change']:+.2f}%"
    )

    signal_data = analyze_with_deepseek_with_retry(price_data)
    if signal_data.get("is_fallback"):
        print("⚠️ 使用备用信号")

    execute_intelligent_trade(signal_data, price_data)


def main() -> None:
    """程序入口：初始化并循环执行策略。"""
    print("股票版 DeepSeek 智能策略启动中……")
    print(
        f"标的: {TRADE_CONFIG['symbol']} | 周期: {TRADE_CONFIG['timeframe']} | 测试模式: {TRADE_CONFIG['test_mode']}"
    )

    if not setup_market_environment():
        print("初始化失败，退出程序。")
        return

    print("执行频率：与 K 线周期保持一致，整周期触发。")

    while True:
        trading_bot()
        time.sleep(60)


if __name__ == "__main__":
    main()
