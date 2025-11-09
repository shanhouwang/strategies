"""基于 DeepSeek AI 信号的股票策略，支持在回测框架里逐根调用大模型。"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Iterable, Protocol

import pandas as pd
from dotenv import load_dotenv
import requests

try:  # pragma: no cover - optional dependency
    from openai import OpenAI as OpenAIAPI
except ImportError:  # pragma: no cover - lazy import
    OpenAIAPI = None

from ... import plotting
from ...config import Settings
from ...backtest import Trade
from ..base import Strategy


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


def _safe_json_parse(content: str) -> Dict[str, Any] | None:
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


class LLMClient(Protocol):
    """统一的 LLM 客户端协议。"""

    def create_completion(
        self,
        messages: List[Dict[str, str]],
        *,
        model: str,
        temperature: float,
    ) -> str | None:
        ...


class DeepseekLLMClient:
    """直接调用 DeepSeek REST API。"""

    def __init__(self, *, api_key: str, base_url: str, timeout: float) -> None:
        self.api_key = api_key
        base = base_url.rstrip("/")
        if not base.endswith("/chat/completions"):
            base = base + "/chat/completions"
        self.endpoint = base
        self.timeout = timeout
        self.session = requests.Session()

    def create_completion(
        self,
        messages: List[Dict[str, str]],
        *,
        model: str,
        temperature: float,
    ) -> str | None:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = self.session.post(
            self.endpoint,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        if resp.status_code == 402:
            print("⚠️ DeepSeek API 返回余额不足，请检查账户额度。")
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return None


class OpenAILLMClient:
    """调用 OpenAI 官方 SDK（Responses API）。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None,
        timeout: float,
    ) -> None:
        if OpenAIAPI is None:  # pragma: no cover - optional dep
            raise RuntimeError("未安装 openai 包，无法使用 OpenAI 提供商。")
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAIAPI(**kwargs)
        self.timeout = timeout

    @staticmethod
    def _format_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        formatted: List[Dict[str, str]] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            formatted.append({"role": role, "content": str(content)})
        return formatted

    @staticmethod
    def _extract_output_text(response: Any) -> str | None:
        text = getattr(response, "output_text", None)
        if text:
            if isinstance(text, list):
                return "".join(str(item) for item in text)
            return str(text)

        data: Dict[str, Any] | None = None
        if hasattr(response, "model_dump"):
            try:
                data = response.model_dump()
            except Exception:  # pragma: no cover - defensive
                data = None
        if data is None and hasattr(response, "to_dict"):
            try:
                data = response.to_dict()  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - defensive
                data = None
        if not data:
            return None

        parts: List[str] = []
        for item in data.get("output", []) or []:
            for content in item.get("content", []) or []:
                text_block = content.get("text")
                if isinstance(text_block, dict):
                    parts.append(str(text_block.get("value", "")))
                elif isinstance(text_block, str):
                    parts.append(text_block)
        if parts:
            return "".join(parts)
        return None

    def create_completion(
        self,
        messages: List[Dict[str, str]],
        *,
        model: str,
        temperature: float,
    ) -> str | None:
        formatted_messages = self._format_messages(messages)
        client = self.client
        if self.timeout and hasattr(self.client, "with_options"):
            client = self.client.with_options(timeout=self.timeout)
        request_kwargs: Dict[str, Any] = {
            "model": model,
            "input": formatted_messages,
        }
        if temperature is not None:
            request_kwargs["temperature"] = temperature
        try:
            response = client.responses.create(**request_kwargs)
        except Exception as exc:
            if "Unsupported parameter" in str(exc) and "temperature" in str(exc):
                request_kwargs.pop("temperature", None)
                response = client.responses.create(**request_kwargs)
            else:
                raise
        return self._extract_output_text(response)


class StockDeepseekStrategy(Strategy):
    """将 DeepSeek 决策嵌入回测，生成多空入场/离场信号。"""

    name = "stock_deepseek_ai"

    def __init__(self, *, settings: Settings, **params: Any) -> None:
        super().__init__(settings=settings, **params)
        load_dotenv()
        self.temperature = float(params.get("temperature", os.getenv("DEEPSEEK_TEMPERATURE", 0.35)))
        self.request_timeout = float(params.get("deepseek_timeout", os.getenv("DEEPSEEK_TIMEOUT", 30)))

        self.allow_short = bool(params.get("allow_short", _env_bool("ALLOW_SHORT", False)))
        self.analysis_window = int(params.get("analysis_window", 96))
        self.cache_path = Path(
            params.get(
                "deepseek_cache",
                os.getenv("DEEPSEEK_CACHE", "artifacts/deepseek_ai_cache.json"),
            )
        )
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if self.cache_path.exists():
            try:
                self.signal_cache: Dict[str, Dict[str, Any]] = json.loads(
                    self.cache_path.read_text(encoding="utf-8")
                )
            except json.JSONDecodeError:
                self.signal_cache = {}
        else:
            self.signal_cache = {}
        self._cache_dirty = False

        self.base_cash_amount = _env_float("BASE_CASH_AMOUNT", 1_000.0)
        self.max_position_ratio = _env_float("MAX_POSITION_RATIO", 0.25)
        self.llm_provider = (params.get("llm_provider") or os.getenv("LLM_PROVIDER", "deepseek")).lower()
        self.model_name = params.get("model") or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

        if self.llm_provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("缺少 OPENAI_API_KEY，无法调用 OpenAI。")
            base_url = os.getenv("OPENAI_BASE_URL")
            self.model_name = params.get("openai_model") or 'gpt-5'
            self.llm_client: LLMClient = OpenAILLMClient(
                api_key=api_key,
                base_url=base_url,
                timeout=self.request_timeout,
            )
        else:
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise RuntimeError("缺少 DEEPSEEK_API_KEY，无法调用 DeepSeek。")
            base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
            self.model_name = params.get("model") or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
            self.llm_client = DeepseekLLMClient(
                api_key=api_key,
                base_url=base_url,
                timeout=self.request_timeout,
            )

    def _persist_cache(self) -> None:
        if self._cache_dirty:
            self.cache_path.write_text(
                json.dumps(self.signal_cache, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self._cache_dirty = False

    def _call_llm(self, messages: List[Dict[str, str]]) -> str | None:
        try:
            return self.llm_client.create_completion(
                messages=messages,
                model=self.model_name,
                temperature=self.temperature,
            )
        except requests.RequestException as exc:
            print(f"⚠️ {self.llm_provider} API 请求失败: {exc}")
            return None
        except Exception as exc:  # pragma: no cover - SDK 级别异常
            print(f"⚠️ {self.llm_provider} API 异常: {exc}")
            return None

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        result["sma_5"] = result["close"].rolling(window=5, min_periods=1).mean()
        result["sma_20"] = result["close"].rolling(window=20, min_periods=1).mean()
        result["sma_50"] = result["close"].rolling(window=50, min_periods=1).mean()
        result["ema_12"] = result["close"].ewm(span=12).mean()
        result["ema_26"] = result["close"].ewm(span=26).mean()
        result["macd"] = result["ema_12"] - result["ema_26"]
        result["macd_signal"] = result["macd"].ewm(span=9).mean()
        result["macd_histogram"] = result["macd"] - result["macd_signal"]

        delta = result["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        result["rsi"] = 100 - (100 / (1 + rs))

        result["bb_middle"] = result["close"].rolling(20).mean()
        bb_std = result["close"].rolling(20).std()
        result["bb_upper"] = result["bb_middle"] + bb_std * 2
        result["bb_lower"] = result["bb_middle"] - bb_std * 2
        with pd.option_context("mode.use_inf_as_na", True):
            result["bb_position"] = (result["close"] - result["bb_lower"]) / (
                result["bb_upper"] - result["bb_lower"]
            )
        result["volume_ma"] = result["volume"].rolling(20).mean()
        result["volume_ratio"] = result["volume"] / result["volume_ma"]
        result["resistance"] = result["high"].rolling(20).max()
        result["support"] = result["low"].rolling(20).min()
        return result.bfill().ffill()

    @staticmethod
    def _support_levels(df_slice: pd.DataFrame, lookback: int = 20) -> Dict[str, float]:
        tail = df_slice.tail(lookback)
        if tail.empty:
            return {}
        current_price = tail["close"].iloc[-1]
        resistance = tail["high"].max()
        support = tail["low"].min()
        bb_upper = tail["bb_upper"].iloc[-1]
        bb_lower = tail["bb_lower"].iloc[-1]
        return {
            "static_resistance": float(resistance),
            "static_support": float(support),
            "dynamic_resistance": float(bb_upper),
            "dynamic_support": float(bb_lower),
            "price_vs_resistance": float((resistance - current_price) / current_price * 100)
            if current_price
            else 0.0,
            "price_vs_support": float((current_price - support) / support * 100)
            if support
            else 0.0,
        }

    @staticmethod
    def _trend_from_row(row: pd.Series) -> Dict[str, Any]:
        price = row["close"]
        sma20 = row.get("sma_20")
        sma50 = row.get("sma_50")
        macd = row.get("macd")
        macd_sig = row.get("macd_signal")
        short = "上涨" if pd.notna(sma20) and price > sma20 else "下跌"
        medium = "上涨" if pd.notna(sma50) and price > sma50 else "下跌"
        if short == "上涨" and medium == "上涨":
            overall = "强势上涨"
        elif short == "下跌" and medium == "下跌":
            overall = "强势下跌"
        else:
            overall = "震荡整理"
        macd_trend = "多头" if pd.notna(macd) and pd.notna(macd_sig) and macd > macd_sig else "空头"
        return {
            "short_term": short,
            "medium_term": medium,
            "overall": overall,
            "macd": macd_trend,
            "rsi_level": float(row.get("rsi", 50) or 50),
        }

    @staticmethod
    def _sentiment_from_returns(returns: pd.Series) -> Dict[str, Any] | None:
        clean = returns.dropna()
        if clean.empty:
            return None
        positive = float((clean > 0).mean())
        negative = float((clean < 0).mean())
        net = positive - negative
        return {
            "positive_ratio": positive,
            "negative_ratio": negative,
            "net_sentiment": net,
            "short_momentum": float(clean.tail(5).sum()),
            "long_momentum": float(clean.tail(21).sum()),
        }

    @staticmethod
    def _kline_records(df_slice: pd.DataFrame) -> List[Dict[str, Any]]:
        subset = df_slice.tail(10).reset_index()
        time_col = subset.columns[0]
        subset["timestamp"] = pd.to_datetime(subset[time_col]).dt.strftime("%Y-%m-%d %H:%M:%S")
        return subset[["timestamp", "open", "high", "low", "close", "volume"]].to_dict("records")

    def _technical_text(
        self, symbol: str, row: pd.Series, trend: Dict[str, Any], levels: Dict[str, Any]
    ) -> str:
        def safe(value: Any, default: float = 0.0) -> float:
            try:
                if value is None or (isinstance(value, float) and pd.isna(value)):
                    return default
                return float(value)
            except (TypeError, ValueError):
                return default

        return (
            f"【{symbol} 技术摘要】\n"
            f"MA(5/20/50): {safe(row.get('sma_5')):.2f} / {safe(row.get('sma_20')):.2f} / {safe(row.get('sma_50')):.2f}\n"
            f"RSI: {safe(row.get('rsi')):.2f} | MACD: {safe(row.get('macd')):.4f} vs {safe(row.get('macd_signal')):.4f}\n"
            f"布林位置: {safe(row.get('bb_position')):.2%}\n"
            f"趋势: 短期 {trend['short_term']} / 中期 {trend['medium_term']} / 总体 {trend['overall']}\n"
            f"阻力 {safe(levels.get('static_resistance')):.2f} / 支撑 {safe(levels.get('static_support')):.2f}"
        )

    def _build_prompt(
        self,
        symbol: str,
        timeframe: str,
        price_data: Dict[str, Any],
        trend: Dict[str, Any],
        levels: Dict[str, Any],
        sentiment_text: str,
        kline_text: str,
        technical_text: str,
        position_text: str,
        last_signal_desc: str,
    ) -> str:
        return f"""
你是一名专业的量化交易分析师，当前分析标的是 {symbol}，周期为 {timeframe}。

{kline_text}

{technical_text}

{sentiment_text}

【当前行情】
- 最新价格: {price_data['price']:.2f}
- 时间: {price_data['timestamp']}
- 当根范围: {price_data['low']:.2f} - {price_data['high']:.2f}
- 成交量: {price_data['volume']:.2f}
- 涨跌幅: {price_data['price_change']:+.2f}%
- 持仓: {position_text}

【上次信号】{last_signal_desc}

请严格输出 JSON：
{{
  "signal": "BUY|SELL|HOLD",
  "reason": "简要原因",
  "stop_loss": 数值,
  "take_profit": 数值,
  "confidence": "HIGH|MEDIUM|LOW"
}}
"""

    def _request_signal(
        self,
        ts_key: str,
        prompt: str,
    ) -> Dict[str, Any]:
        cached = self.signal_cache.get(ts_key)
        if cached and not cached.get("is_fallback"):
            return cached

        messages = [
            {
                "role": "system",
                "content": "你是偏趋势、顺势交易的股票量化顾问，回答时需结构化且果断。",
            },
            {"role": "user", "content": prompt},
        ]
        content = self._call_llm(messages) or ""
        parsed = _safe_json_parse(content or "")
        if not parsed:
            return {
                "signal": "HOLD",
                "reason": "AI 输出解析失败，保守观望。",
                "stop_loss": 0.0,
                "take_profit": 0.0,
                "confidence": "LOW",
                "is_fallback": True,
            }

        if parsed.get("signal") not in {"BUY", "SELL", "HOLD"}:
            parsed["signal"] = "HOLD"
            parsed["is_fallback"] = True
            return parsed

        print(
            f"[DeepSeek] {ts_key} signal={parsed.get('signal')} "
            f"conf={parsed.get('confidence')} reason={parsed.get('reason', '')[:60]}"
        )
        self.signal_cache[ts_key] = parsed
        self._cache_dirty = True
        return parsed

    def _price_payload(
        self,
        df_slice: pd.DataFrame,
        timeframe: str,
    ) -> Dict[str, Any]:
        current = df_slice.iloc[-1]
        prev = df_slice.iloc[-2] if len(df_slice) > 1 else current
        return {
            "symbol": current.get("symbol", ""),
            "price": float(current["close"]),
            "timestamp": df_slice.index[-1].strftime("%Y-%m-%d %H:%M:%S"),
            "high": float(current["high"]),
            "low": float(current["low"]),
            "volume": float(current["volume"]),
            "timeframe": timeframe,
            "price_change": float(
                (current["close"] - prev["close"]) / prev["close"] * 100
            )
            if prev["close"]
            else 0.0,
        }

    def prepare_data(self, price_df: pd.DataFrame) -> pd.DataFrame:
        if price_df is None or price_df.empty:
            raise ValueError("输入数据为空，无法生成 DeepSeek 信号。")

        df = price_df.copy()
        if "symbol" not in df.columns:
            df["symbol"] = self.params.get("symbol", self.settings.symbol)
        df = self._calculate_indicators(df)
        long_entry: List[int] = []
        long_exit: List[int] = []
        short_entry: List[int] = []
        short_exit: List[int] = []
        signal_list: List[str] = []
        confidence_list: List[str] = []
        reason_list: List[str] = []

        signal_history: List[Dict[str, Any]] = []
        current_side: str | None = None
        timeframe = self.params.get("timeframe", self.settings.interval)

        for idx in range(len(df)):
            window_start = max(0, idx + 1 - self.analysis_window)
            df_slice = df.iloc[window_start : idx + 1]
            ts = df_slice.index[-1]
            ts_key = ts.isoformat()
            row = df_slice.iloc[-1]
            levels = self._support_levels(df_slice)
            trend = self._trend_from_row(row)
            returns = df_slice["close"].pct_change()
            sentiment = self._sentiment_from_returns(returns)
            sentiment_text = (
                "【市场情绪】上涨占比 {0:.1%}，下跌占比 {1:.1%}，净值 {2:+.2f}".format(
                    sentiment["positive_ratio"],
                    sentiment["negative_ratio"],
                    sentiment["net_sentiment"],
                )
                if sentiment
                else "【市场情绪】暂无有效样本。"
            )

            kline_records = self._kline_records(df_slice)
            kline_text_lines = []
            for i, rec in enumerate(kline_records, 1):
                change = (
                    (rec["close"] - rec["open"]) / rec["open"] * 100 if rec["open"] else 0
                )
                trend_word = "阳线" if rec["close"] >= rec["open"] else "阴线"
                kline_text_lines.append(
                    f"K{i}: {trend_word} 开:{rec['open']:.2f} 收:{rec['close']:.2f} 涨跌:{change:+.2f}%"
                )
            kline_text = "【最近K线】\n" + "\n".join(kline_text_lines)

            technical_text = self._technical_text(row["symbol"], row, trend, levels)
            position_text = (
                f"{current_side} 仓位" if current_side else "无持仓"
            )
            last_signal = signal_history[-1] if signal_history else None
            last_desc = (
                f"{last_signal.get('signal')} / {last_signal.get('confidence')} / {last_signal.get('reason', '')}"
                if last_signal
                else "暂无历史信号"
            )
            price_payload = self._price_payload(df_slice, timeframe)
            prompt = self._build_prompt(
                row["symbol"],
                timeframe,
                price_payload,
                trend,
                levels,
                sentiment_text,
                kline_text,
                technical_text,
                position_text,
                last_desc,
            )
            signal_data = self._request_signal(ts_key, prompt)
            signal_history.append(signal_data)

            ai_signal = (signal_data.get("signal") or "HOLD").upper()
            confidence = signal_data.get("confidence", "").upper() or "LOW"
            reason = signal_data.get("reason", "")

            open_long = open_short = close_long = close_short = 0
            if ai_signal == "BUY":
                if current_side == "short":
                    close_short = 1
                if current_side != "long":
                    open_long = 1
                current_side = "long"
            elif ai_signal == "SELL":
                if current_side == "long":
                    close_long = 1
                if self.allow_short:
                    if current_side != "short":
                        open_short = 1
                    current_side = "short"
                else:
                    current_side = None
            else:  # HOLD
                if current_side == "long":
                    close_long = 1
                elif current_side == "short":
                    close_short = 1
                current_side = None

            long_entry.append(open_long)
            short_entry.append(open_short)
            long_exit.append(close_long)
            short_exit.append(close_short)
            signal_list.append(ai_signal)
            confidence_list.append(confidence)
            reason_list.append(reason)

        df["long_entry"] = long_entry
        df["long_exit"] = long_exit
        df["short_entry"] = short_entry
        df["short_exit"] = short_exit
        df["ai_signal"] = signal_list
        df["ai_confidence"] = confidence_list
        df["ai_reason"] = reason_list

        self._persist_cache()
        return df

    def plot(
        self,
        data: pd.DataFrame,
        trades: Iterable[Trade],
        output_path: Path,
        equity_curve: pd.Series | None = None,
        initial_capital: float | None = None,
    ) -> Path:
        """输出带策略名称的资金曲线图。"""
        return plotting.plot_supertrend(
            data=data,
            trades=trades,
            equity_curve=equity_curve,
            initial_capital=initial_capital,
            output_path=output_path,
            title=f"{self.name} 策略资金曲线",
        )


__all__ = ["StockDeepseekStrategy"]
