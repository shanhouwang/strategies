"""超级趋势回测流程的配置解析。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """从环境变量中读取的应用配置。"""

    # LongPort 凭证
    longport_app_key: str = Field(..., env="LONGPORT_APP_KEY")
    longport_app_secret: str = Field(..., env="LONGPORT_APP_SECRET")
    longport_access_token: str = Field(..., env="LONGPORT_ACCESS_TOKEN")

    # 数据与策略参数
    symbol: str = Field("SOXL.US")  # 回测标的代码
    interval: str = Field("15m")  # K线周期（需符合 LongPort 规范）
    start: Optional[datetime] = None  # 回测起始时间
    end: Optional[datetime] = None  # 回测结束时间
    supertrend_period: int = 10  # 超级趋势 ATR 周期
    supertrend_multiplier: float = 3.0  # 超级趋势 ATR 乘数
    enable_long: bool = True  # 是否启用做多逻辑
    enable_short: bool = False  # 是否启用做空逻辑
    atr_period: Optional[int] = None  # ATR 止盈止损计算周期，未指定则使用超级趋势周期
    atr_stop_multiplier: float = 0.0  # ATR 止损乘数（0 表示关闭），打开后回测效果不是很理想
    atr_take_profit_multiplier: float = 0.0  # ATR 止盈乘数（0 表示关闭），打开后回测效果不是很理想
    volatility_filter_atr_pct: float = 0.0  # ATR/价格 波动率过滤阈值
    trend_filter_period: int = 0  # 趋势过滤均线周期（0 表示关闭）

    # 回测参数
    initial_capital: float = 10_000.0  # 初始资金
    risk_free_rate: float = 0.0  # 无风险利率（占位）
    commission: float = 0.0  # 单笔固定佣金
    slippage: float = 0.0  # 每股滑点
    max_position: int = 1  # 最大同时持仓数量（预留）
    risk_per_trade_pct: float = 0.0  # 每笔交易风险占权益百分比，0 表示全仓
    risk_atr_multiplier: Optional[float] = None  # 用于风险单位计算的 ATR 乘数（默认退回策略设置）
    cooldown_bars: int = 0  # 信号冷却期，单位：K 线数量

    # 策略选择
    strategy: str = Field("supertrend", env="STRATEGY")
    strategy_params: Dict[str, Any] = Field(default_factory=dict, env="STRATEGY_PARAMS")

    # 费率（参考长桥官网，可按需覆盖）
    commission_rate: float = 0.0  # 佣金率（默认免佣，可覆盖）
    commission_min: float = 0.0  # 最低佣金
    platform_fee_fixed_per_share: float = 0.005  # 固定平台费每股成本
    platform_fee_tiers: List[Tuple[int, Optional[int], float]] = Field(
        default_factory=lambda: [
            (1, 5000, 0.0070),  # 区间下限、上限、每股费用
            (5001, 10000, 0.0060),
            (10001, 100000, 0.0050),
            (100001, 1_000_000, 0.0040),
            (1_000_001, None, 0.0030),
        ]
    )
    stock_transfer_in_fee: float = 0.0  # 股票转入费用

    # 输出路径
    artifacts_dir: Path = Path("artifacts")  # 回测产物目录
    trades_csv: Path = Path("artifacts/trades.csv")  # 交易记录文件
    chart_path: Optional[Path] = Field(
        default=None,
        env="CHART_PATH",
        description="图表输出路径，未设置时按策略名称自动生成。",
    )  # 图表输出路径

    # pydantic v2 设置
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )

    @field_validator("interval")
    def validate_interval(cls, value: str) -> str:
        """确保时间周期符合 LongPort 要求。"""
        allowed = {"1m", "5m", "15m", "30m", "60m", "1d"}
        if value not in allowed:
            raise ValueError(f"interval 必须是 {sorted(allowed)} 之一")
        return value

    @field_validator("strategy_params", mode="before")
    def _parse_strategy_params(cls, value: Any) -> Dict[str, Any]:
        if value in (None, "", {}):
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("STRATEGY_PARAMS 必须是 JSON 对象字符串。") from exc
            if not isinstance(parsed, dict):
                raise ValueError("STRATEGY_PARAMS 解析后必须是对象。")
            return parsed
        raise ValueError(f"无法解析策略参数：{value!r}")

    @field_validator("artifacts_dir", "trades_csv", "chart_path", mode="before")
    def _expand_paths(cls, value: Path | str | None) -> Path | None:
        if value is None:
            return None
        return Path(value).expanduser()

    def chart_path_for(self, strategy_name: str) -> Path:
        """获取指定策略的图表输出路径。"""
        if self.chart_path:
            return Path(self.chart_path)
        filename = f"{strategy_name}.png" if strategy_name else "chart.png"
        return Path(self.artifacts_dir) / filename

    def trades_path_for(self, strategy_name: str) -> Path:
        """获取指定策略的交易记录输出路径。"""
        base = Path(self.trades_csv)
        if base.suffix:
            return base.with_name(f"{strategy_name}{base.suffix}")
        return base / f"{strategy_name}.csv"
