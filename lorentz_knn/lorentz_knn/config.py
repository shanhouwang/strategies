"""Configuration objects for the Lorentzian KNN backtest pipeline."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import BaseModel, BaseSettings, Field, validator


class FeatureConfig(BaseModel):
    """Configuration for a single feature used by the KNN model."""

    name: str = Field(..., description="Feature identifier, e.g. RSI/WT/CCI/ADX.")
    param_a: int = Field(14, ge=1, description="Primary lookback length.")
    param_b: int = Field(1, ge=1, description="Secondary smoothing parameter.")

    @validator("name")
    def validate_name(cls, value: str) -> str:
        allowed = {"RSI", "WT", "CCI", "ADX"}
        if value.upper() not in allowed:
            raise ValueError(f"feature name 必须是 {sorted(allowed)} 之一")
        return value.upper()


class PartialTrailingLeg(BaseModel):
    """Definition for a partial trailing take-profit leg."""

    ratio: float = Field(..., gt=0.0, le=1.0, description="该腿占总仓位比例 (0-1)。")
    trigger_pct: float = Field(..., ge=0.0, description="触发移动止盈的收益百分比。")
    offset_pct: float = Field(..., ge=0.0, description="触发后允许的回撤百分比。")


class PartialStopLeg(BaseModel):
    """Definition for a partial stop-loss leg."""

    ratio: float = Field(..., gt=0.0, le=1.0, description="该腿占总仓位比例 (0-1)。")
    stop_loss_pct: float = Field(..., ge=0.0, description="该腿触发止损的亏损百分比。")


class Settings(BaseSettings):
    """Application configuration loaded from environment variables or CLI."""

    # LongPort credentials
    longport_app_key: Optional[str] = Field(None, env="LONGPORT_APP_KEY")
    longport_app_secret: Optional[str] = Field(None, env="LONGPORT_APP_SECRET")
    longport_access_token: Optional[str] = Field(None, env="LONGPORT_ACCESS_TOKEN")

    # Data slice
    symbol: str = Field("SOXL.US", description="回测标的代码")
    interval: str = Field("15m", description="K线周期，需符合 LongPort 定义")
    start: Optional[datetime] = Field(
        None, description="回测起始时间（ISO8601，本地时区）"
    )
    end: Optional[datetime] = Field(
        None, description="回测结束时间（ISO8601，本地时区）"
    )

    # Strategy hyper-parameters
    neighbors_count: int = Field(8, ge=1, le=100, description="KNN 邻居数量")
    max_bars_back: int = Field(
        8000, ge=100, description="搜索邻居时允许的最大历史长度"
    )
    feature_count: int = Field(
        5, ge=2, le=5, description="使用的特征数量，按照 features 列表顺序取前 N 个"
    )
    features: List[FeatureConfig] = Field(
        default_factory=lambda: [
            FeatureConfig(name="RSI", param_a=14, param_b=1),
            FeatureConfig(name="WT", param_a=10, param_b=11),
            FeatureConfig(name="CCI", param_a=20, param_b=1),
            FeatureConfig(name="ADX", param_a=20, param_b=2),
            FeatureConfig(name="RSI", param_a=9, param_b=1),
        ],
        description="可按 TradingView 指标配置的特征序列",
    )

    # Filters
    use_volatility_filter: bool = Field(True, description="是否启用波动率过滤")
    use_regime_filter: bool = Field(True, description="是否启用趋势状态过滤")
    use_adx_filter: bool = Field(False, description="是否启用 ADX 过滤")
    regime_threshold: float = Field(
        -0.1, description="趋势过滤阈值，取值为收益率百分比"
    )
    adx_threshold: float = Field(20.0, ge=0.0, le=100.0, description="ADX 阈值")
    use_ema_filter: bool = Field(False, description="是否使用 EMA 过滤多空")
    ema_period: int = Field(200, ge=1, description="EMA 滤波周期")
    use_sma_filter: bool = Field(False, description="是否使用 SMA 过滤多空")
    sma_period: int = Field(200, ge=1, description="SMA 滤波周期")

    # Kernel filter
    use_kernel_filter: bool = Field(True, description="是否借助核回归过滤信号")
    use_kernel_smoothing: bool = Field(
        False, description="核估计使用平滑交叉逻辑（对应 Pine useKernelSmoothing）"
    )
    kernel_window: int = Field(8, ge=3, description="核回归观察窗口大小（h）")
    kernel_relative_weight: float = Field(
        8.0,
        gt=0.0,
        description="核回归相对权重参数（r），数值越小越重视长周期",
    )
    kernel_regression_level: int = Field(
        25, ge=2, description="回归起点（x），控制拟合松紧程度"
    )
    kernel_lag: int = Field(2, ge=1, le=5, description="核平滑滞后长度")

    # Risk management
    open_short: bool = Field(True, description="是否允许开仓做空")
    enable_moving_profit: bool = Field(False, description="启用移动止盈逻辑")
    enable_stop_loss_only: bool = Field(False, description="仅使用固定止损")
    enable_fixed_profit: bool = Field(False, description="使用固定盈亏比止盈止损")
    enable_atr_profit: bool = Field(False, description="使用 ATR 止盈止损")
    enable_partial_trailing: bool = Field(False, description="启用分批移动止盈")
    enable_partial_stop_loss: bool = Field(False, description="启用分批止损")
    long_stop_loss_pct: float = Field(3.0, ge=0.1, description="多头固定止损百分比")
    long_trailing_trigger_pct: float = Field(
        10.0, ge=0.0, description="多头移动止盈触发百分比"
    )
    long_trailing_offset_pct: float = Field(
        30.0, ge=0.0, description="多头盈利回调幅度（百分比）"
    )
    long_profit_loss_ratio: float = Field(
        1.5, gt=0.0, description="多头固定盈亏比"
    )
    long_partial_trailing_legs: List[PartialTrailingLeg] = Field(
        default_factory=lambda: [
            PartialTrailingLeg(ratio=0.5, trigger_pct=5.0, offset_pct=40.0),
            PartialTrailingLeg(ratio=0.5, trigger_pct=10.0, offset_pct=30.0),
        ],
        description="多头分批移动止盈腿配置",
    )
    long_partial_stop_legs: List[PartialStopLeg] = Field(
        default_factory=lambda: [
            PartialStopLeg(ratio=0.5, stop_loss_pct=2.0),
            PartialStopLeg(ratio=0.5, stop_loss_pct=4.0),
        ],
        description="多头分批止损腿配置",
    )
    short_stop_loss_pct: float = Field(
        3.0, ge=0.1, description="空头固定止损百分比"
    )
    short_trailing_trigger_pct: float = Field(
        10.0, ge=0.0, description="空头移动止盈触发百分比"
    )
    short_trailing_offset_pct: float = Field(
        30.0, ge=0.0, description="空头盈利回调幅度"
    )
    short_profit_loss_ratio: float = Field(
        1.5, gt=0.0, description="空头固定盈亏比"
    )
    short_partial_trailing_legs: List[PartialTrailingLeg] = Field(
        default_factory=lambda: [
            PartialTrailingLeg(ratio=0.5, trigger_pct=5.0, offset_pct=40.0),
            PartialTrailingLeg(ratio=0.5, trigger_pct=10.0, offset_pct=30.0),
        ],
        description="空头分批移动止盈腿配置",
    )
    short_partial_stop_legs: List[PartialStopLeg] = Field(
        default_factory=lambda: [
            PartialStopLeg(ratio=0.5, stop_loss_pct=2.0),
            PartialStopLeg(ratio=0.5, stop_loss_pct=4.0),
        ],
        description="空头分批止损腿配置",
    )
    atr_period: int = Field(14, ge=1, description="ATR 计算周期（风控）")
    atr_multiplier: float = Field(
        1.5, gt=0.0, description="ATR 模式中止盈止损的倍数"
    )

    # Backtest settings
    initial_capital: float = Field(1_000.0, gt=0.0, description="初始资金")
    commission: float = Field(0.0, ge=0.0, description="单笔佣金（固定）")
    slippage: float = Field(0.0, ge=0.0, description="滑点（按价格）")
    commission_rate: float = Field(0.0, ge=0.0, description="佣金率（占成交金额百分比）")
    commission_min: float = Field(0.0, ge=0.0, description="每笔最低佣金")
    platform_fee_fixed_per_share: float = Field(
        0.005, ge=0.0, description="固定平台费（每股）"
    )
    platform_fee_tiers: List[Tuple[int, Optional[int], float]] = Field(
        default_factory=lambda: [
            (1, 5000, 0.0070),
            (5001, 10000, 0.0060),
            (10001, 100000, 0.0050),
            (100001, 1_000_000, 0.0040),
            (1_000_001, None, 0.0030),
        ],
        description="分段平台费：成交股数下限、上限（可选）以及每股费用",
    )
    stock_transfer_in_fee: float = Field(
        0.0, ge=0.0, description="股票转入费用（单笔）"
    )

    # Output artifacts
    artifacts_dir: Path = Field(Path("artifacts"), description="输出目录")
    trades_csv: Path = Field(Path("artifacts/lorentz_knn_trades.csv"))
    chart_path: Path = Field(Path("artifacts/lorentz_knn.png"))
    # 追加：本地数据开关与路径
    use_local_data: bool = Field(False, description="是否优先使用本地CSV数据")
    local_data_csv: Path = Field(Path("artifacts/data.csv"), description="本地K线数据CSV路径")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @validator("interval")
    def validate_interval(cls, value: str) -> str:
        allowed = {"1m", "5m", "15m", "30m", "60m", "1d"}
        if value not in allowed:
            raise ValueError(f"interval 必须是 {sorted(allowed)} 之一")
        return value

    @validator("feature_count")
    def validate_feature_count(cls, value: int, values) -> int:
        features = values.get("features") or []
        if features and value > len(features):
            raise ValueError("feature_count 不能大于 features 列表长度")
        return value

    @validator(
        "long_partial_trailing_legs",
        "short_partial_trailing_legs",
        "long_partial_stop_legs",
        "short_partial_stop_legs",
    )
    def validate_partial_leg_ratios(cls, value: List[PartialTrailingLeg | PartialStopLeg]) -> List[PartialTrailingLeg | PartialStopLeg]:
        total_ratio = sum(getattr(leg, "ratio", 0.0) for leg in value)
        if total_ratio > 1.0 + 1e-6:
            raise ValueError("分批腿的 ratio 总和不能超过 1")
        return value

    @validator("artifacts_dir", "trades_csv", "chart_path", "local_data_csv", pre=True)
    def expand_paths(cls, value: Path | str) -> Path:
        return Path(value).expanduser()
