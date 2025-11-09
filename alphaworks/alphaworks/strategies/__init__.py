"""策略加载与注册工具。"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Type

from .base import Strategy

_BUILTIN_REGISTRY: Dict[str, str] = {
    "supertrend": "alphaworks.strategies.supertrend.SupertrendStrategy",
    "turtle": "alphaworks.strategies.turtle.TurtleStrategy",
    "stock_deepseek_ai": "alphaworks.strategies.ai.stock_deepseek_strategy.StockDeepseekStrategy",
    "stock_deepseek_ok_plus": "alphaworks.strategies.ai.stock_deepseek_indicators_plus.StockDeepseekOkPlusStrategy",
}


def resolve_strategy_path(identifier: str) -> str:
    """将策略别名转换为可导入的全路径。"""
    if not identifier:
        raise ValueError("策略标识不能为空。")
    return _BUILTIN_REGISTRY.get(identifier, identifier)


def load_strategy_class(identifier: str) -> Type[Strategy]:
    """根据标识加载策略类型。"""
    dotted_path = resolve_strategy_path(identifier)
    if "." not in dotted_path:
        raise ValueError(
            f"策略 '{identifier}' 解析后为 '{dotted_path}'，但缺少类路径，请检查配置。"
        )
    module_name, class_name = dotted_path.rsplit(".", 1)
    module = import_module(module_name)
    try:
        cls = getattr(module, class_name)
    except AttributeError as exc:
        raise ImportError(f"无法从 {module_name} 导入策略类 {class_name}") from exc
    if not issubclass(cls, Strategy):
        raise TypeError(f"{cls} 不是 Strategy 的子类。")
    return cls


def create_strategy(
    identifier: str,
    *,
    settings: Any,
    params: Dict[str, Any] | None = None,
) -> Strategy:
    """实例化策略对象。"""
    strategy_cls = load_strategy_class(identifier)
    extra = params or {}
    if "settings" in extra:
        raise ValueError("策略参数中不能包含 'settings' 键。")
    return strategy_cls(settings=settings, **extra)


__all__ = [
    "Strategy",
    "create_strategy",
    "load_strategy_class",
    "resolve_strategy_path",
]
