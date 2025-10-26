## Supertrend 量化策略设计

### 目标
- 通过 LongPort OpenAPI 获取 `SOXL.US` 的 15 分钟历史 K 线。
- 计算超级趋势（Supertrend）指标并生成交易信号。
- 运行矢量化回测，输出交易统计数据和逐笔交易记录。
- 可视化价格、超级趋势上下轨以及交易点。
- 预留下单相关封装（默认不触发真实下单）。

### 总体流程
1. `scripts/run_backtest.py` 负责整体流程：
   1. 从环境变量构建 API 上下文；
   2. 拉取历史数据；
   3. 计算指标与信号；
   4. 执行回测；
   5. 生成图表并写入交易日志。
2. `supertrend/longport_client.py` 使用官方 QuoteContext 获取历史 K 线并管理上下文。
3. `supertrend/data.py` 将 LongPort K 线转换为整洁的 pandas 数据框。
4. `supertrend/indicators.py` 依托 pandas-ta 生成超级趋势指标并整理信号列。
5. `supertrend/backtest.py` 执行策略回测并返回绩效与交易列表。
6. `supertrend/plotting.py` 生成 matplotlib 图表用于收益回顾。

### 依赖
- `longport-openapi` Python SDK。
- `pandas`、`numpy` 用于数据处理。
- `matplotlib` 用于静态图表。
- `pydantic` 负责配置校验。
- `pandas-ta` 负责超级趋势等技术指标计算。
- 长桥自有 SDK `longport` 提供数据与交易接口。

### 交易规则
- 使用 15 分钟 K 线。
- 当价格向上突破超级趋势下轨且趋势方向为多头时建立多头。
- 当超级趋势翻转为空头（价格收盘跌破上轨）时平仓或反手。
- 同一时间仅允许一个仓位（空仓或多头），默认不开空。
- 手续费与滑点可配置（默认 0）。

### 回测输出
- 净值曲线、回撤、胜率、盈利因子、交易笔数等指标。
- 逐笔交易日志，包含时间、开平仓价、持仓周期、盈亏等。
- 保存价格与超级趋势图（含交易点）到 `artifacts/`。

### 后续工作
- 按上述结构实现各模块。
- 提供配置文件或命令行参数用于灵活控制。
- 编写 README，说明环境配置、回测运行及向实盘扩展的思路。
