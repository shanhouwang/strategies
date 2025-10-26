# Lorentzian KNN Strategy

该目录实现了 TradingView `some.pine` 指标的 Python 版本，包含特征工程、KNN 预测、过滤器以及多种风控模式。回测流程与 `supertrend/` 目录保持一致：加载数据 → 生成指标 → 执行回测 → 导出交易记录与图表。

## 主要组件

- `lorentz_knn/config.py`：通过 `pydantic` 解析环境变量和 CLI 覆盖项，统一使用 LongPort 拉取数据，并支持佣金与平台费配置。
- `lorentz_knn/indicator.py`：复刻指标逻辑，计算 Lorentzian 距离的 KNN 预测、趋势/波动过滤器以及核回归筛选。
- `lorentz_knn/backtest.py`：顺序撮合，支持做空开关、分批移动止盈/止损、移动止盈、固定止损、固定盈亏比和 ATR 风控。
- `lorentz_knn/pipeline.py`：串联数据加载、指标、回测和产物输出（交易日志与图表）。
- `scripts/run_backtest.py`：命令行入口，调用 Pipeline 并打印回测指标。

## 快速使用

1. 安装依赖（包含 `pandas-ta` 与 LongPort SDK）：

   ```bash
   pip install -e lorentz_knn
   ```

2. 若使用 LongPort 拉取数据，需要在环境变量或 `.env` 中配置：

   ```
   LONGPORT_APP_KEY=...
   LONGPORT_APP_SECRET=...
   LONGPORT_ACCESS_TOKEN=...
   ```

3. 运行 CLI 回测（默认使用 LongPort 数据源）：

   ```bash
   PYTHONPATH=lorentz_knn python3 -m lorentz_knn.scripts.run_backtest \
     --symbol SOXL.US \
     --interval 15m \
     --initial-capital 10000
   ```

   默认策略允许做空，并使用原 Pine 脚本的特征和参数。CLI 可通过选项覆盖邻居数量、风控开关等；若需要启用分批移动止盈/止损，可追加 `--partial-trailing` 与 `--partial-stop`。

## 输出

- `artifacts/lorentz_knn_trades.csv`：逐笔交易记录。
- `artifacts/lorentz_knn.png`：价格+信号示意图。
- 控制台打印交易统计（收益率、最大回撤、胜率、盈利因子等）。

> 注意：回测依赖 `pandas-ta` 以及 LongPort SDK。若策略需要更多 Pine 内嵌特征，可在 `features.py` 中补充。
