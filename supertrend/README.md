## 项目概览

本项目实现了基于超级趋势（Supertrend）指标的 SOXL.US 15 分钟级别量化交易回测流程，数据来源为 [LongPort OpenAPI](https://github.com/longportapp/openapi/tree/master/python)。代码结构参考了 fastquant 的模块化设计，涵盖数据获取、指标计算、回测执行与图表输出。

## 环境准备

1. 建议使用 Python 3.12 版本，并创建虚拟环境：
   ```bash
   /opt/homebrew/bin/python3.12 -m venv .venv
   source .venv/bin/activate
   ```
2. 安装依赖：
   ```bash
   pip install -e .
   ```
3. 设置 LongPort 相关环境变量（以下变量已经在本地配置的可跳过）：
   ```bash
   export LONGPORT_APP_KEY=xxxx
   export LONGPORT_APP_SECRET=xxxx
   export LONGPORT_ACCESS_TOKEN=xxxx
   export LONGPORT_REGION=hk  # 如需使用其他市场可调整
   export LONGPORT_HTTP_URL=https://openapi.longportapp.com  # 可选，自定义 OpenAPI 域名
   # 如需自定义行情/交易 WebSocket，可设置 LONGPORT_QUOTE_WS_URL、LONGPORT_TRADE_WS_URL
   ```

## 回测运行

执行脚本 `scripts/run_backtest.py` 即可开启回测，默认参数为 SOXL.US 的 15 分钟数据，超级趋势周期 10，乘数 3：

```bash
python scripts/run_backtest.py \
  --symbol SOXL.US \
  --interval 15m \
  --start 2023-01-01T09:30:00 \
  --end 2023-03-01T16:00:00 \
  --period 10 \
  --multiplier 3.0 \
  --initial-capital 10000 \
  --commission 0.0 \
  --slippage 0.0
```

脚本运行后会输出主要回测指标，并在 `artifacts/` 目录生成：

- `trades.csv`：逐笔交易记录；
- `supertrend.png`：价格、超级趋势轨道与交易点可视化；
- 其余日志留待扩展。

## 模块说明

- `supertrend/config.py`：读取环境变量与运行参数。
- `supertrend/longport_client.py`：通过官方 QuoteContext 获取历史 K 线并管理上下文。
- `supertrend/data.py`：K 线数据转换为 pandas 数据框。
- `supertrend/indicators.py`：借助 pandas-ta 计算超级趋势并整理信号。
- `supertrend/backtest.py`：完成仓位管理、绩效指标与交易日志输出。
- `supertrend/plotting.py`：绘制价格与指标图表。
- `supertrend/pipeline.py`：整合上述模块，形成一条回测流水线。

## 后续拓展建议

1. 接入实时推送与下单逻辑，实现模拟或实盘执行。
2. 扩展策略参数优化，例如蒙特卡洛搜索或网格搜索。
3. 增加更多绩效分析图表，如回撤曲线、收益分布等。
4. 引入单元测试，覆盖数据转换与指标计算等关键模块。
