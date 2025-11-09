#!/usr/bin/env bash

# 运行 stock_deepseek_ok_plus 回测。需要提前在当前目录配置 .venv
# 并设置 LongPort / DeepSeek 相关环境变量。

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "未找到 .venv，请先创建并安装依赖：python3.12 -m venv .venv && source .venv/bin/activate && pip install -e ."
  exit 1
fi

source .venv/bin/activate

rm -f artifacts/deepseek_ok_plus_cache.json

python scripts/run_backtest.py \
  --strategy stock_deepseek_ok_plus \
  --symbol SOXL.US \
  --interval 5m \
  --start 2025-11-06T00:00:00 \
  --end 2025-11-08T00:00:00 \
  --strategy-param analysis_window=60 \
  --strategy-param deepseek_cache=artifacts/deepseek_ok_plus_cache.json \
  --enable-short \
  --strategy-param allow_short=1 \

