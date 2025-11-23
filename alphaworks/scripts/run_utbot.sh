#!/usr/bin/env bash

# 运行 Alpha UT 多空引擎（utbot_strategy）回测。需要提前创建 .venv 并安装依赖。

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -f ".venv/bin/activate" ]]; then
  echo "未找到 .venv，请先执行：python3 -m venv .venv && source .venv/bin/activate && pip install -e ."
  exit 1
fi

source .venv/bin/activate

python scripts/run_backtest.py \
  --strategy utbot \
  --symbol ${UTBOT_SYMBOL:-SOXL.US} \
  --interval ${UTBOT_INTERVAL:-15m} \
  --start ${UTBOT_START:-2025-02-03T00:00:00} \
  --end ${UTBOT_END:-2025-11-15T00:00:00} \
  --strategy-param stc_fast=${UTBOT_STC_FAST:-26} \
  --strategy-param stc_slow=${UTBOT_STC_SLOW:-50} \
  --strategy-param stc_smooth=${UTBOT_STC_SMOOTH:-0.5} \
  --strategy-param stc_bull_threshold=${UTBOT_STC_BULL:-30} \
  --strategy-param stc_bear_threshold=${UTBOT_STC_BEAR:-70} \
  --strategy-param use_stc_filter=${UTBOT_USE_STC_FILTER:-1} \
  --strategy-param use_qqe_filter=${UTBOT_USE_QQE_FILTER:-1} \
  --strategy-param filter_require_both=${UTBOT_FILTER_REQUIRE_BOTH:-0} \
  --strategy-param enable_short=${UTBOT_ENABLE_SHORT:-1} \
  --strategy-param atr_period=${UTBOT_ATR_PERIOD:-1} \
  --strategy-param atr_multiplier=${UTBOT_ATR_MULTIPLIER:-3} \
  --strategy-param qqe_threshold=${UTBOT_QQE_THRESHOLD:-3} \
  --strategy-param boll_length=${UTBOT_BOLL_LENGTH:-50} \
  --strategy-param boll_multiplier=${UTBOT_BOLL_MULTIPLIER:-0.35}
  
