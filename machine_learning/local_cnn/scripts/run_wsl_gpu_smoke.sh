#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/mnt/d/Projects/tinyml_home_weather_forecasting/machine_learning/local_cnn"
POETRY_BIN="${HOME}/.local/bin/poetry"

cd "$PROJECT_DIR"
export POETRY_VIRTUALENVS_IN_PROJECT=false
export PYTHONPATH="${PROJECT_DIR}/src"

"$POETRY_BIN" run python /mnt/d/Projects/tinyml_home_weather_forecasting/machine_learning/local_cnn/scripts/gpu_smoke_test.py &
TEST_PID=$!

for _ in $(seq 1 8); do
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader || true
  sleep 2
done

wait "$TEST_PID"
