#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/mnt/d/Projects/tinyml_home_weather_forecasting/machine_learning/local_cnn"
POETRY_BIN="${HOME}/.local/bin/poetry"

cd "$PROJECT_DIR"
export POETRY_VIRTUALENVS_IN_PROJECT=false

"$POETRY_BIN" env info --path
"$POETRY_BIN" run python -m pip show tensorflow || true
"$POETRY_BIN" run python -m pip show nvidia-cudnn-cu12 || true
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
