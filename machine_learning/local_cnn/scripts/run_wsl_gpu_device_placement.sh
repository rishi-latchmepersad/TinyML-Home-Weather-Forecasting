#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/mnt/d/Projects/tinyml_home_weather_forecasting/machine_learning/local_cnn"
POETRY_BIN="${HOME}/.local/bin/poetry"

cd "$PROJECT_DIR"
export POETRY_VIRTUALENVS_IN_PROJECT=false

"$POETRY_BIN" run python /mnt/d/Projects/tinyml_home_weather_forecasting/machine_learning/local_cnn/scripts/gpu_device_placement.py
