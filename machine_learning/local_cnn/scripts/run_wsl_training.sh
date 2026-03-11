#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/mnt/d/Projects/tinyml_home_weather_forecasting/machine_learning/local_cnn"
POETRY_BIN="${HOME}/.local/bin/poetry"

cd "$PROJECT_DIR"
export POETRY_VIRTUALENVS_IN_PROJECT=false
export PYTHONPATH="${PROJECT_DIR}/src"

"$POETRY_BIN" run python -u -m local_cnn.train --device gpu --fast --skip-open-meteo --log-level INFO
