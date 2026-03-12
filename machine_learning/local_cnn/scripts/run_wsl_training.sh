#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/mnt/d/Projects/tinyml_home_weather_forecasting/machine_learning/local_cnn"
POETRY_BIN="${HOME}/.local/bin/poetry"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

log "Entering project directory"
cd "$PROJECT_DIR"
export POETRY_VIRTUALENVS_IN_PROJECT=false
export PYTHONPATH="${PROJECT_DIR}/src"
export PYTHONUNBUFFERED=1

log "Resolving Poetry environment"
POETRY_ENV_PATH="$("$POETRY_BIN" env info --path)"
log "Using Poetry environment: ${POETRY_ENV_PATH}"

log "Starting local_cnn.train"
exec "${POETRY_ENV_PATH}/bin/python" -u -m local_cnn.train --device gpu --fast --skip-open-meteo --log-level INFO
