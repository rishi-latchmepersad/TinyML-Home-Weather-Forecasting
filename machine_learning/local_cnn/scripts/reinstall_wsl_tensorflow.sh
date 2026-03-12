#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/mnt/d/Projects/tinyml_home_weather_forecasting/machine_learning/local_cnn"
POETRY_BIN="${HOME}/.local/bin/poetry"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

cd "$PROJECT_DIR"
export POETRY_VIRTUALENVS_IN_PROJECT=false

if ENV_PATH="$("$POETRY_BIN" env info --path 2>/dev/null)"; then
  log "Removing existing Poetry environment: ${ENV_PATH}"
  "$POETRY_BIN" env remove --all
else
  log "No existing Poetry environment found"
fi

log "Creating fresh Poetry environment"
"$POETRY_BIN" env use python3.12
ENV_PATH="$("$POETRY_BIN" env info --path)"
log "Using Poetry environment: ${ENV_PATH}"

log "Refreshing Poetry lock file"
"$POETRY_BIN" lock

log "Upgrading pip tooling"
"${ENV_PATH}/bin/python" -m pip install --upgrade pip setuptools wheel

log "Installing Poetry-managed dependencies"
"$POETRY_BIN" install --without dev

log "Force reinstalling TensorFlow 2.20 with CUDA extras"
"${ENV_PATH}/bin/pip" install --upgrade --force-reinstall --no-cache-dir "tensorflow[and-cuda]==2.20.*" "keras==3.13.2" "numpy==2.1.3"

log "Verifying TensorFlow and GPU visibility"
"${ENV_PATH}/bin/python" -c 'import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices("GPU"))'
