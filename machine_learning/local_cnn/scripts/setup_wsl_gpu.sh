#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/mnt/d/Projects/tinyml_home_weather_forecasting/machine_learning/local_cnn"
POETRY_BIN="${HOME}/.local/bin/poetry"

cd "$PROJECT_DIR"
export POETRY_VIRTUALENVS_IN_PROJECT=false

"$POETRY_BIN" env use python3.12
"$POETRY_BIN" run python -m pip install --upgrade pip
"$POETRY_BIN" install --without dev --no-root
"$POETRY_BIN" run python -m pip install "tensorflow[and-cuda]==2.20.*" "keras==3.13.2" "numpy==2.1.3"
"$POETRY_BIN" run python -c 'import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices("GPU"))'
