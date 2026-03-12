#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${HOME}/tmp-tf-gpu-test"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1"
}

log "Removing old clean test environment if present"
rm -rf "${VENV_DIR}"

log "Creating clean virtual environment"
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

log "Upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

log "Installing TensorFlow 2.20 with CUDA extras"
pip install "tensorflow[and-cuda]==2.20.*"

log "Probing TensorFlow GPU enumeration"
python -u -c 'import tensorflow as tf; print(tf.__version__, flush=True); print(tf.config.list_physical_devices("GPU"), flush=True)'
