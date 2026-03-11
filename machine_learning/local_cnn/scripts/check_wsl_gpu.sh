#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/mnt/d/Projects/tinyml_home_weather_forecasting/machine_learning/local_cnn"
POETRY_BIN="${HOME}/.local/bin/poetry"

cd "$PROJECT_DIR"
export POETRY_VIRTUALENVS_IN_PROJECT=false

"$POETRY_BIN" run python -c 'import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices("GPU"))'
"$POETRY_BIN" run python -c 'code = """
import tensorflow as tf
gpus = tf.config.list_physical_devices(\"GPU\")
print(\"visible_gpus\", gpus, flush=True)
assert gpus, \"No GPU detected by TensorFlow.\"
with tf.device(\"/GPU:0\"):
    a = tf.random.normal((512, 512))
    b = tf.random.normal((512, 512))
    c = tf.matmul(a, b)
    _ = c.numpy()
print(\"matmul_device\", c.device, flush=True)
"""; exec(code)'
