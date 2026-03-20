from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import tensorflow as tf


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / "artifacts" / "best_pruned_model.keras"
    print("start", flush=True)
    print(model_path, flush=True)
    print("tensorflow_imported", flush=True)
    model = tf.keras.models.load_model(model_path, compile=False)
    print(type(model).__name__, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
