from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras


def main() -> int:
    print("probe_start", flush=True)
    inputs = keras.Input(shape=(48, 9), name="input_window")
    print("input_built", flush=True)
    x = keras.layers.Conv1D(24, 5, padding="same", activation="relu", name="conv1")(inputs)
    print("conv1_built", flush=True)
    x = keras.layers.Conv1D(24, 3, padding="same", activation="relu", name="conv2")(x)
    print("conv2_built", flush=True)
    x = keras.layers.GlobalAveragePooling1D(name="gap")(x)
    print("gap_built", flush=True)
    outputs = keras.layers.Dense(24, name="forecast")(x)
    print("dense_built", flush=True)
    model = keras.Model(inputs=inputs, outputs=outputs, name="probe_model")
    print("model_built", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
