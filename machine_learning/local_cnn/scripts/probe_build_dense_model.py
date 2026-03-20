from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras


def main() -> int:
    print("probe_start", flush=True)
    inputs = keras.Input(shape=(48, 9), name="input_window")
    print("input_built", flush=True)
    x = keras.layers.Flatten(name="flatten")(inputs)
    print("flatten_built", flush=True)
    x = keras.layers.Dense(64, activation="relu", name="dense1")(x)
    print("dense1_built", flush=True)
    x = keras.layers.Dense(32, activation="relu", name="dense2")(x)
    print("dense2_built", flush=True)
    outputs = keras.layers.Dense(24, name="forecast")(x)
    print("dense3_built", flush=True)
    model = keras.Model(inputs=inputs, outputs=outputs, name="probe_dense_model")
    print("model_built", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
