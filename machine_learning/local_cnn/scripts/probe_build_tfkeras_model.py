from __future__ import annotations

import os

import tensorflow as tf


def main() -> int:
    print("probe_start", flush=True)
    inputs = tf.keras.Input(shape=(48, 9), name="input_window")
    print("input_built", flush=True)
    x = tf.keras.layers.Flatten(name="flatten")(inputs)
    print("flatten_built", flush=True)
    x = tf.keras.layers.Dense(64, activation="relu", name="dense1")(x)
    print("dense1_built", flush=True)
    x = tf.keras.layers.Dense(32, activation="relu", name="dense2")(x)
    print("dense2_built", flush=True)
    outputs = tf.keras.layers.Dense(24, name="forecast")(x)
    print("dense3_built", flush=True)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="probe_tfkeras_model")
    print("model_built", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
