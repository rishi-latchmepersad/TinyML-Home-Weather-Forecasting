from __future__ import annotations

import time

import numpy as np
import tensorflow as tf


def main() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow version: {tf.__version__}", flush=True)
    print(f"Visible GPUs: {gpus}", flush=True)
    if not gpus:
        raise SystemExit("No GPU detected by TensorFlow.")

    with tf.device("/GPU:0"):
        a = tf.random.normal((4096, 4096))
        b = tf.random.normal((4096, 4096))
        start = time.perf_counter()
        for _ in range(20):
            c = tf.matmul(a, b)
        _ = c.numpy()
        elapsed = time.perf_counter() - start
        print(f"GPU matmul elapsed_s: {elapsed:.2f}", flush=True)

    inputs = tf.random.normal((4096, 48, 9))
    targets = tf.random.normal((4096, 24))
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(48, 9)),
            tf.keras.layers.Conv1D(32, 5, activation="relu", padding="same"),
            tf.keras.layers.SeparableConv1D(64, 3, activation="relu", padding="same"),
            tf.keras.layers.Conv1D(16, 1, activation="relu", padding="same"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(24),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    history = model.fit(inputs, targets, epochs=3, batch_size=128, verbose=2)
    print(f"Final smoke loss: {history.history['loss'][-1]:.4f}", flush=True)


if __name__ == "__main__":
    main()
