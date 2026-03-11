from __future__ import annotations

import tensorflow as tf


def main() -> None:
    tf.debugging.set_log_device_placement(True)
    gpus = tf.config.list_physical_devices("GPU")
    print(f"Visible GPUs: {gpus}", flush=True)
    if not gpus:
        raise SystemExit("No GPU detected by TensorFlow.")

    with tf.device("/GPU:0"):
        a = tf.random.normal((1024, 1024))
        b = tf.random.normal((1024, 1024))
        c = tf.matmul(a, b)
        _ = c.numpy()
    print(f"Matmul tensor device: {c.device}", flush=True)


if __name__ == "__main__":
    main()
