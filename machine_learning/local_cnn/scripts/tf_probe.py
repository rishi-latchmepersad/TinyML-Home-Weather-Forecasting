from __future__ import annotations

print("before import", flush=True)

import tensorflow as tf

print(tf.__version__, flush=True)
print("after import", flush=True)
print(tf.config.list_physical_devices("GPU"), flush=True)
