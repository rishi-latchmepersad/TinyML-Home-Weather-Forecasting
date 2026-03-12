from __future__ import annotations

print("before local_cnn imports", flush=True)

from local_cnn.config import PipelineConfig
from local_cnn.runtime import configure_tensorflow_runtime

print("after local_cnn imports", flush=True)

configure_tensorflow_runtime(PipelineConfig(device_preference="gpu"))

print("after runtime configuration", flush=True)
