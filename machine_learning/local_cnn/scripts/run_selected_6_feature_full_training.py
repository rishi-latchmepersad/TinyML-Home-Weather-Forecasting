from __future__ import annotations

from pathlib import Path

from local_cnn.config import DEFAULT_MEASUREMENTS_DIRECTORY, PipelineConfig
from local_cnn.pipeline import run_pipeline
from local_cnn.runtime import configure_tensorflow_runtime


SELECTED_6_FEATURES = (
    "temperature",
    "humidity",
    "pressure",
    "illuminance_lux",
    "delta_T",
    "pressure_delta",
)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    output_directory = project_root / "artifacts"

    config = PipelineConfig(
        measurements_directory=DEFAULT_MEASUREMENTS_DIRECTORY,
        output_directory=output_directory,
        include_open_meteo=True,
        skip_pruning=False,
        skip_quantization=False,
        device_preference="gpu",
        gpu_memory_growth=True,
        selected_feature_columns=SELECTED_6_FEATURES,
    )
    configure_tensorflow_runtime(config)
    run_pipeline(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
