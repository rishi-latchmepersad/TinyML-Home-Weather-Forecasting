from __future__ import annotations

from pathlib import Path

from local_cnn.config import DEFAULT_MEASUREMENTS_DIRECTORY, PipelineConfig
from local_cnn.pipeline import run_pipeline
from local_cnn.runtime import configure_tensorflow_runtime


LONG_HORIZON_FEATURES = (
    "temperature",
    "humidity",
    "pressure",
    "illuminance_lux",
    "delta_T",
    "pressure_delta",
    "temp_mean_24h",
    "sin_hour",
    "cos_hour",
)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    output_directory = project_root / "artifacts" / "long_horizon_residual_openmeteo"
    tuner_directory = project_root / "keras_tuner_dir" / "long_horizon_residual_openmeteo"

    config = PipelineConfig(
        measurements_directory=DEFAULT_MEASUREMENTS_DIRECTORY,
        output_directory=output_directory,
        tuner_directory=tuner_directory,
        include_open_meteo=True,
        skip_pruning=False,
        skip_quantization=False,
        device_preference="gpu",
        gpu_memory_growth=True,
        selected_feature_columns=LONG_HORIZON_FEATURES,
        tuner_max_trials=2,
        tuner_executions_per_trial=1,
        tuner_epochs=50,
        tuner_patience=8,
        fine_tune_epochs=24,
        pruning_epochs=18,
    )
    configure_tensorflow_runtime(config)
    run_pipeline(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
