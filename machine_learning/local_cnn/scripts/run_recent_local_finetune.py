from __future__ import annotations

import json
from pathlib import Path

import tensorflow as tf

from local_cnn.config import DEFAULT_MEASUREMENTS_DIRECTORY, PipelineConfig
from local_cnn.data import build_weather_dataframe
from local_cnn.features import prepare_dataset
from local_cnn.modeling import (
    HorizonBlend,
    HybridTemperatureAnchor,
    LastTimeStep,
    SeasonalNaiveTrajectory,
    WeightedForecastHuber,
    fine_tune_model,
    horizon_360m_mae,
    horizon_60m_mae,
    seed_everything,
)
from local_cnn.pipeline import _save_scaler
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
    base_model_path = (
        project_root / "artifacts" / "long_horizon_residual_openmeteo" / "best_pruned_model.keras"
    )
    output_directory = project_root / "artifacts" / "recent_local_finetune"
    output_directory.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig(
        measurements_directory=DEFAULT_MEASUREMENTS_DIRECTORY,
        output_directory=output_directory,
        tuner_directory=project_root / "keras_tuner_dir" / "recent_local_finetune_unused",
        include_open_meteo=False,
        skip_pruning=True,
        skip_quantization=True,
        device_preference="gpu",
        gpu_memory_growth=True,
        enable_op_determinism=False,
        measurement_recent_days=7,
        selected_feature_columns=LONG_HORIZON_FEATURES,
        fine_tune_epochs=18,
        fine_tune_batch_size=64,
        fine_tune_learning_rate=1e-4,
    )

    seed_everything(config.random_seed, enable_op_determinism=config.enable_op_determinism)
    configure_tensorflow_runtime(config)

    weather_dataframe = build_weather_dataframe(config)
    dataset = prepare_dataset(weather_dataframe, config)

    model = tf.keras.models.load_model(
        base_model_path,
        custom_objects={
            "WeightedForecastHuber": WeightedForecastHuber,
            "LastTimeStep": LastTimeStep,
            "HorizonBlend": HorizonBlend,
            "SeasonalNaiveTrajectory": SeasonalNaiveTrajectory,
            "HybridTemperatureAnchor": HybridTemperatureAnchor,
            "horizon_60m_mae": horizon_60m_mae,
            "horizon_360m_mae": horizon_360m_mae,
        },
    )

    fine_tuned_model, validation_mae = fine_tune_model(model, dataset, config)

    validate_metrics = fine_tuned_model.evaluate(
        dataset.X_validate,
        dataset.y_validate,
        verbose=0,
        return_dict=True,
    )
    test_metrics = fine_tuned_model.evaluate(
        dataset.X_test,
        dataset.y_test,
        verbose=0,
        return_dict=True,
    )

    model_path = output_directory / "best_recent_local_finetune.keras"
    fine_tuned_model.save(model_path)
    scaler_path = _save_scaler(dataset, output_directory / "feature_scaler.json")

    summary = {
        "base_model_path": str(base_model_path),
        "fine_tuned_model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "feature_columns": dataset.feature_columns,
        "config": {
            "measurement_recent_days": config.measurement_recent_days,
            "include_open_meteo": config.include_open_meteo,
            "fine_tune_epochs": config.fine_tune_epochs,
            "fine_tune_batch_size": config.fine_tune_batch_size,
            "fine_tune_learning_rate": config.fine_tune_learning_rate,
        },
        "dataset_shapes": {
            "X_train": list(dataset.X_train.shape),
            "X_validate": list(dataset.X_validate.shape),
            "X_test": list(dataset.X_test.shape),
        },
        "validation_mae": float(validation_mae),
        "validate_metrics": {key: float(value) for key, value in validate_metrics.items()},
        "test_metrics": {key: float(value) for key, value in test_metrics.items()},
    }
    summary_path = output_directory / "fine_tune_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
