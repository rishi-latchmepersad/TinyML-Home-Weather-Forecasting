from __future__ import annotations

import json
import shutil
from pathlib import Path

from local_cnn.config import DEFAULT_MEASUREMENTS_DIRECTORY, PipelineConfig
from local_cnn.data import build_weather_dataframe
from local_cnn.features import prepare_dataset
from local_cnn.modeling import (
    TrainingResult,
    estimate_conv1d_macs,
    estimate_model_size_bytes,
    fine_tune_model,
    seed_everything,
    tune_model,
)
from local_cnn.pipeline import _save_scaler
from local_cnn.pruning import run_pruning_sweep
from local_cnn.quantization import export_quantized_models
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


def _copy_if_present(source_path: Path | None, destination_path: Path) -> None:
    if source_path is None or not source_path.exists():
        return
    shutil.copy2(source_path, destination_path)


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    final_output_directory = project_root / "artifacts"
    staging_directory = final_output_directory / "final_best_training_stage"
    full_tuner_directory = project_root / "keras_tuner_dir" / "final_best_full_training"
    final_output_directory.mkdir(parents=True, exist_ok=True)
    staging_directory.mkdir(parents=True, exist_ok=True)
    full_tuner_directory.mkdir(parents=True, exist_ok=True)

    feature_columns = LONG_HORIZON_FEATURES

    broad_config = PipelineConfig(
        measurements_directory=DEFAULT_MEASUREMENTS_DIRECTORY,
        output_directory=staging_directory / "broad_training",
        tuner_directory=full_tuner_directory,
        include_open_meteo=True,
        strict_open_meteo=False,
        skip_pruning=True,
        skip_quantization=True,
        device_preference="gpu",
        gpu_memory_growth=True,
        enable_op_determinism=False,
        selected_feature_columns=feature_columns,
        tuner_max_trials=5,
        tuner_executions_per_trial=1,
        tuner_epochs=70,
        tuner_batch_size=64,
        tuner_patience=12,
        fine_tune_epochs=28,
        fine_tune_batch_size=32,
        fine_tune_learning_rate=2e-4,
        random_seed=42,
    )

    recent_local_config = PipelineConfig(
        measurements_directory=DEFAULT_MEASUREMENTS_DIRECTORY,
        output_directory=staging_directory / "recent_local_adaptation",
        tuner_directory=project_root / "keras_tuner_dir" / "final_best_recent_unused",
        include_open_meteo=False,
        skip_pruning=True,
        skip_quantization=True,
        device_preference="gpu",
        gpu_memory_growth=True,
        enable_op_determinism=False,
        measurement_recent_days=14,
        selected_feature_columns=feature_columns,
        fine_tune_epochs=36,
        fine_tune_batch_size=64,
        fine_tune_learning_rate=8e-5,
        pruning_epochs=24,
        pruning_batch_size=32,
        pruning_learning_rate=2e-4,
        calibration_max_samples=2048,
        random_seed=42,
    )

    seed_everything(
        broad_config.random_seed,
        enable_op_determinism=broad_config.enable_op_determinism,
    )
    configure_tensorflow_runtime(broad_config)

    broad_weather_dataframe = build_weather_dataframe(broad_config)
    broad_dataset = prepare_dataset(broad_weather_dataframe, broad_config)
    tuned_model, tuned_validation_mae = tune_model(broad_dataset, broad_config)
    broad_fine_tuned_model, broad_fine_tuned_validation_mae = fine_tune_model(
        tuned_model,
        broad_dataset,
        broad_config,
    )
    broad_training = TrainingResult(
        tuned_model=tuned_model,
        tuned_validation_mae=tuned_validation_mae,
        fine_tuned_model=broad_fine_tuned_model,
        fine_tuned_validation_mae=broad_fine_tuned_validation_mae,
    )

    recent_weather_dataframe = build_weather_dataframe(recent_local_config)
    recent_dataset = prepare_dataset(recent_weather_dataframe, recent_local_config)
    recent_fine_tuned_model, recent_fine_tuned_validation_mae = fine_tune_model(
        broad_fine_tuned_model,
        recent_dataset,
        recent_local_config,
    )

    pruning = run_pruning_sweep(recent_fine_tuned_model, recent_dataset, recent_local_config)
    quantization = export_quantized_models(
        keras_model=pruning.best_pruned_model,
        dataset=recent_dataset,
        output_directory=staging_directory / "quantized_exports",
        calibration_max_samples=recent_local_config.calibration_max_samples,
    )

    final_model_path = final_output_directory / "best_pruned_model.keras"
    pruning.best_pruned_model.save(final_model_path)
    final_scaler_path = _save_scaler(recent_dataset, final_output_directory / "feature_scaler.json")
    _copy_if_present(quantization.float32_tflite_path, final_output_directory / "pruned_float32.tflite")
    _copy_if_present(quantization.int8_tflite_path, final_output_directory / "pruned_int8.tflite")

    summary = {
        "feature_columns": list(feature_columns),
        "stages": {
            "broad_training": {
                "include_open_meteo": broad_config.include_open_meteo,
                "dataset_shapes": {
                    "X_train": list(broad_dataset.X_train.shape),
                    "X_validate": list(broad_dataset.X_validate.shape),
                    "X_test": list(broad_dataset.X_test.shape),
                },
                "tuned_validation_mae": float(broad_training.tuned_validation_mae),
                "fine_tuned_validation_mae": float(broad_training.fine_tuned_validation_mae),
            },
            "recent_local_adaptation": {
                "include_open_meteo": recent_local_config.include_open_meteo,
                "measurement_recent_days": recent_local_config.measurement_recent_days,
                "dataset_shapes": {
                    "X_train": list(recent_dataset.X_train.shape),
                    "X_validate": list(recent_dataset.X_validate.shape),
                    "X_test": list(recent_dataset.X_test.shape),
                },
                "fine_tuned_validation_mae": float(recent_fine_tuned_validation_mae),
                "validate_metrics": {
                    key: float(value)
                    for key, value in recent_fine_tuned_model.evaluate(
                        recent_dataset.X_validate,
                        recent_dataset.y_validate,
                        verbose=0,
                        return_dict=True,
                    ).items()
                },
                "test_metrics": {
                    key: float(value)
                    for key, value in recent_fine_tuned_model.evaluate(
                        recent_dataset.X_test,
                        recent_dataset.y_test,
                        verbose=0,
                        return_dict=True,
                    ).items()
                },
            },
        },
        "pruning": {
            "keep_map": pruning.best_keep_map,
            "validation_mae": float(pruning.best_validation_mae),
            "test_mae": float(pruning.final_test_mae),
            "pruned_model_params": int(pruning.best_pruned_model.count_params()),
            "pruned_model_size_bytes": int(pruning.best_model_size_bytes),
            "pruned_model_macs": int(pruning.pruned_macs),
        },
        "quantization": {
            "keras_val_mae": float(quantization.keras_val_mae),
            "keras_test_mae": float(quantization.keras_test_mae),
            "tflite_fp32_val_mae": None
            if quantization.tflite_fp32_val_mae is None
            else float(quantization.tflite_fp32_val_mae),
            "tflite_fp32_test_mae": None
            if quantization.tflite_fp32_test_mae is None
            else float(quantization.tflite_fp32_test_mae),
            "tflite_int8_val_mae": None
            if quantization.tflite_int8_val_mae is None
            else float(quantization.tflite_int8_val_mae),
            "tflite_int8_test_mae": None
            if quantization.tflite_int8_test_mae is None
            else float(quantization.tflite_int8_test_mae),
            "float32_size_bytes": None
            if quantization.float32_size_bytes is None
            else int(quantization.float32_size_bytes),
            "int8_size_bytes": None
            if quantization.int8_size_bytes is None
            else int(quantization.int8_size_bytes),
            "int8_weight_bytes": None
            if quantization.int8_weight_bytes is None
            else int(quantization.int8_weight_bytes),
        },
        "artifacts": {
            "best_pruned_model": str(final_model_path),
            "feature_scaler": str(final_scaler_path),
            "pruned_float32_tflite": str(final_output_directory / "pruned_float32.tflite"),
            "pruned_int8_tflite": str(final_output_directory / "pruned_int8.tflite"),
            "staging_directory": str(staging_directory),
        },
        "footprint": {
            "recent_adapted_model_params": int(recent_fine_tuned_model.count_params()),
            "recent_adapted_model_size_bytes": int(estimate_model_size_bytes(recent_fine_tuned_model)),
            "recent_adapted_model_macs": int(estimate_conv1d_macs(recent_fine_tuned_model)),
        },
    }
    summary_path = final_output_directory / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
