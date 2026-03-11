from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .config import PipelineConfig
from .data import build_weather_dataframe
from .features import PreparedDataset, plot_weather_features, prepare_dataset
from .modeling import (
    TrainingResult,
    estimate_conv1d_macs,
    estimate_model_size_bytes,
    fine_tune_model,
    seed_everything,
    tune_model,
)
from .pruning import PruningResult, run_pruning_sweep, summarize_unpruned_model
from .quantization import (
    QuantizationResult,
    export_quantized_models,
    summarize_keras_model_without_tflite,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PipelineArtifacts:
    dataset: PreparedDataset
    training: TrainingResult
    pruning: PruningResult
    quantization: QuantizationResult
    keras_model_path: Path
    scaler_path: Path
    summary_path: Path


def _save_scaler(dataset: PreparedDataset, output_path: Path) -> Path:
    payload = {
        "feature_columns": dataset.feature_columns,
        "mean": dataset.scaler.mean_.tolist(),
        "scale": dataset.scaler.scale_.tolist(),
        "var": dataset.scaler.var_.tolist(),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _save_summary(
    config: PipelineConfig,
    dataset: PreparedDataset,
    training: TrainingResult,
    pruning: PruningResult,
    quantization: QuantizationResult,
    output_path: Path,
) -> Path:
    summary = {
        "feature_columns": dataset.feature_columns,
        "dataset_shapes": {
            "X_train": list(dataset.X_train.shape),
            "y_train": list(dataset.y_train.shape),
            "X_validate": list(dataset.X_validate.shape),
            "y_validate": list(dataset.y_validate.shape),
            "X_test": list(dataset.X_test.shape),
            "y_test": list(dataset.y_test.shape),
        },
        "config": {
            "historical_window_slots": config.historical_window_slots,
            "forecast_horizon_slots": config.forecast_horizon_slots,
            "resample_frequency": config.resample_frequency,
            "include_open_meteo": config.include_open_meteo,
            "open_meteo_start_date": config.open_meteo_start_date,
            "open_meteo_end_date": config.open_meteo_end_date,
            "skip_quantization": config.skip_quantization,
            "skip_pruning": config.skip_pruning,
            "device_preference": config.device_preference,
            "gpu_memory_growth": config.gpu_memory_growth,
            "enable_op_determinism": config.enable_op_determinism,
        },
        "metrics": {
            "tuned_validation_mae": training.tuned_validation_mae,
            "fine_tuned_validation_mae": training.fine_tuned_validation_mae,
            "pruned_validation_mae": pruning.best_validation_mae,
            "pruned_test_mae": pruning.final_test_mae,
            "keras_val_mae": quantization.keras_val_mae,
            "tflite_fp32_val_mae": quantization.tflite_fp32_val_mae,
            "tflite_int8_val_mae": quantization.tflite_int8_val_mae,
            "keras_test_mae": quantization.keras_test_mae,
            "tflite_fp32_test_mae": quantization.tflite_fp32_test_mae,
            "tflite_int8_test_mae": quantization.tflite_int8_test_mae,
        },
        "footprint": {
            "fine_tuned_model_params": training.fine_tuned_model.count_params(),
            "fine_tuned_model_size_bytes": estimate_model_size_bytes(training.fine_tuned_model),
            "fine_tuned_model_macs": estimate_conv1d_macs(training.fine_tuned_model),
            "pruned_model_params": pruning.best_pruned_model.count_params(),
            "pruned_model_size_bytes": pruning.best_model_size_bytes,
            "pruned_model_macs": pruning.pruned_macs,
            "int8_tflite_size_bytes": quantization.int8_size_bytes,
            "int8_weight_bytes": quantization.int8_weight_bytes,
        },
        "pruning": {
            "keep_map": pruning.best_keep_map,
            "score": pruning.pruning_score,
        },
        "artifacts": {
            "float32_tflite": str(quantization.float32_tflite_path)
            if quantization.float32_tflite_path is not None
            else None,
            "int8_tflite": str(quantization.int8_tflite_path)
            if quantization.int8_tflite_path is not None
            else None,
        },
    }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return output_path


def run_pipeline(config: PipelineConfig) -> PipelineArtifacts:
    seed_everything(config.random_seed, enable_op_determinism=config.enable_op_determinism)
    config.ensure_directories()

    weather_dataframe = build_weather_dataframe(config)
    if config.plot_path is not None:
        plot_weather_features(weather_dataframe, config.plot_path)
        logger.info("Saved weather plot to %s", config.plot_path)

    dataset = prepare_dataset(weather_dataframe, config)
    logger.info(
        "Prepared dataset: train=%s validate=%s test=%s",
        dataset.X_train.shape,
        dataset.X_validate.shape,
        dataset.X_test.shape,
    )

    tuned_model, tuned_validation_mae = tune_model(dataset, config)
    fine_tuned_model, fine_tuned_validation_mae = fine_tune_model(tuned_model, dataset, config)
    training = TrainingResult(
        tuned_model=tuned_model,
        tuned_validation_mae=tuned_validation_mae,
        fine_tuned_model=fine_tuned_model,
        fine_tuned_validation_mae=fine_tuned_validation_mae,
    )

    if config.skip_pruning:
        pruning = summarize_unpruned_model(fine_tuned_model, dataset)
    else:
        pruning = run_pruning_sweep(fine_tuned_model, dataset, config)
    keras_model_path = config.output_directory / "best_pruned_model.keras"
    pruning.best_pruned_model.save(keras_model_path)
    scaler_path = _save_scaler(dataset, config.output_directory / "feature_scaler.json")

    if config.skip_quantization:
        quantization = summarize_keras_model_without_tflite(
            keras_model=pruning.best_pruned_model,
            dataset=dataset,
        )
    else:
        quantization = export_quantized_models(
            keras_model=pruning.best_pruned_model,
            dataset=dataset,
            output_directory=config.output_directory,
            calibration_max_samples=config.calibration_max_samples,
        )

    summary_path = _save_summary(
        config=config,
        dataset=dataset,
        training=training,
        pruning=pruning,
        quantization=quantization,
        output_path=config.output_directory / "training_summary.json",
    )
    return PipelineArtifacts(
        dataset=dataset,
        training=training,
        pruning=pruning,
        quantization=quantization,
        keras_model_path=keras_model_path,
        scaler_path=scaler_path,
        summary_path=summary_path,
    )
