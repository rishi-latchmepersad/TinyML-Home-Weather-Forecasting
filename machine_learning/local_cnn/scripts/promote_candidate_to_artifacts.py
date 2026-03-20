from __future__ import annotations

import argparse
import json
from pathlib import Path

from local_cnn.calibration import compute_horizon_bias_calibration, save_horizon_bias_calibration
from local_cnn.config import DEFAULT_MEASUREMENTS_DIRECTORY, DEFAULT_OUTPUT_DIRECTORY, PipelineConfig
from local_cnn.data import build_weather_dataframe
from local_cnn.features import prepare_dataset
from local_cnn.preflash_evaluation import _load_keras_model, _load_saved_scaler
from local_cnn.quantization import export_quantized_models
from local_cnn.modeling import build_regression_loss, horizon_360m_mae
import keras


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Promote a passing search candidate into the main artifacts directory.")
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--measurements-dir", type=Path, default=DEFAULT_MEASUREMENTS_DIRECTORY)
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidate_training_summary_path = args.candidate_dir / "training_summary.json"
    candidate_summary = json.loads(candidate_training_summary_path.read_text(encoding="utf-8"))
    feature_columns = tuple(candidate_summary["feature_columns"])
    recent_stage = candidate_summary["stages"]["recent_local_adaptation"]
    measurement_recent_days = int(recent_stage["measurement_recent_days"])

    scaler_path = args.candidate_dir / "feature_scaler.json"
    scaler = _load_saved_scaler(scaler_path, feature_columns)

    config = PipelineConfig(
        measurements_directory=args.measurements_dir,
        output_directory=args.output_dir,
        include_open_meteo=False,
        strict_open_meteo=False,
        skip_pruning=True,
        skip_quantization=False,
        measurement_recent_days=measurement_recent_days,
        selected_feature_columns=feature_columns,
        enable_op_determinism=False,
        device_preference="gpu",
        gpu_memory_growth=True,
    )

    weather_dataframe = build_weather_dataframe(config)
    dataset = prepare_dataset(weather_dataframe, config, existing_scaler=scaler)

    model_path = args.candidate_dir / "best_pruned_model.keras"
    model = _load_keras_model(model_path)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=build_regression_loss(forecast_horizon_slots=config.forecast_horizon_slots),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae"), horizon_360m_mae],
    )
    output_model_path = args.output_dir / "best_pruned_model.keras"
    model.save(output_model_path)

    output_scaler_path = args.output_dir / "feature_scaler.json"
    output_scaler_path.write_text(scaler_path.read_text(encoding="utf-8"), encoding="utf-8")

    quantization = export_quantized_models(
        keras_model=model,
        dataset=dataset,
        output_directory=args.output_dir,
        calibration_max_samples=config.calibration_max_samples,
    )

    bias_calibration = compute_horizon_bias_calibration(model, dataset.X_validate, dataset.y_validate)
    bias_calibration_path = save_horizon_bias_calibration(args.output_dir / "horizon_bias_calibration.json", bias_calibration)

    summary = {
        "feature_columns": list(feature_columns),
        "dataset_shapes": {
            "X_train": list(dataset.X_train.shape),
            "X_validate": list(dataset.X_validate.shape),
            "X_test": list(dataset.X_test.shape),
        },
        "config": {
            "measurement_recent_days": measurement_recent_days,
            "include_open_meteo": False,
            "selected_feature_columns": list(feature_columns),
            "resample_frequency": config.resample_frequency,
            "historical_window_slots": config.historical_window_slots,
            "forecast_horizon_slots": config.forecast_horizon_slots,
            "promotion_source": str(args.candidate_dir),
        },
        "metrics": {
            "recent_validation_mae": float(recent_stage["validate_metrics"]["mae"]),
            "recent_validation_360m_mae": float(recent_stage["validate_metrics"]["horizon_360m_mae"]),
            "recent_test_mae": float(recent_stage["test_metrics"]["mae"]),
            "recent_test_360m_mae": float(recent_stage["test_metrics"]["horizon_360m_mae"]),
            "keras_val_mae": float(quantization.keras_val_mae),
            "keras_test_mae": float(quantization.keras_test_mae),
            "tflite_fp32_val_mae": float(quantization.tflite_fp32_val_mae) if quantization.tflite_fp32_val_mae is not None else None,
            "tflite_fp32_test_mae": float(quantization.tflite_fp32_test_mae) if quantization.tflite_fp32_test_mae is not None else None,
            "tflite_int8_val_mae": float(quantization.tflite_int8_val_mae) if quantization.tflite_int8_val_mae is not None else None,
            "tflite_int8_test_mae": float(quantization.tflite_int8_test_mae) if quantization.tflite_int8_test_mae is not None else None,
        },
        "footprint": {
            "keras_model_params": int(model.count_params()),
            "int8_tflite_size_bytes": int(quantization.int8_size_bytes) if quantization.int8_size_bytes is not None else None,
            "int8_weight_bytes": int(quantization.int8_weight_bytes) if quantization.int8_weight_bytes is not None else None,
            "macs": int(quantization.macs),
        },
        "artifacts": {
            "best_pruned_model": str(output_model_path),
            "feature_scaler": str(output_scaler_path),
            "pruned_float32_tflite": str(quantization.float32_tflite_path) if quantization.float32_tflite_path is not None else None,
            "pruned_int8_tflite": str(quantization.int8_tflite_path) if quantization.int8_tflite_path is not None else None,
            "horizon_bias_calibration": str(bias_calibration_path),
            "source_candidate_training_summary": str(candidate_training_summary_path),
            "source_candidate_preflash_summary": str(args.candidate_dir / "preflash" / "preflash_evaluation_summary.json"),
        },
    }
    summary_path = args.output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    promotion_summary_path = args.output_dir / "promotion_summary.json"
    promotion_summary_path.write_text(
        json.dumps(
            {
                "candidate_dir": str(args.candidate_dir),
                "output_dir": str(args.output_dir),
                "training_summary": str(summary_path),
                "float32_tflite": str(quantization.float32_tflite_path) if quantization.float32_tflite_path is not None else None,
                "int8_tflite": str(quantization.int8_tflite_path) if quantization.int8_tflite_path is not None else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
