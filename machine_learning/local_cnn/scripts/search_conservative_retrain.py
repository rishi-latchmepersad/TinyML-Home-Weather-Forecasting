from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import tensorflow as tf

from local_cnn.config import DEFAULT_MEASUREMENTS_DIRECTORY, PipelineConfig
from local_cnn.data import build_weather_dataframe
from local_cnn.features import prepare_dataset
from local_cnn.modeling import (
    TrainingResult,
    build_regression_loss,
    fine_tune_model,
    horizon_360m_mae,
    seed_everything,
)
from local_cnn.pipeline import _save_scaler
from local_cnn.search_layers import (
    HorizonBlend,
    HorizonTrustSchedule,
    LatestTemporalFeatures,
    SeasonalNaiveTrajectory,
)


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


def build_search_model(
    *,
    input_window_length: int,
    number_of_input_features: int,
    forecast_horizon_slots: int,
    temperature_feature_mean: float,
    temperature_feature_scale: float,
    temperature_feature_index: int,
    blend_long_weight: float = 1.0,
    trust_schedule_start: float = 0.95,
    trust_schedule_end: float = 0.25,
) -> keras.Model:
    input_layer = keras.Input(shape=(input_window_length, number_of_input_features), name="input_window")
    x = keras.layers.Conv1D(24, 5, padding="same", activation="relu", name="conv1_k5")(input_layer)
    x = keras.layers.Conv1D(24, 3, padding="same", activation="relu", name="conv2_k3")(x)
    x = keras.layers.SeparableConv1D(48, 3, padding="same", activation="relu", name="sepconv1_k3")(x)
    x = keras.layers.SeparableConv1D(48, 3, padding="same", activation="relu", name="sepconv2_k3")(x)
    gap = keras.layers.GlobalAveragePooling1D(name="gap_temporal_features")(x)
    gmp = keras.layers.GlobalMaxPooling1D(name="gmp_temporal_features")(x)
    latest = LatestTemporalFeatures(name="latest_temporal_features")(x)

    seasonal_trajectory = SeasonalNaiveTrajectory(
        forecast_horizon_slots=forecast_horizon_slots,
        temperature_feature_index=temperature_feature_index,
        temperature_feature_mean=temperature_feature_mean,
        temperature_feature_scale=temperature_feature_scale,
        name="seasonal_naive_temperature_trajectory",
    )(input_layer)

    summary = keras.layers.Concatenate(name="residual_context_features")([gap, gmp, latest, seasonal_trajectory])
    short_dense = keras.layers.Dense(64, activation="relu", name="dense_short_projection")(summary)
    short_dense = keras.layers.Dropout(0.10, name="dropout_regularization")(short_dense)
    short_delta = keras.layers.Dense(forecast_horizon_slots, dtype="float32", name="predicted_temperature_delta_short_horizon")(short_dense)

    long_dense = keras.layers.Dense(32, activation="relu", name="dense_long_projection")(summary)
    long_delta = keras.layers.Dense(forecast_horizon_slots, dtype="float32", name="predicted_temperature_delta_long_horizon")(long_dense)

    blended = HorizonBlend(
        weights=np.linspace(0.0, blend_long_weight, forecast_horizon_slots, dtype=np.float32),
        name="blended_temperature_residual_horizon",
    )([short_delta, long_delta])
    trusted = HorizonTrustSchedule(
        schedule=np.linspace(trust_schedule_start, trust_schedule_end, forecast_horizon_slots, dtype=np.float32),
        name="trusted_temperature_residual_horizon",
    )(blended)
    output_layer = keras.layers.Add(name="predicted_temperature_horizon")([seasonal_trajectory, trusted])

    model = keras.Model(inputs=input_layer, outputs=output_layer, name="search_1d_cnn_forecaster")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=build_regression_loss(forecast_horizon_slots=forecast_horizon_slots),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae"), horizon_360m_mae],
    )
    return model

def _candidate_name(candidate: dict[str, object]) -> str:
    return (
        f"days{candidate['measurement_recent_days']}"
        f"_ep{candidate['fine_tune_epochs']}"
        f"_lr{str(candidate['fine_tune_learning_rate']).replace('.', 'p')}"
        f"_bs{candidate['fine_tune_batch_size']}"
        f"_blend{str(candidate['blend_long_weight']).replace('.', 'p')}"
        f"_trust{str(candidate['trust_schedule_start']).replace('.', 'p')}to{str(candidate['trust_schedule_end']).replace('.', 'p')}"
    )


def _clone_model(model: keras.Model) -> keras.Model:
    cloned_model = keras.models.clone_model(model)
    cloned_model.set_weights(model.get_weights())
    cloned_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=model.loss,
        metrics=[keras.metrics.MeanAbsoluteError(name="mae"), horizon_360m_mae],
    )
    return cloned_model


def _write_candidate_training_summary(
    *,
    output_directory: Path,
    feature_columns: tuple[str, ...],
    broad_dataset,
    broad_training: TrainingResult,
    recent_dataset,
    candidate: dict[str, object],
    recent_validation_mae: float,
    recent_fine_tuned_model: keras.Model,
    scaler_path: Path,
    model_path: Path,
) -> Path:
    validate_metrics = {
        key: float(value)
        for key, value in recent_fine_tuned_model.evaluate(
            recent_dataset.X_validate,
            recent_dataset.y_validate,
            verbose=0,
            return_dict=True,
        ).items()
    }
    test_metrics = {
        key: float(value)
        for key, value in recent_fine_tuned_model.evaluate(
            recent_dataset.X_test,
            recent_dataset.y_test,
            verbose=0,
            return_dict=True,
        ).items()
    }
    summary = {
        "feature_columns": list(feature_columns),
        "stages": {
            "broad_training": {
                "include_open_meteo": False,
                "measurement_recent_days": 180,
                "dataset_shapes": {
                    "X_train": list(broad_dataset.X_train.shape),
                    "X_validate": list(broad_dataset.X_validate.shape),
                    "X_test": list(broad_dataset.X_test.shape),
                },
                "tuned_validation_mae": float(broad_training.tuned_validation_mae),
                "fine_tuned_validation_mae": float(broad_training.fine_tuned_validation_mae),
            },
            "recent_local_adaptation": {
                "include_open_meteo": False,
                "measurement_recent_days": int(candidate["measurement_recent_days"]),
                "fine_tune_epochs": int(candidate["fine_tune_epochs"]),
                "fine_tune_batch_size": int(candidate["fine_tune_batch_size"]),
                "fine_tune_learning_rate": float(candidate["fine_tune_learning_rate"]),
                "scaler_strategy": "reuse_broad_training_scaler",
                "dataset_shapes": {
                    "X_train": list(recent_dataset.X_train.shape),
                    "X_validate": list(recent_dataset.X_validate.shape),
                    "X_test": list(recent_dataset.X_test.shape),
                },
                "fine_tuned_validation_mae": float(recent_validation_mae),
                "validate_metrics": validate_metrics,
                "test_metrics": test_metrics,
            },
        },
        "artifacts": {
            "best_pruned_model": str(model_path),
            "feature_scaler": str(scaler_path),
        },
    }
    summary_path = output_directory / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary_path


def _run_preflash_gate(project_root: Path, output_directory: Path, feature_columns: tuple[str, ...]) -> dict[str, object]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "local_cnn.preflash_evaluation",
        "--model-path",
        str(output_directory / "best_pruned_model.keras"),
        "--training-summary-path",
        str(output_directory / "training_summary.json"),
        "--scaler-path",
        str(output_directory / "feature_scaler.json"),
        "--output-dir",
        str(output_directory / "preflash"),
        "--daily-days",
        "28",
    ]
    for feature_column in feature_columns:
        command.extend(["--feature-column", feature_column])
    subprocess.run(command, cwd=project_root, check=True)
    summary_path = output_directory / "preflash" / "preflash_evaluation_summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    search_root = project_root / "artifacts" / "conservative_retrain_search_v2"
    candidates_root = search_root / "candidates"
    final_output_directory = project_root / "artifacts"
    for directory in (search_root, candidates_root):
        directory.mkdir(parents=True, exist_ok=True)

    feature_columns = LONG_HORIZON_FEATURES
    broad_config = PipelineConfig(
        measurements_directory=DEFAULT_MEASUREMENTS_DIRECTORY,
        output_directory=search_root / "broad_training",
        include_open_meteo=False,
        strict_open_meteo=False,
        skip_pruning=True,
        skip_quantization=True,
        device_preference="gpu",
        gpu_memory_growth=True,
        enable_op_determinism=False,
        measurement_recent_days=180,
        selected_feature_columns=feature_columns,
        fine_tune_epochs=16,
        fine_tune_batch_size=64,
        fine_tune_learning_rate=2e-4,
        random_seed=42,
    )

    candidate_grid: list[dict[str, object]] = [
        {"measurement_recent_days": 10, "fine_tune_epochs": 18, "fine_tune_batch_size": 64, "fine_tune_learning_rate": 5e-5, "blend_long_weight": 0.75, "trust_schedule_start": 0.85, "trust_schedule_end": 0.10},
        {"measurement_recent_days": 10, "fine_tune_epochs": 18, "fine_tune_batch_size": 64, "fine_tune_learning_rate": 5e-5, "blend_long_weight": 0.60, "trust_schedule_start": 0.80, "trust_schedule_end": 0.05},
        {"measurement_recent_days": 10, "fine_tune_epochs": 24, "fine_tune_batch_size": 64, "fine_tune_learning_rate": 6e-5, "blend_long_weight": 0.85, "trust_schedule_start": 0.90, "trust_schedule_end": 0.12},
        {"measurement_recent_days": 14, "fine_tune_epochs": 18, "fine_tune_batch_size": 64, "fine_tune_learning_rate": 5e-5, "blend_long_weight": 0.75, "trust_schedule_start": 0.85, "trust_schedule_end": 0.10},
        {"measurement_recent_days": 14, "fine_tune_epochs": 18, "fine_tune_batch_size": 64, "fine_tune_learning_rate": 5e-5, "blend_long_weight": 0.60, "trust_schedule_start": 0.80, "trust_schedule_end": 0.05},
        {"measurement_recent_days": 14, "fine_tune_epochs": 24, "fine_tune_batch_size": 64, "fine_tune_learning_rate": 6e-5, "blend_long_weight": 0.85, "trust_schedule_start": 0.90, "trust_schedule_end": 0.12},
    ]

    seed_everything(broad_config.random_seed, enable_op_determinism=broad_config.enable_op_determinism)

    print("Building broad local-only training dataset...", flush=True)
    broad_weather_dataframe = build_weather_dataframe(broad_config)
    broad_dataset = prepare_dataset(broad_weather_dataframe, broad_config)
    print("Running broad fixed-architecture training stage...", flush=True)
    temperature_feature_index = broad_dataset.feature_columns.index("temperature")
    temperature_feature_mean = float(broad_dataset.scaler.mean_[temperature_feature_index])
    temperature_feature_scale = float(broad_dataset.scaler.scale_[temperature_feature_index])
    print("Building broad model...", flush=True)
    tuned_model = build_search_model(
        input_window_length=broad_dataset.X_train.shape[1],
        number_of_input_features=broad_dataset.X_train.shape[-1],
        forecast_horizon_slots=broad_dataset.y_train.shape[-1],
        temperature_feature_mean=temperature_feature_mean,
        temperature_feature_scale=temperature_feature_scale,
        temperature_feature_index=temperature_feature_index,
        blend_long_weight=0.75,
        trust_schedule_start=0.85,
        trust_schedule_end=0.10,
    )
    print("Broad model compiled.", flush=True)
    print("Starting broad fine-tuning...", flush=True)
    broad_fine_tuned_model, broad_fine_tuned_validation_mae = fine_tune_model(
        tuned_model,
        broad_dataset,
        broad_config,
    )
    broad_training = TrainingResult(
        tuned_model=tuned_model,
        tuned_validation_mae=float("nan"),
        fine_tuned_model=broad_fine_tuned_model,
        fine_tuned_validation_mae=broad_fine_tuned_validation_mae,
    )

    candidate_results: list[dict[str, object]] = []
    best_rank: tuple[int, float, float, float] | None = None
    best_candidate_output_dir: Path | None = None
    best_candidate_summary: dict[str, object] | None = None

    for candidate in candidate_grid:
        candidate_name = _candidate_name(candidate)
        candidate_output_directory = candidates_root / candidate_name
        candidate_output_directory.mkdir(parents=True, exist_ok=True)
        print(f"Evaluating candidate {candidate_name}...", flush=True)

        recent_config = PipelineConfig(
            measurements_directory=DEFAULT_MEASUREMENTS_DIRECTORY,
            output_directory=candidate_output_directory,
            include_open_meteo=False,
            skip_pruning=True,
            skip_quantization=True,
            device_preference="gpu",
            gpu_memory_growth=True,
            enable_op_determinism=False,
            measurement_recent_days=int(candidate["measurement_recent_days"]),
            selected_feature_columns=feature_columns,
            fine_tune_epochs=int(candidate["fine_tune_epochs"]),
            fine_tune_batch_size=int(candidate["fine_tune_batch_size"]),
            fine_tune_learning_rate=float(candidate["fine_tune_learning_rate"]),
            random_seed=42,
        )

        recent_weather_dataframe = build_weather_dataframe(recent_config)
        recent_dataset = prepare_dataset(
            recent_weather_dataframe,
            recent_config,
            existing_scaler=broad_dataset.scaler,
        )
        print(f"Starting recent local fine-tuning for {candidate_name}...", flush=True)
        candidate_model = build_search_model(
            input_window_length=broad_dataset.X_train.shape[1],
            number_of_input_features=broad_dataset.X_train.shape[-1],
            forecast_horizon_slots=broad_dataset.y_train.shape[-1],
            temperature_feature_mean=temperature_feature_mean,
            temperature_feature_scale=temperature_feature_scale,
            temperature_feature_index=temperature_feature_index,
            blend_long_weight=float(candidate["blend_long_weight"]),
            trust_schedule_start=float(candidate["trust_schedule_start"]),
            trust_schedule_end=float(candidate["trust_schedule_end"]),
        )
        candidate_model.set_weights(broad_fine_tuned_model.get_weights())
        recent_fine_tuned_model, recent_validation_mae = fine_tune_model(
            candidate_model,
            recent_dataset,
            recent_config,
        )

        model_path = candidate_output_directory / "best_pruned_model.keras"
        recent_fine_tuned_model.save(model_path)
        scaler_path = _save_scaler(broad_dataset, candidate_output_directory / "feature_scaler.json")
        training_summary_path = _write_candidate_training_summary(
            output_directory=candidate_output_directory,
            feature_columns=feature_columns,
            broad_dataset=broad_dataset,
            broad_training=broad_training,
            recent_dataset=recent_dataset,
            candidate=candidate,
            recent_validation_mae=recent_validation_mae,
            recent_fine_tuned_model=recent_fine_tuned_model,
            scaler_path=scaler_path,
            model_path=model_path,
        )
        print(f"Running preflash gate for {candidate_name}...", flush=True)
        gate_summary = _run_preflash_gate(project_root, candidate_output_directory, feature_columns)
        gate = gate_summary["gate"]
        test_metrics = recent_fine_tuned_model.evaluate(recent_dataset.X_test, recent_dataset.y_test, verbose=0, return_dict=True)

        candidate_record = {
            "candidate_name": candidate_name,
            "measurement_recent_days": int(candidate["measurement_recent_days"]),
            "fine_tune_epochs": int(candidate["fine_tune_epochs"]),
            "fine_tune_batch_size": int(candidate["fine_tune_batch_size"]),
            "fine_tune_learning_rate": float(candidate["fine_tune_learning_rate"]),
            "blend_long_weight": float(candidate["blend_long_weight"]),
            "trust_schedule_start": float(candidate["trust_schedule_start"]),
            "trust_schedule_end": float(candidate["trust_schedule_end"]),
            "recent_validation_mae": float(recent_validation_mae),
            "recent_test_mae": float(test_metrics["mae"]),
            "recent_test_360m_mae": float(test_metrics["horizon_360m_mae"]),
            "overall_pass": bool(gate["overall_pass"]),
            "blind_recent_360_delta_c": gate["blind_recent_window_delta_c"],
            "blind_daily_mean_360_delta_c": gate["blind_daily_mean_delta_c"],
            "blind_daily_worst_360_delta_c": gate["blind_daily_worst_delta_c"],
            "blind_startup_mean_360_delta_c": gate["blind_startup_mean_delta_c"],
            "blind_startup_worst_360_delta_c": gate["blind_startup_worst_delta_c"],
            "artifacts_dir": str(candidate_output_directory),
            "training_summary_path": str(training_summary_path),
        }
        candidate_results.append(candidate_record)

        rank = (
            0 if gate["overall_pass"] else 1,
            float(gate["blind_daily_worst_delta_c"] if gate["blind_daily_worst_delta_c"] is not None else 1e9),
            float(gate["blind_startup_worst_delta_c"] if gate["blind_startup_worst_delta_c"] is not None else 1e9),
            float(gate["blind_recent_window_delta_c"] if gate["blind_recent_window_delta_c"] is not None else 1e9),
        )
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_candidate_output_dir = candidate_output_directory
            best_candidate_summary = {
                "candidate": candidate_record,
                "gate": gate_summary,
            }

    results_path = search_root / "candidate_results.json"
    results_path.write_text(json.dumps(candidate_results, indent=2), encoding="utf-8")

    final_summary = {
        "feature_columns": list(feature_columns),
        "best_candidate": best_candidate_summary,
        "results_path": str(results_path),
        "promoted_to_main_artifacts": False,
    }

    if best_candidate_output_dir is not None and best_candidate_summary is not None:
        if bool(best_candidate_summary["gate"]["gate"]["overall_pass"]):
            shutil.copy2(best_candidate_output_dir / "best_pruned_model.keras", final_output_directory / "best_pruned_model.keras")
            shutil.copy2(best_candidate_output_dir / "feature_scaler.json", final_output_directory / "feature_scaler.json")
            shutil.copy2(best_candidate_output_dir / "training_summary.json", final_output_directory / "training_summary.json")
            final_summary["promoted_to_main_artifacts"] = True
            final_summary["promoted_candidate_directory"] = str(best_candidate_output_dir)

    summary_path = search_root / "search_summary.json"
    summary_path.write_text(json.dumps(final_summary, indent=2), encoding="utf-8")
    print(summary_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


