from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from .calibration import apply_horizon_bias_calibration, load_horizon_bias_calibration
from .config import DEFAULT_MEASUREMENTS_DIRECTORY, DEFAULT_OUTPUT_DIRECTORY, PipelineConfig
from .data import build_weather_dataframe
from .evaluate_vs_measurements import (
    _compute_baseline_same_grid_metrics,
    _format_timestamp_iso8601,
    _load_measurements,
    _select_measurement_paths,
    _select_split_indices,
)
from .features import prepare_dataset
from .modeling import horizon_60m_mae


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the trained local CNN against the baseline log across multiple horizons.",
    )
    parser.add_argument(
        "--measurements-dir",
        type=Path,
        default=DEFAULT_MEASUREMENTS_DIRECTORY,
        help="Directory containing measurements_*.csv files and the baseline log.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY / "best_pruned_model.keras",
        help="Path to the trained Keras model artifact.",
    )
    parser.add_argument(
        "--baseline-log",
        type=Path,
        default=DEFAULT_MEASUREMENTS_DIRECTORY / "baseline_forecast.log",
        help="Baseline log used to define the evaluation time span.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for benchmark outputs.",
    )
    parser.add_argument(
        "--bias-calibration-path",
        type=Path,
        default=None,
        help="Optional per-horizon bias calibration artifact to subtract from predictions.",
    )
    parser.add_argument(
        "--merge-tolerance-minutes",
        type=int,
        default=15,
        help="Maximum time difference for forecast/measurement pairing.",
    )
    parser.add_argument(
        "--split",
        choices=["all", "train", "validate", "test"],
        default="test",
        help="Dataset split to benchmark.",
    )
    parser.add_argument(
        "--feature-column",
        action="append",
        dest="feature_columns",
        help="Optional explicit feature list to use when rebuilding the dataset for this model.",
    )
    parser.add_argument(
        "--horizons-minutes",
        nargs="*",
        type=int,
        help="Optional explicit list of horizons to benchmark. Defaults to all horizons present in the baseline log.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Number of bootstrap samples for the MAE difference confidence interval.",
    )
    parser.add_argument(
        "--sensor",
        default="bme280",
        help="Measurement sensor name.",
    )
    parser.add_argument(
        "--quantity",
        default="temperature_c",
        help="Measurement quantity name.",
    )
    return parser


def _extract_baseline_horizons_minutes(baseline_log: pd.DataFrame) -> list[int]:
    horizons: list[int] = []
    for column_name in baseline_log.columns:
        if not column_name.startswith("forecast_t+") or not column_name.endswith("m_c"):
            continue
        horizon_label = column_name.removeprefix("forecast_t+").removesuffix("m_c")
        horizons.append(int(horizon_label))
    return sorted(horizons)


def _align_cnn_predictions(
    replay_sample_timestamps_utc: pd.DatetimeIndex,
    replay_sample_timestamps_local: pd.DatetimeIndex,
    predicted_values: np.ndarray,
    measurements_df: pd.DataFrame,
    horizon_minutes: int,
    merge_tolerance_minutes: int,
) -> pd.DataFrame:
    horizon_delta = pd.Timedelta(minutes=horizon_minutes)
    replay_dataframe = pd.DataFrame(
        {
            "timestamp_iso8601": [_format_timestamp_iso8601(ts) for ts in replay_sample_timestamps_utc],
            "timestamp_local": replay_sample_timestamps_local.astype(str),
            "forecasted_timestamp_iso8601": [
                _format_timestamp_iso8601(ts + horizon_delta) for ts in replay_sample_timestamps_utc
            ],
            "predicted_temperature_c": predicted_values.astype(float),
        }
    )
    replay_dataframe["timestamp_forecast_gen"] = pd.to_datetime(replay_dataframe["timestamp_iso8601"], utc=True)
    replay_dataframe["forecasted_timestamp"] = pd.to_datetime(
        replay_dataframe["forecasted_timestamp_iso8601"],
        utc=True,
    )

    aligned_df = pd.merge_asof(
        replay_dataframe.loc[:, ["timestamp_forecast_gen", "forecasted_timestamp", "predicted_temperature_c"]],
        measurements_df.loc[:, ["timestamp", "measured_temperature_c"]].sort_values("timestamp"),
        left_on="forecasted_timestamp",
        right_on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=merge_tolerance_minutes),
        suffixes=("_forecast_gen", "_measurement"),
    ).dropna(subset=["measured_temperature_c"])

    if aligned_df.empty:
        raise ValueError(f"No overlapping local CNN forecast/measurement pairs were found for +{horizon_minutes}m.")

    aligned_df["absolute_error"] = (
        aligned_df["predicted_temperature_c"] - aligned_df["measured_temperature_c"]
    ).abs()
    aligned_df["squared_error"] = (
        aligned_df["predicted_temperature_c"] - aligned_df["measured_temperature_c"]
    ) ** 2
    aligned_df["signed_error"] = (
        aligned_df["predicted_temperature_c"] - aligned_df["measured_temperature_c"]
    )
    return aligned_df


def _bootstrap_mae_difference_ci(
    cnn_absolute_errors: np.ndarray,
    baseline_absolute_errors: np.ndarray,
    bootstrap_samples: int,
    seed: int = 42,
) -> tuple[float, float]:
    if len(cnn_absolute_errors) != len(baseline_absolute_errors):
        raise ValueError("CNN and baseline error arrays must have the same length for paired bootstrap.")
    if len(cnn_absolute_errors) == 0:
        raise ValueError("Cannot bootstrap an empty evaluation set.")

    rng = np.random.default_rng(seed)
    sample_size = len(cnn_absolute_errors)
    bootstrap_deltas = np.empty(bootstrap_samples, dtype=np.float64)
    for index in range(bootstrap_samples):
        sampled_indices = rng.integers(0, sample_size, size=sample_size)
        bootstrap_deltas[index] = (
            float(np.mean(cnn_absolute_errors[sampled_indices]))
            - float(np.mean(baseline_absolute_errors[sampled_indices]))
        )
    lower, upper = np.quantile(bootstrap_deltas, [0.025, 0.975])
    return float(lower), float(upper)


def main() -> int:
    args = build_argument_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = PipelineConfig(
        measurements_directory=args.measurements_dir,
        output_directory=args.output_dir,
        include_open_meteo=False,
    )
    if args.feature_columns:
        config.selected_feature_columns = tuple(args.feature_columns)
    weather_dataframe = build_weather_dataframe(config)
    dataset = prepare_dataset(weather_dataframe, config)

    sample_timestamps_local = pd.DatetimeIndex(
        dataset.combined_dataframe.index[config.historical_window_slots - 1 :]
    )
    if len(sample_timestamps_local) != len(dataset.input_sequences):
        raise ValueError("Sample timestamp count does not match the number of model input sequences.")

    split_indices = _select_split_indices(dataset, args.split)
    split_sample_timestamps_local = sample_timestamps_local[split_indices]
    sample_timestamps_utc = split_sample_timestamps_local.tz_localize(config.timezone).tz_convert("UTC")

    baseline_log = pd.read_csv(args.baseline_log, parse_dates=["timestamp_iso8601"], on_bad_lines="skip")
    baseline_timestamps_utc = pd.to_datetime(baseline_log["timestamp_iso8601"], utc=True, errors="coerce").dropna()
    if baseline_timestamps_utc.empty:
        raise ValueError(f"No valid timestamps found in baseline log {args.baseline_log}")

    baseline_start_utc = baseline_timestamps_utc.min()
    baseline_end_utc = baseline_timestamps_utc.max()
    in_range_mask = (sample_timestamps_utc >= baseline_start_utc) & (sample_timestamps_utc <= baseline_end_utc)
    replay_indices = split_indices[in_range_mask]
    replay_sample_timestamps_utc = sample_timestamps_utc[in_range_mask]
    replay_sample_timestamps_local = split_sample_timestamps_local[in_range_mask]

    if len(replay_indices) == 0:
        raise ValueError(
            "No local CNN samples overlap the baseline log range after split filtering. "
            "Try --split all to inspect the full replay coverage."
        )

    model = tf.keras.models.load_model(
        args.model_path,
        custom_objects={"horizon_60m_mae": horizon_60m_mae},
        compile=False,
    )
    predictions = model.predict(dataset.input_sequences[replay_indices], verbose=0)
    bias_calibration = load_horizon_bias_calibration(args.bias_calibration_path)
    predictions = apply_horizon_bias_calibration(predictions, bias_calibration)

    available_horizons = _extract_baseline_horizons_minutes(baseline_log)
    horizons_minutes = args.horizons_minutes or available_horizons
    horizons_minutes = [h for h in horizons_minutes if h in available_horizons]
    if not horizons_minutes:
        raise ValueError("No valid benchmark horizons were selected.")

    max_horizon_minutes = max(horizons_minutes)
    max_horizon_delta = pd.Timedelta(minutes=max_horizon_minutes)
    measurement_paths = _select_measurement_paths(
        args.measurements_dir,
        pd.Series(replay_sample_timestamps_utc + max_horizon_delta),
    )
    measurements_df = _load_measurements(
        measurement_paths=measurement_paths,
        sensor=args.sensor,
        quantity=args.quantity,
    )

    resample_delta = pd.Timedelta(config.resample_frequency)
    benchmark_rows: list[dict[str, object]] = []
    per_horizon_artifacts: dict[str, dict[str, str]] = {}

    for horizon_minutes in horizons_minutes:
        horizon_delta = pd.Timedelta(minutes=horizon_minutes)
        if horizon_delta % resample_delta != pd.Timedelta(0):
            continue
        horizon_index = int(horizon_delta / resample_delta) - 1
        if horizon_index < 0 or horizon_index >= predictions.shape[1]:
            continue

        cnn_aligned_df = _align_cnn_predictions(
            replay_sample_timestamps_utc=replay_sample_timestamps_utc,
            replay_sample_timestamps_local=replay_sample_timestamps_local,
            predicted_values=predictions[:, horizon_index],
            measurements_df=measurements_df,
            horizon_minutes=horizon_minutes,
            merge_tolerance_minutes=args.merge_tolerance_minutes,
        )
        baseline_aligned_df, baseline_mae, baseline_median_ae = _compute_baseline_same_grid_metrics(
            replay_timestamps_utc=pd.Series(replay_sample_timestamps_utc),
            baseline_log=baseline_log,
            measurements_df=measurements_df,
            horizon_minutes=horizon_minutes,
            merge_tolerance_minutes=args.merge_tolerance_minutes,
        )

        paired_df = cnn_aligned_df.merge(
            baseline_aligned_df.loc[:, ["replay_timestamp", "absolute_error", "predicted_temperature_c"]].rename(
                columns={
                    "replay_timestamp": "timestamp_forecast_gen",
                    "absolute_error": "baseline_absolute_error",
                    "predicted_temperature_c": "baseline_predicted_temperature_c",
                }
            ),
            on="timestamp_forecast_gen",
            how="inner",
        )
        if paired_df.empty:
            raise ValueError(f"No paired CNN/baseline rows remained for +{horizon_minutes}m after alignment.")

        delta_mae = float(paired_df["absolute_error"].mean() - paired_df["baseline_absolute_error"].mean())
        ci_lower, ci_upper = _bootstrap_mae_difference_ci(
            cnn_absolute_errors=paired_df["absolute_error"].to_numpy(dtype=np.float64),
            baseline_absolute_errors=paired_df["baseline_absolute_error"].to_numpy(dtype=np.float64),
            bootstrap_samples=args.bootstrap_samples,
        )

        cnn_mae = float(cnn_aligned_df["absolute_error"].mean())
        cnn_median_ae = float(cnn_aligned_df["absolute_error"].median())
        cnn_rmse = float(np.sqrt(cnn_aligned_df["squared_error"].mean()))
        cnn_bias = float(cnn_aligned_df["signed_error"].mean())
        baseline_rmse = float(np.sqrt(np.mean(np.square(baseline_aligned_df["absolute_error"].to_numpy()))))
        baseline_wins = int((paired_df["absolute_error"] < paired_df["baseline_absolute_error"]).sum())
        ties = int((paired_df["absolute_error"] == paired_df["baseline_absolute_error"]).sum())

        benchmark_rows.append(
            {
                "horizon_minutes": horizon_minutes,
                "paired_rows": int(len(cnn_aligned_df)),
                "cnn_mae_c": cnn_mae,
                "baseline_mae_c": baseline_mae,
                "mae_delta_cnn_minus_baseline_c": delta_mae,
                "mae_delta_95ci_low_c": ci_lower,
                "mae_delta_95ci_high_c": ci_upper,
                "cnn_median_ae_c": cnn_median_ae,
                "baseline_median_ae_c": baseline_median_ae,
                "cnn_rmse_c": cnn_rmse,
                "baseline_rmse_c": baseline_rmse,
                "cnn_bias_c": cnn_bias,
                "cnn_win_rows": baseline_wins,
                "baseline_win_rows": int(len(paired_df) - baseline_wins - ties),
                "tie_rows": ties,
            }
        )

        cnn_output_path = args.output_dir / f"benchmark_cnn_aligned_t_plus_{horizon_minutes}m.csv"
        baseline_output_path = args.output_dir / f"benchmark_baseline_aligned_t_plus_{horizon_minutes}m.csv"
        paired_output_path = args.output_dir / f"benchmark_paired_t_plus_{horizon_minutes}m.csv"
        cnn_aligned_df.to_csv(cnn_output_path, index=False)
        baseline_aligned_df.to_csv(baseline_output_path, index=False)
        paired_df.to_csv(paired_output_path, index=False)
        per_horizon_artifacts[str(horizon_minutes)] = {
            "cnn_aligned_csv": str(cnn_output_path),
            "baseline_aligned_csv": str(baseline_output_path),
            "paired_csv": str(paired_output_path),
        }

    results_df = pd.DataFrame(benchmark_rows).sort_values("horizon_minutes").reset_index(drop=True)
    results_csv_path = args.output_dir / "benchmark_vs_measurements.csv"
    results_json_path = args.output_dir / "benchmark_vs_measurements.json"
    results_df.to_csv(results_csv_path, index=False)

    summary = {
        "split": args.split,
        "bootstrap_samples": args.bootstrap_samples,
        "merge_tolerance_minutes": args.merge_tolerance_minutes,
        "baseline_log": str(args.baseline_log),
        "model_path": str(args.model_path),
        "bias_calibration_path": str(args.bias_calibration_path) if bias_calibration is not None else None,
        "bias_calibration_applied": bias_calibration is not None,
        "measurement_files": [str(path) for path in measurement_paths],
        "split_coverage_utc": {
            "start": _format_timestamp_iso8601(sample_timestamps_utc.min()),
            "end": _format_timestamp_iso8601(sample_timestamps_utc.max()),
            "sample_count": int(len(split_indices)),
        },
        "baseline_coverage_utc": {
            "start": _format_timestamp_iso8601(baseline_start_utc),
            "end": _format_timestamp_iso8601(baseline_end_utc),
            "sample_count": int(len(baseline_timestamps_utc)),
        },
        "replay_coverage_utc": {
            "start": _format_timestamp_iso8601(replay_sample_timestamps_utc.min()),
            "end": _format_timestamp_iso8601(replay_sample_timestamps_utc.max()),
            "sample_count": int(len(replay_indices)),
        },
        "results": benchmark_rows,
        "artifacts": {
            "results_csv": str(results_csv_path),
            "results_json": str(results_json_path),
            "per_horizon": per_horizon_artifacts,
        },
    }
    results_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Split: {args.split}")
    print(f"Horizons benchmarked: {', '.join(str(h) for h in results_df['horizon_minutes'])}")
    print(f"Results CSV: {results_csv_path}")
    print(f"Results JSON: {results_json_path}")
    for row in benchmark_rows:
        print(
            "horizon=+{h}m cnn_mae={cnn:.3f} baseline_mae={base:.3f} "
            "delta={delta:.3f} ci95=[{low:.3f}, {high:.3f}]".format(
                h=row["horizon_minutes"],
                cnn=row["cnn_mae_c"],
                base=row["baseline_mae_c"],
                delta=row["mae_delta_cnn_minus_baseline_c"],
                low=row["mae_delta_95ci_low_c"],
                high=row["mae_delta_95ci_high_c"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
