from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from .calibration import apply_horizon_bias_calibration, load_horizon_bias_calibration
from .config import DEFAULT_MEASUREMENTS_DIRECTORY, DEFAULT_OUTPUT_DIRECTORY, PipelineConfig
from .data import _read_measurement_csv, build_weather_dataframe
from .features import prepare_dataset
from .modeling import horizon_360m_mae, horizon_60m_mae


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay the trained local CNN against measurements over a baseline log time span.",
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
        help="Directory for replay outputs.",
    )
    parser.add_argument(
        "--bias-calibration-path",
        type=Path,
        default=None,
        help="Optional per-horizon bias calibration artifact to subtract from predictions.",
    )
    parser.add_argument(
        "--horizon-minutes",
        type=int,
        default=60,
        help="Forecast horizon to score, in minutes.",
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
        help="Dataset split to replay.",
    )
    parser.add_argument(
        "--feature-column",
        action="append",
        dest="feature_columns",
        help="Optional explicit feature list to use when rebuilding the dataset for this model.",
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


def _load_measurements(
    measurement_paths: list[Path],
    sensor: str,
    quantity: str,
) -> pd.DataFrame:
    measurement_frames: list[pd.DataFrame] = []
    for path in measurement_paths:
        raw = _read_measurement_csv(path)
        filtered = raw.loc[
            (raw["sensor"] == sensor) & (raw["quantity"] == quantity),
            ["timestamp_iso8601", "value"],
        ].copy()
        if filtered.empty:
            continue
        filtered = filtered.rename(
            columns={
                "timestamp_iso8601": "timestamp",
                "value": "measured_temperature_c",
            }
        )
        filtered["timestamp"] = pd.to_datetime(filtered["timestamp"], utc=True, errors="coerce")
        filtered["measured_temperature_c"] = pd.to_numeric(filtered["measured_temperature_c"], errors="coerce")
        filtered = filtered.dropna(subset=["timestamp", "measured_temperature_c"])
        filtered["source_file"] = path.name
        measurement_frames.append(filtered.loc[:, ["timestamp", "measured_temperature_c", "source_file"]])

    if not measurement_frames:
        raise ValueError(f"No measurements found for sensor={sensor!r} quantity={quantity!r}.")

    return pd.concat(measurement_frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def _select_measurement_paths(measurements_dir: Path, forecasted_timestamps_utc: pd.Series) -> list[Path]:
    start_day = forecasted_timestamps_utc.min().date()
    end_day = forecasted_timestamps_utc.max().date()
    day_range = pd.date_range(start_day, end_day, freq="D")

    paths = []
    for day in day_range:
        candidate = measurements_dir / f"measurements_{day.date().isoformat()}.csv"
        if candidate.exists():
            paths.append(candidate)

    if not paths:
        raise FileNotFoundError("No matching measurement files found for the replay date range.")

    return paths


def _select_split_indices(dataset, split: str) -> np.ndarray:
    sample_count = len(dataset.input_sequences)
    train_count = len(dataset.X_train)
    validate_count = len(dataset.X_validate)
    test_count = len(dataset.X_test)

    if split == "all":
        return np.arange(sample_count)
    if split == "train":
        return np.arange(0, train_count)
    if split == "validate":
        return np.arange(train_count, train_count + validate_count)
    if split == "test":
        return np.arange(train_count + validate_count, train_count + validate_count + test_count)
    raise ValueError(f"Unsupported split {split!r}")


def _format_timestamp_iso8601(timestamp: pd.Timestamp) -> str:
    return timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def _compute_baseline_same_grid_metrics(
    replay_timestamps_utc: pd.Series,
    baseline_log: pd.DataFrame,
    measurements_df: pd.DataFrame,
    horizon_minutes: int,
    merge_tolerance_minutes: int,
) -> tuple[pd.DataFrame, float, float]:
    forecast_column = f"forecast_t+{horizon_minutes}m_c"
    if forecast_column not in baseline_log.columns:
        raise ValueError(
            f"Column {forecast_column!r} is missing from the baseline log. "
            f"Available forecast columns: {[column for column in baseline_log.columns if column.startswith('forecast_t+')]}"
        )

    baseline_timestamps = pd.to_datetime(
        baseline_log["timestamp_iso8601"],
        utc=True,
        errors="coerce",
    )
    baseline_reference_df = pd.DataFrame(
        {
            "baseline_timestamp": baseline_timestamps,
            "predicted_temperature_c": pd.to_numeric(baseline_log[forecast_column], errors="coerce"),
        }
    ).dropna()
    baseline_reference_df = baseline_reference_df.sort_values("baseline_timestamp")

    replay_reference_df = pd.DataFrame({"replay_timestamp": replay_timestamps_utc}).sort_values("replay_timestamp")
    matched_baseline_df = pd.merge_asof(
        replay_reference_df,
        baseline_reference_df,
        left_on="replay_timestamp",
        right_on="baseline_timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=merge_tolerance_minutes),
    ).dropna(subset=["predicted_temperature_c"])
    if matched_baseline_df.empty:
        raise ValueError("No baseline rows matched the replay timestamps within the tolerance window.")

    matched_baseline_df["forecasted_timestamp"] = matched_baseline_df["baseline_timestamp"] + pd.Timedelta(
        minutes=horizon_minutes
    )
    aligned_baseline_df = pd.merge_asof(
        matched_baseline_df.sort_values("forecasted_timestamp"),
        measurements_df.loc[:, ["timestamp", "measured_temperature_c"]].sort_values("timestamp"),
        left_on="forecasted_timestamp",
        right_on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=merge_tolerance_minutes),
        suffixes=("_forecast_gen", "_measurement"),
    ).dropna(subset=["measured_temperature_c"])
    if aligned_baseline_df.empty:
        raise ValueError("No baseline forecast/measurement pairs were found on the replay grid.")

    aligned_baseline_df["absolute_error"] = (
        aligned_baseline_df["predicted_temperature_c"] - aligned_baseline_df["measured_temperature_c"]
    ).abs()
    mae = float(aligned_baseline_df["absolute_error"].mean())
    median_ae = float(aligned_baseline_df["absolute_error"].median())
    return aligned_baseline_df, mae, median_ae


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

    sample_timestamps_utc = (
        split_sample_timestamps_local.tz_localize(config.timezone).tz_convert("UTC")
    )

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
        custom_objects={"horizon_60m_mae": horizon_60m_mae, "horizon_360m_mae": horizon_360m_mae},
        compile=False,
    )
    predictions = model.predict(dataset.input_sequences[replay_indices], verbose=0)
    bias_calibration = load_horizon_bias_calibration(args.bias_calibration_path)
    predictions = apply_horizon_bias_calibration(predictions, bias_calibration)

    resample_delta = pd.Timedelta(config.resample_frequency)
    horizon_delta = pd.Timedelta(minutes=args.horizon_minutes)
    if horizon_delta % resample_delta != pd.Timedelta(0):
        raise ValueError(
            f"Horizon {args.horizon_minutes} minutes is not divisible by resample frequency {config.resample_frequency}."
        )
    horizon_index = int(horizon_delta / resample_delta) - 1
    if horizon_index < 0 or horizon_index >= predictions.shape[1]:
        raise ValueError(
            f"Horizon index {horizon_index} is out of bounds for model outputs with shape {predictions.shape}."
        )

    replay_dataframe = pd.DataFrame(
        {
            "timestamp_iso8601": [_format_timestamp_iso8601(ts) for ts in replay_sample_timestamps_utc],
            "timestamp_local": replay_sample_timestamps_local.astype(str),
            "forecasted_timestamp_iso8601": [
                _format_timestamp_iso8601(ts + horizon_delta) for ts in replay_sample_timestamps_utc
            ],
            f"forecast_t+{args.horizon_minutes}m_c": predictions[:, horizon_index].astype(float),
        }
    )

    replay_forecasted_timestamps_utc = pd.to_datetime(
        replay_dataframe["forecasted_timestamp_iso8601"],
        utc=True,
    )
    measurement_paths = _select_measurement_paths(args.measurements_dir, replay_forecasted_timestamps_utc)
    measurements_df = _load_measurements(
        measurement_paths=measurement_paths,
        sensor=args.sensor,
        quantity=args.quantity,
    )

    forecast_column = f"forecast_t+{args.horizon_minutes}m_c"
    replay_alignment_df = replay_dataframe.rename(columns={forecast_column: "predicted_temperature_c"}).copy()
    replay_alignment_df["forecasted_timestamp"] = replay_forecasted_timestamps_utc
    replay_alignment_df["timestamp"] = pd.to_datetime(replay_alignment_df["timestamp_iso8601"], utc=True)
    replay_alignment_df = replay_alignment_df.sort_values("forecasted_timestamp")

    aligned_df = pd.merge_asof(
        replay_alignment_df.loc[:, ["timestamp", "forecasted_timestamp", "predicted_temperature_c"]],
        measurements_df.loc[:, ["timestamp", "measured_temperature_c"]].sort_values("timestamp"),
        left_on="forecasted_timestamp",
        right_on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=args.merge_tolerance_minutes),
        suffixes=("_forecast_gen", "_measurement"),
    )
    aligned_df = aligned_df.rename(columns={"timestamp_measurement": "timestamp"}).dropna(
        subset=["measured_temperature_c"]
    )
    if aligned_df.empty:
        raise ValueError("No overlapping local CNN forecast/measurement pairs were found.")

    aligned_df["absolute_error"] = (
        aligned_df["predicted_temperature_c"] - aligned_df["measured_temperature_c"]
    ).abs()
    mae = float(aligned_df["absolute_error"].mean())
    median_ae = float(aligned_df["absolute_error"].median())

    aligned_baseline_df, baseline_same_grid_mae, baseline_same_grid_median_ae = (
        _compute_baseline_same_grid_metrics(
            replay_timestamps_utc=pd.to_datetime(replay_dataframe["timestamp_iso8601"], utc=True),
            baseline_log=baseline_log,
            measurements_df=measurements_df,
            horizon_minutes=args.horizon_minutes,
            merge_tolerance_minutes=args.merge_tolerance_minutes,
        )
    )

    replay_output_path = args.output_dir / f"replay_forecast_t_plus_{args.horizon_minutes}m.csv"
    aligned_output_path = args.output_dir / f"replay_aligned_t_plus_{args.horizon_minutes}m.csv"
    baseline_aligned_output_path = args.output_dir / f"baseline_aligned_on_replay_grid_t_plus_{args.horizon_minutes}m.csv"
    summary_output_path = args.output_dir / f"replay_summary_t_plus_{args.horizon_minutes}m.json"
    replay_dataframe.to_csv(replay_output_path, index=False)
    aligned_df.to_csv(aligned_output_path, index=False)
    aligned_baseline_df.to_csv(baseline_aligned_output_path, index=False)

    split_sample_count = len(split_indices)
    split_start_utc = sample_timestamps_utc.min()
    split_end_utc = sample_timestamps_utc.max()
    summary = {
        "split": args.split,
        "horizon_minutes": args.horizon_minutes,
        "merge_tolerance_minutes": args.merge_tolerance_minutes,
        "baseline_log": str(args.baseline_log),
        "model_path": str(args.model_path),
        "bias_calibration_path": str(args.bias_calibration_path) if bias_calibration is not None else None,
        "bias_calibration_applied": bias_calibration is not None,
        "replay_sample_count": len(replay_dataframe),
        "paired_rows": len(aligned_df),
        "mae_c": mae,
        "median_ae_c": median_ae,
        "baseline_same_grid": {
            "paired_rows": len(aligned_baseline_df),
            "mae_c": baseline_same_grid_mae,
            "median_ae_c": baseline_same_grid_median_ae,
        },
        "split_coverage_utc": {
            "start": _format_timestamp_iso8601(split_start_utc),
            "end": _format_timestamp_iso8601(split_end_utc),
            "sample_count": split_sample_count,
        },
        "baseline_coverage_utc": {
            "start": _format_timestamp_iso8601(baseline_start_utc),
            "end": _format_timestamp_iso8601(baseline_end_utc),
            "sample_count": int(len(baseline_timestamps_utc)),
        },
        "replay_coverage_utc": {
            "start": replay_dataframe["timestamp_iso8601"].min(),
            "end": replay_dataframe["timestamp_iso8601"].max(),
        },
        "measurement_files": [str(path) for path in measurement_paths],
        "artifacts": {
            "replay_csv": str(replay_output_path),
            "aligned_csv": str(aligned_output_path),
            "baseline_aligned_csv": str(baseline_aligned_output_path),
        },
    }
    summary_output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Split: {args.split}")
    print(f"Split coverage (UTC): {split_start_utc} to {split_end_utc} ({split_sample_count} samples)")
    print(f"Baseline coverage (UTC): {baseline_start_utc} to {baseline_end_utc} ({len(baseline_timestamps_utc)} rows)")
    print(f"Replay rows in baseline range: {len(replay_dataframe)}")
    print(f"Paired rows: {len(aligned_df)}")
    print(f"Local CNN MAE @ +{args.horizon_minutes}m: {mae:.3f} °C")
    print(f"Local CNN median AE @ +{args.horizon_minutes}m: {median_ae:.3f} °C")
    print(f"Baseline same-grid MAE @ +{args.horizon_minutes}m: {baseline_same_grid_mae:.3f} °C")
    print(f"Baseline same-grid median AE @ +{args.horizon_minutes}m: {baseline_same_grid_median_ae:.3f} °C")
    print(f"Replay CSV: {replay_output_path}")
    print(f"Aligned CSV: {aligned_output_path}")
    print(f"Baseline aligned CSV: {baseline_aligned_output_path}")
    print(f"Summary JSON: {summary_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
