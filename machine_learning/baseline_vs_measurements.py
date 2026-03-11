from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_DATA_DIRS = (Path("../measurements"), Path("measurements"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline forecasts against true measurements.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing baseline_forecast.log and measurements_YYYY-MM-DD.csv files.",
    )
    parser.add_argument(
        "--baseline-log",
        default="baseline_forecast.log",
        help="Baseline forecast CSV/log filename.",
    )
    parser.add_argument(
        "--measurement-file",
        dest="measurement_files",
        action="append",
        default=None,
        help="Explicit measurement filename to include. Repeat to add multiple files.",
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
    parser.add_argument(
        "--horizon-minutes",
        type=int,
        default=60,
        help="Forecast horizon in minutes.",
    )
    parser.add_argument(
        "--merge-tolerance-minutes",
        type=int,
        default=15,
        help="Maximum timestamp difference when pairing forecast and measurement rows.",
    )
    parser.add_argument(
        "--save-aligned",
        type=Path,
        default=None,
        help="Optional output CSV path for aligned rows.",
    )
    return parser.parse_args()


def resolve_data_dir(explicit_data_dir: Path | None) -> Path:
    if explicit_data_dir is not None:
        if not explicit_data_dir.exists():
            raise FileNotFoundError(f"Could not find data directory at {explicit_data_dir.resolve()}")
        return explicit_data_dir

    for candidate in DEFAULT_DATA_DIRS:
        if candidate.exists():
            return candidate

    raise FileNotFoundError("Could not find measurements directory.")


def forecast_column_for_horizon(minutes: int) -> str:
    return f"forecast_t+{minutes}m_c"


def load_baseline_frame(baseline_log_path: Path, horizon_minutes: int) -> pd.DataFrame:
    forecast_column = forecast_column_for_horizon(horizon_minutes)
    baseline_raw = pd.read_csv(
        baseline_log_path,
        parse_dates=["timestamp_iso8601"],
        on_bad_lines="skip",
    )
    if baseline_raw.empty:
        raise ValueError(f"Baseline log {baseline_log_path.name} has no rows.")
    if forecast_column not in baseline_raw.columns:
        available = [column for column in baseline_raw.columns if column.startswith("forecast_t+")]
        raise ValueError(
            f"Column {forecast_column!r} was not found in {baseline_log_path.name}. "
            f"Available forecast columns: {available}"
        )

    baseline_df = (
        baseline_raw.rename(
            columns={
                "timestamp_iso8601": "timestamp",
                forecast_column: "predicted_temperature_c",
            }
        )
        .assign(source_file=baseline_log_path.name)
        .loc[:, ["timestamp", "predicted_temperature_c", "source_file"]]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    baseline_df["forecasted_timestamp"] = baseline_df["timestamp"] + pd.Timedelta(minutes=horizon_minutes)
    return baseline_df.loc[:, ["timestamp", "forecasted_timestamp", "predicted_temperature_c", "source_file"]]


def select_measurement_paths(
    data_dir: Path,
    measurement_filenames: list[str] | None,
    baseline_df: pd.DataFrame,
) -> list[Path]:
    if measurement_filenames:
        selected_measurement_paths = [data_dir / filename for filename in measurement_filenames]
    else:
        selected_measurement_paths = []
        baseline_dates = pd.date_range(
            baseline_df["forecasted_timestamp"].min().normalize(),
            baseline_df["forecasted_timestamp"].max().normalize(),
            freq="D",
            tz=baseline_df["forecasted_timestamp"].dt.tz,
        )
        for day in baseline_dates:
            candidate = data_dir / f"measurements_{day.date().isoformat()}.csv"
            if candidate.exists():
                selected_measurement_paths.append(candidate)

    if not selected_measurement_paths:
        raise FileNotFoundError("Could not find matching measurement files.")

    missing_measurements = [path for path in selected_measurement_paths if not path.exists()]
    if missing_measurements:
        missing_names = ", ".join(path.name for path in missing_measurements)
        raise FileNotFoundError(f"Missing measurement files: {missing_names}")

    return selected_measurement_paths


def load_measurements_frame(
    measurement_paths: list[Path],
    sensor: str,
    quantity: str,
) -> pd.DataFrame:
    measurement_frames: list[pd.DataFrame] = []
    for path in measurement_paths:
        raw = pd.read_csv(path, parse_dates=["timestamp_iso8601"], on_bad_lines="skip")
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
        filtered["measured_temperature_c"] = pd.to_numeric(filtered["measured_temperature_c"], errors="coerce")
        filtered = filtered.dropna(subset=["timestamp", "measured_temperature_c"])
        filtered["source_file"] = path.name
        measurement_frames.append(filtered.loc[:, ["timestamp", "measured_temperature_c", "source_file"]])

    if not measurement_frames:
        raise ValueError(f"No measurements found for sensor={sensor!r} quantity={quantity!r}.")

    return pd.concat(measurement_frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def align_forecasts_to_measurements(
    baseline_df: pd.DataFrame,
    measurements_df: pd.DataFrame,
    tolerance_minutes: int,
) -> pd.DataFrame:
    aligned_df = pd.merge_asof(
        baseline_df.sort_values("forecasted_timestamp"),
        measurements_df.loc[:, ["timestamp", "measured_temperature_c"]].sort_values("timestamp"),
        left_on="forecasted_timestamp",
        right_on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=tolerance_minutes),
        suffixes=("_forecast_gen", "_measurement"),
    )
    aligned_df = aligned_df.rename(columns={"timestamp_measurement": "timestamp"})
    aligned_df = aligned_df.dropna(subset=["measured_temperature_c"]).copy()
    if aligned_df.empty:
        raise ValueError(
            "No overlapping baseline forecast/measurement pairs were found within the tolerance window."
        )
    aligned_df["absolute_error"] = (
        aligned_df["predicted_temperature_c"] - aligned_df["measured_temperature_c"]
    ).abs()
    return aligned_df


def main() -> None:
    args = parse_args()
    data_dir = resolve_data_dir(args.data_dir)
    baseline_log_path = data_dir / args.baseline_log
    if not baseline_log_path.exists():
        raise FileNotFoundError(f"Could not find baseline log at {baseline_log_path}")

    baseline_df = load_baseline_frame(baseline_log_path, args.horizon_minutes)
    measurement_paths = select_measurement_paths(data_dir, args.measurement_files, baseline_df)
    measurements_df = load_measurements_frame(measurement_paths, args.sensor, args.quantity)
    aligned_df = align_forecasts_to_measurements(
        baseline_df=baseline_df,
        measurements_df=measurements_df,
        tolerance_minutes=args.merge_tolerance_minutes,
    )

    mae = aligned_df["absolute_error"].mean()
    median_ae = aligned_df["absolute_error"].median()

    print(f"Using data directory: {data_dir.resolve()}")
    print(f"Using baseline log: {baseline_log_path.name}")
    print("Selected measurement files:")
    for path in measurement_paths:
        print(f"- {path.name}")
    print(f"Baseline timestamp range: {baseline_df['timestamp'].min()} to {baseline_df['timestamp'].max()}")
    print(
        "Baseline forecasted range: "
        f"{baseline_df['forecasted_timestamp'].min()} to {baseline_df['forecasted_timestamp'].max()}"
    )
    print(f"Measurement range: {measurements_df['timestamp'].min()} to {measurements_df['timestamp'].max()}")
    print(f"Paired rows: {len(aligned_df)}")
    print(f"Baseline MAE: {mae:.3f} °C")
    print(f"Baseline median AE: {median_ae:.3f} °C")

    if args.save_aligned is not None:
        aligned_df.to_csv(args.save_aligned, index=False)
        print(f"Aligned output saved to: {args.save_aligned.resolve()}")


if __name__ == "__main__":
    main()
