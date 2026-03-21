from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


import numpy as np
import pandas as pd
import tensorflow as tf

from .benchmark_vs_measurements import _align_cnn_predictions
from .config import DEFAULT_MEASUREMENTS_DIRECTORY, DEFAULT_OUTPUT_DIRECTORY, PipelineConfig
from .data import build_weather_dataframe
from .evaluate_vs_measurements import (
    _compute_baseline_same_grid_metrics,
    _load_measurements,
    _select_measurement_paths,
)
from sklearn.preprocessing import StandardScaler

from .features import prepare_dataset
from .modeling import horizon_360m_mae, horizon_60m_mae
from .search_layers import HorizonBlend, HorizonTrustSchedule, LatestTemporalFeatures, SeasonalNaiveTrajectory

MEASUREMENT_FILENAME_PATTERN = re.compile(r"measurements_(\d{4}-\d{2}-\d{2})\.csv$")


@dataclass(slots=True)
class EvaluationWindow:
    name: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    overlaps_recent_local_adaptation: bool


def _samples_per_day(resample_frequency: str) -> int:
    return int(pd.Timedelta(days=1) / pd.Timedelta(resample_frequency))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run pre-flash evaluation gates against real measurement files, including "
            "recent-window replay, daily walk-forward summaries, and early-slice checks."
        )
    )
    parser.add_argument("--measurements-dir", type=Path, default=DEFAULT_MEASUREMENTS_DIRECTORY)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_OUTPUT_DIRECTORY / "best_pruned_model.keras")
    parser.add_argument(
        "--training-summary-path",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY / "training_summary.json",
    )
    parser.add_argument(
        "--scaler-path",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY / "feature_scaler.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY / "preflash_evaluation",
    )
    parser.add_argument("--recent-days", type=int, default=21)
    parser.add_argument("--daily-days", type=int, default=14)
    parser.add_argument("--startup-hours", type=int, default=6)
    parser.add_argument("--merge-tolerance-minutes", type=int, default=15)
    parser.add_argument("--feature-column", action="append", dest="feature_columns")
    parser.add_argument("--horizons-minutes", nargs="*", type=int, default=[30, 60, 180, 360])
    parser.add_argument("--guard-horizons-minutes", nargs="*", type=int, default=[60, 180])
    parser.add_argument("--short-horizon-guard-margin-c", type=float, default=0.25)
    parser.add_argument("--sensor", default="bme280")
    parser.add_argument("--quantity", default="temperature_c")
    return parser


def _latest_measurement_date(measurements_dir: Path) -> pd.Timestamp:
    dated_measurements: list[pd.Timestamp] = []
    for path in measurements_dir.glob("measurements_*.csv"):
        match = MEASUREMENT_FILENAME_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        dated_measurements.append(pd.Timestamp(match.group(1)))
    if not dated_measurements:
        raise FileNotFoundError(f"No dated measurements_YYYY-MM-DD.csv files found in {measurements_dir}")
    return max(dated_measurements)


def _adaptation_window_days(training_summary_path: Path) -> int:
    if not training_summary_path.exists():
        return 7
    summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
    stages = summary.get("stages")
    if isinstance(stages, dict):
        recent_stage = stages.get("recent_local_adaptation")
        if isinstance(recent_stage, dict) and recent_stage.get("measurement_recent_days") is not None:
            return int(recent_stage["measurement_recent_days"])

    config = summary.get("config")
    if isinstance(config, dict) and config.get("measurement_recent_days") is not None:
        return int(config["measurement_recent_days"])

    return 7


def _resolve_scaler_path(training_summary_path: Path, scaler_path: Path) -> Path:
    if scaler_path.exists():
        return scaler_path
    if training_summary_path.exists():
        try:
            summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
            artifacts = summary.get("artifacts", {})
            candidate = artifacts.get("feature_scaler")
            if candidate:
                candidate_path = Path(candidate)
                if candidate_path.exists():
                    return candidate_path
        except (json.JSONDecodeError, OSError, TypeError):
            pass
    return scaler_path


def _load_saved_scaler(scaler_path: Path, expected_feature_columns: tuple[str, ...] | None) -> StandardScaler:
    payload = json.loads(scaler_path.read_text(encoding="utf-8"))
    feature_columns = tuple(payload["feature_columns"])
    if expected_feature_columns is not None and feature_columns != expected_feature_columns:
        raise ValueError(
            f"Saved scaler feature order {feature_columns} does not match expected feature order {expected_feature_columns}."
        )
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(payload["mean"], dtype=np.float64)
    scaler.scale_ = np.asarray(payload["scale"], dtype=np.float64)
    scaler.var_ = np.asarray(payload["var"], dtype=np.float64)
    scaler.n_features_in_ = len(feature_columns)
    return scaler


def _resolve_runtime_model_path(model_path: Path, training_summary_path: Path) -> Path:
    if model_path.suffix == ".tflite" and model_path.exists():
        return model_path

    candidate_paths: list[Path] = []
    if training_summary_path.exists():
        try:
            summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
            artifacts = summary.get("artifacts", {})
            for key in ("pruned_int8_tflite", "pruned_float32_tflite"):
                candidate = artifacts.get(key)
                if candidate:
                    candidate_paths.append(Path(candidate))
        except (json.JSONDecodeError, OSError, TypeError):
            pass

    candidate_paths.extend(
        [
            model_path.with_name("pruned_int8.tflite"),
            model_path.with_name("pruned_float32.tflite"),
        ]
    )
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate
    return model_path


def _load_keras_model(model_path: Path) -> tf.keras.Model:
    return tf.keras.models.load_model(
        model_path,
        custom_objects={"horizon_60m_mae": horizon_60m_mae, "horizon_360m_mae": horizon_360m_mae, "LatestTemporalFeatures": LatestTemporalFeatures, "SeasonalNaiveTrajectory": SeasonalNaiveTrajectory, "HorizonBlend": HorizonBlend, "HorizonTrustSchedule": HorizonTrustSchedule},
        compile=False,
        safe_mode=False,
    )


def _predict_with_tflite(model_path: Path, inputs: np.ndarray) -> np.ndarray:
    interpreter = tf.lite.Interpreter(
        model_path=str(model_path),
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
    )
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_dtype = input_details["dtype"]
    output_dtype = output_details["dtype"]
    input_scale, input_zero_point = input_details["quantization"]
    output_scale, output_zero_point = output_details["quantization"]

    output_shape = tuple(int(dimension) for dimension in output_details["shape"])
    predictions = np.empty((inputs.shape[0], *output_shape[1:]), dtype=np.float32)

    for index in range(inputs.shape[0]):
        sample = inputs[index : index + 1].astype(np.float32)
        if input_dtype == np.int8:
            quantized_input = np.round(sample / input_scale + input_zero_point).astype(np.int8)
            interpreter.set_tensor(input_details["index"], quantized_input)
        else:
            interpreter.set_tensor(input_details["index"], sample)

        interpreter.invoke()
        raw_output = interpreter.get_tensor(output_details["index"])
        if output_dtype == np.int8:
            predictions[index] = output_scale * (raw_output.astype(np.float32) - output_zero_point)
        else:
            predictions[index] = raw_output.astype(np.float32)

    return predictions


def _predict(model_path: Path, inputs: np.ndarray) -> tuple[np.ndarray, str]:
    if model_path.suffix == ".tflite":
        return _predict_with_tflite(model_path, inputs), "tflite"
    model = _load_keras_model(model_path)
    return model.predict(inputs, verbose=0), "keras"




def _deduplicate_baseline_log(baseline_log: pd.DataFrame) -> pd.DataFrame:
    forecast_columns = [column for column in baseline_log.columns if column.startswith("forecast_t+")]
    deduplicated = baseline_log.copy()
    deduplicated["timestamp_iso8601"] = pd.to_datetime(
        deduplicated["timestamp_iso8601"],
        utc=True,
        errors="coerce",
    )
    deduplicated = deduplicated.dropna(subset=["timestamp_iso8601"]).sort_values("timestamp_iso8601")
    if not forecast_columns:
        return deduplicated.reset_index(drop=True)

    for column in forecast_columns:
        deduplicated[column] = pd.to_numeric(deduplicated[column], errors="coerce")

    deduplicated = deduplicated.drop_duplicates(subset=forecast_columns, keep="first")
    return deduplicated.reset_index(drop=True)

def _build_dataset(
    measurements_dir: Path,
    feature_columns: tuple[str, ...] | None,
    scaler_path: Path,
) -> tuple[PipelineConfig, object]:
    config = PipelineConfig(
        measurements_directory=measurements_dir,
        output_directory=DEFAULT_OUTPUT_DIRECTORY,
        include_open_meteo=False,
    )
    if feature_columns is None:
        payload = json.loads(scaler_path.read_text(encoding="utf-8"))
        feature_columns = tuple(payload["feature_columns"])
    config.selected_feature_columns = feature_columns
    weather_dataframe = build_weather_dataframe(config)
    scaler = _load_saved_scaler(scaler_path, tuple(config.selected_feature_columns))
    dataset = prepare_dataset(weather_dataframe, config, existing_scaler=scaler)
    return config, dataset


def _sample_timestamps(dataset, config: PipelineConfig) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    sample_timestamps_local = pd.DatetimeIndex(
        dataset.combined_dataframe.index[config.historical_window_slots - 1 :]
    )
    if len(sample_timestamps_local) != len(dataset.input_sequences):
        raise ValueError("Sample timestamp count does not match the number of model input sequences.")
    sample_timestamps_utc = sample_timestamps_local.tz_localize(config.timezone).tz_convert("UTC")
    return sample_timestamps_local, sample_timestamps_utc


def _window_mask(
    sample_timestamps_local: pd.DatetimeIndex,
    sample_timestamps_utc: pd.DatetimeIndex,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    baseline_start_utc: pd.Timestamp,
    baseline_end_utc: pd.Timestamp,
    generation_start_local: pd.Timestamp | None = None,
    generation_end_local: pd.Timestamp | None = None,
) -> np.ndarray:
    local_dates = sample_timestamps_local.normalize()
    mask = (
        (local_dates >= start_date)
        & (local_dates <= end_date)
        & (sample_timestamps_utc >= baseline_start_utc)
        & (sample_timestamps_utc <= baseline_end_utc)
    )
    if generation_start_local is not None:
        mask &= sample_timestamps_local >= generation_start_local
    if generation_end_local is not None:
        mask &= sample_timestamps_local < generation_end_local
    return mask


def _score_window(
    *,
    name: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    overlaps_recent_local_adaptation: bool,
    horizons_minutes: list[int],
    sample_timestamps_local: pd.DatetimeIndex,
    sample_timestamps_utc: pd.DatetimeIndex,
    predictions: np.ndarray,
    baseline_log: pd.DataFrame,
    measurements_dir: Path,
    merge_tolerance_minutes: int,
    sensor: str,
    quantity: str,
    generation_start_local: pd.Timestamp | None = None,
    generation_end_local: pd.Timestamp | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    baseline_timestamps_utc = pd.to_datetime(baseline_log["timestamp_iso8601"], utc=True, errors="coerce").dropna()
    mask = _window_mask(
        sample_timestamps_local=sample_timestamps_local,
        sample_timestamps_utc=sample_timestamps_utc,
        start_date=start_date,
        end_date=end_date,
        baseline_start_utc=baseline_timestamps_utc.min(),
        baseline_end_utc=baseline_timestamps_utc.max(),
        generation_start_local=generation_start_local,
        generation_end_local=generation_end_local,
    )
    replay_indices = np.flatnonzero(mask)
    if len(replay_indices) == 0:
        return [], {
            "window": name,
            "start_date": str(start_date.date()),
            "end_date": str(end_date.date()),
            "overlaps_recent_local_adaptation": overlaps_recent_local_adaptation,
            "sample_count": 0,
            "measurement_files": [],
            "results": {},
        }

    replay_sample_timestamps_local = sample_timestamps_local[replay_indices]
    replay_sample_timestamps_utc = sample_timestamps_utc[replay_indices]
    max_horizon_delta = pd.Timedelta(minutes=max(horizons_minutes))
    try:
        measurement_paths = _select_measurement_paths(
            measurements_dir,
            pd.Series(replay_sample_timestamps_utc + max_horizon_delta),
        )
    except FileNotFoundError:
        return [], {
            "window": name,
            "start_date": str(start_date.date()),
            "end_date": str(end_date.date()),
            "overlaps_recent_local_adaptation": overlaps_recent_local_adaptation,
            "sample_count": int(len(replay_indices)),
            "measurement_files": [],
            "results": {},
        }

    measurements_df = _load_measurements(measurement_paths=measurement_paths, sensor=sensor, quantity=quantity)

    rows: list[dict[str, object]] = []
    result_map: dict[str, object] = {}
    for horizon_minutes in horizons_minutes:
        horizon_index = horizon_minutes // 30 - 1
        try:
            cnn_aligned_df = _align_cnn_predictions(
                replay_sample_timestamps_utc=replay_sample_timestamps_utc,
                replay_sample_timestamps_local=replay_sample_timestamps_local,
                predicted_values=predictions[replay_indices, horizon_index],
                measurements_df=measurements_df,
                horizon_minutes=horizon_minutes,
                merge_tolerance_minutes=merge_tolerance_minutes,
            )
            _, baseline_mae, baseline_median_ae = _compute_baseline_same_grid_metrics(
                replay_timestamps_utc=pd.Series(replay_sample_timestamps_utc),
                baseline_log=baseline_log,
                measurements_df=measurements_df,
                horizon_minutes=horizon_minutes,
                merge_tolerance_minutes=merge_tolerance_minutes,
            )
        except ValueError:
            continue

        cnn_mae = float(cnn_aligned_df["absolute_error"].mean())
        cnn_median_ae = float(cnn_aligned_df["absolute_error"].median())
        row = {
            "window": name,
            "start_date": str(start_date.date()),
            "end_date": str(end_date.date()),
            "overlaps_recent_local_adaptation": overlaps_recent_local_adaptation,
            "horizon_minutes": horizon_minutes,
            "paired_rows": int(len(cnn_aligned_df)),
            "cnn_mae_c": cnn_mae,
            "baseline_mae_c": float(baseline_mae),
            "delta_cnn_minus_baseline_c": cnn_mae - float(baseline_mae),
            "cnn_median_ae_c": cnn_median_ae,
            "baseline_median_ae_c": float(baseline_median_ae),
        }
        rows.append(row)
        result_map[str(horizon_minutes)] = {
            "paired_rows": row["paired_rows"],
            "cnn_mae_c": row["cnn_mae_c"],
            "baseline_mae_c": row["baseline_mae_c"],
            "delta_cnn_minus_baseline_c": row["delta_cnn_minus_baseline_c"],
            "cnn_median_ae_c": row["cnn_median_ae_c"],
            "baseline_median_ae_c": row["baseline_median_ae_c"],
        }

    return rows, {
        "window": name,
        "start_date": str(start_date.date()),
        "end_date": str(end_date.date()),
        "overlaps_recent_local_adaptation": overlaps_recent_local_adaptation,
        "sample_count": int(len(replay_indices)),
        "measurement_files": [path.name for path in measurement_paths],
        "results": result_map,
    }


def _daily_walkforward_rows(
    *,
    recent_days: int,
    adaptation_days: int,
    latest_date: pd.Timestamp,
    horizons_minutes: list[int],
    sample_timestamps_local: pd.DatetimeIndex,
    sample_timestamps_utc: pd.DatetimeIndex,
    predictions: np.ndarray,
    baseline_log: pd.DataFrame,
    measurements_dir: Path,
    merge_tolerance_minutes: int,
    sensor: str,
    quantity: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    start_date = latest_date - pd.Timedelta(days=recent_days - 1)
    blind_cutoff = latest_date - pd.Timedelta(days=adaptation_days - 1)

    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for day in pd.date_range(start_date, latest_date, freq="D"):
        overlaps_recent_local_adaptation = bool(day >= blind_cutoff)
        rows, summary = _score_window(
            name=f"daily_walkforward_{day.date().isoformat()}",
            start_date=day,
            end_date=day,
            overlaps_recent_local_adaptation=overlaps_recent_local_adaptation,
            horizons_minutes=horizons_minutes,
            sample_timestamps_local=sample_timestamps_local,
            sample_timestamps_utc=sample_timestamps_utc,
            predictions=predictions,
            baseline_log=baseline_log,
            measurements_dir=measurements_dir,
            merge_tolerance_minutes=merge_tolerance_minutes,
            sensor=sensor,
            quantity=quantity,
        )
        if rows:
            all_rows.extend(rows)
            summaries.append(summary)
    return all_rows, summaries


def _startup_slice_rows(
    *,
    recent_days: int,
    adaptation_days: int,
    latest_date: pd.Timestamp,
    startup_hours: int,
    horizons_minutes: list[int],
    sample_timestamps_local: pd.DatetimeIndex,
    sample_timestamps_utc: pd.DatetimeIndex,
    predictions: np.ndarray,
    baseline_log: pd.DataFrame,
    measurements_dir: Path,
    merge_tolerance_minutes: int,
    sensor: str,
    quantity: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    start_date = latest_date - pd.Timedelta(days=recent_days - 1)
    blind_cutoff = latest_date - pd.Timedelta(days=adaptation_days - 1)

    all_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for day in pd.date_range(start_date, latest_date, freq="D"):
        day_mask = sample_timestamps_local.normalize() == day
        day_timestamps = sample_timestamps_local[day_mask]
        if len(day_timestamps) == 0:
            continue
        generation_start_local = day_timestamps.min()
        generation_end_local = generation_start_local + pd.Timedelta(hours=startup_hours)
        overlaps_recent_local_adaptation = bool(day >= blind_cutoff)
        rows, summary = _score_window(
            name=f"startup_slice_{day.date().isoformat()}",
            start_date=day,
            end_date=day,
            overlaps_recent_local_adaptation=overlaps_recent_local_adaptation,
            horizons_minutes=horizons_minutes,
            sample_timestamps_local=sample_timestamps_local,
            sample_timestamps_utc=sample_timestamps_utc,
            predictions=predictions,
            baseline_log=baseline_log,
            measurements_dir=measurements_dir,
            merge_tolerance_minutes=merge_tolerance_minutes,
            sensor=sensor,
            quantity=quantity,
            generation_start_local=generation_start_local,
            generation_end_local=generation_end_local,
        )
        if rows:
            all_rows.extend(rows)
            summaries.append(summary)
    return all_rows, summaries


def _summarize_horizon_passes(df: pd.DataFrame, horizon_minutes: int) -> tuple[bool | None, float | None, float | None, int]:
    if df.empty:
        return None, None, None, 0
    horizon_df = df[df["horizon_minutes"] == horizon_minutes]
    if horizon_df.empty:
        return None, None, None, 0
    deltas = horizon_df["delta_cnn_minus_baseline_c"].astype(float)
    return bool((deltas < 0.0).all()), float(deltas.mean()), float(deltas.max()), int(len(horizon_df))


def _summary_sample_count_by_window(summaries: list[dict[str, object]]) -> dict[str, int]:
    sample_count_by_window: dict[str, int] = {}
    for summary in summaries:
        window_name = str(summary.get("window"))
        sample_count_by_window[window_name] = int(summary.get("sample_count", 0))
    return sample_count_by_window




def _build_gate(
    *,
    recent_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    startup_df: pd.DataFrame,
    daily_summaries: list[dict[str, object]],
    full_day_sample_count: int,
    guard_horizons_minutes: list[int],
    short_horizon_guard_margin_c: float,
) -> dict[str, object]:
    gate = {
        "target_horizon_minutes": 360,
        "full_day_sample_count": full_day_sample_count,
        "blind_recent_window_pass": None,
        "blind_recent_window_delta_c": None,
        "blind_daily_mean_pass": None,
        "blind_daily_mean_delta_c": None,
        "blind_daily_all_pass": None,
        "blind_daily_worst_delta_c": None,
        "blind_daily_blind_day_count": 0,
        "blind_daily_partial_day_count": 0,
        "blind_daily_excluded_partial_windows": [],
        "blind_startup_mean_pass": None,
        "blind_startup_mean_delta_c": None,
        "blind_startup_all_pass": None,
        "blind_startup_worst_delta_c": None,
        "blind_startup_blind_day_count": 0,
        "short_horizon_guards": {},
        "overall_pass": False,
    }

    blind_recent_360 = recent_df[
        (recent_df["window"] == "blind_recent_excluding_local_adaptation")
        & (recent_df["horizon_minutes"] == 360)
    ]
    if not blind_recent_360.empty:
        blind_delta = float(blind_recent_360.iloc[0]["delta_cnn_minus_baseline_c"])
        gate["blind_recent_window_delta_c"] = blind_delta
        gate["blind_recent_window_pass"] = blind_delta < 0.0

    daily_sample_count_by_window = _summary_sample_count_by_window(daily_summaries)
    blind_daily_candidate = daily_df[
        (daily_df["overlaps_recent_local_adaptation"] == False)
        & (daily_df["horizon_minutes"] == 360)
    ]
    blind_daily_full_history_window_names = [
        window_name
        for window_name, sample_count in daily_sample_count_by_window.items()
        if sample_count >= full_day_sample_count
    ]
    blind_daily = blind_daily_candidate[
        blind_daily_candidate["window"].isin(blind_daily_full_history_window_names)
    ]
    excluded_partial_windows = sorted(
        {
            str(window_name)
            for window_name, sample_count in daily_sample_count_by_window.items()
            if sample_count < full_day_sample_count
            and window_name in set(blind_daily_candidate["window"].astype(str))
        }
    )
    all_pass, mean_delta, worst_delta, count = _summarize_horizon_passes(blind_daily, 360)
    gate["blind_daily_all_pass"] = all_pass
    gate["blind_daily_mean_delta_c"] = mean_delta
    gate["blind_daily_mean_pass"] = None if mean_delta is None else mean_delta < 0.0
    gate["blind_daily_worst_delta_c"] = worst_delta
    gate["blind_daily_blind_day_count"] = count
    gate["blind_daily_partial_day_count"] = len(excluded_partial_windows)
    gate["blind_daily_excluded_partial_windows"] = excluded_partial_windows

    blind_startup = startup_df[
        (startup_df["overlaps_recent_local_adaptation"] == False)
        & (startup_df["horizon_minutes"] == 360)
    ]
    startup_all_pass, startup_mean_delta, startup_worst_delta, startup_count = _summarize_horizon_passes(
        blind_startup, 360
    )
    gate["blind_startup_all_pass"] = startup_all_pass
    gate["blind_startup_mean_delta_c"] = startup_mean_delta
    gate["blind_startup_mean_pass"] = None if startup_mean_delta is None else startup_mean_delta < 0.0
    gate["blind_startup_worst_delta_c"] = startup_worst_delta
    gate["blind_startup_blind_day_count"] = startup_count

    required_checks: list[bool] = []
    for horizon_minutes in guard_horizons_minutes:
        guard_df = recent_df[
            (recent_df["window"] == "blind_recent_excluding_local_adaptation")
            & (recent_df["horizon_minutes"] == horizon_minutes)
        ]
        if guard_df.empty:
            gate["short_horizon_guards"][str(horizon_minutes)] = {
                "pass": None,
                "delta_c": None,
                "margin_c": short_horizon_guard_margin_c,
            }
            continue
        delta = float(guard_df.iloc[0]["delta_cnn_minus_baseline_c"])
        guard_pass = delta <= short_horizon_guard_margin_c
        gate["short_horizon_guards"][str(horizon_minutes)] = {
            "pass": guard_pass,
            "delta_c": delta,
            "margin_c": short_horizon_guard_margin_c,
        }
        required_checks.append(guard_pass)

    for check_name in (
        "blind_recent_window_pass",
        "blind_daily_mean_pass",
        "blind_daily_all_pass",
        "blind_startup_mean_pass",
        "blind_startup_all_pass",
    ):
        value = gate[check_name]
        required_checks.append(bool(value))

    gate["overall_pass"] = bool(required_checks) and all(required_checks)
    return gate

def main() -> int:
    args = build_argument_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    latest_date = _latest_measurement_date(args.measurements_dir)
    adaptation_days = _adaptation_window_days(args.training_summary_path)
    recent_start = latest_date - pd.Timedelta(days=args.recent_days - 1)
    blind_end = latest_date - pd.Timedelta(days=adaptation_days)
    effective_daily_days = max(args.daily_days, adaptation_days + 7)

    feature_columns = tuple(args.feature_columns) if args.feature_columns else None
    resolved_scaler_path = _resolve_scaler_path(args.training_summary_path, args.scaler_path)
    print(f"Building local-only dataset with scaler {resolved_scaler_path}...", flush=True)
    config, dataset = _build_dataset(args.measurements_dir, feature_columns, resolved_scaler_path)
    full_day_sample_count = _samples_per_day(config.resample_frequency)
    sample_timestamps_local, sample_timestamps_utc = _sample_timestamps(dataset, config)
    baseline_log = pd.read_csv(
        args.measurements_dir / "baseline_forecast.log",
        parse_dates=["timestamp_iso8601"],
        on_bad_lines="skip",
    )
    board_like_baseline_log = _deduplicate_baseline_log(baseline_log)

    evaluation_start = min(recent_start, latest_date - pd.Timedelta(days=effective_daily_days - 1))
    relevant_mask = (
        (sample_timestamps_local.normalize() >= evaluation_start)
        & (sample_timestamps_local.normalize() <= latest_date)
    )
    sample_timestamps_local = sample_timestamps_local[relevant_mask]
    sample_timestamps_utc = sample_timestamps_utc[relevant_mask]
    relevant_inputs = dataset.input_sequences[relevant_mask]
    print(
        f"Loaded {len(relevant_inputs)} evaluation samples from {evaluation_start.date()} to {latest_date.date()}.",
        flush=True,
    )

    runtime_model_path = _resolve_runtime_model_path(args.model_path, args.training_summary_path)
    print(f"Loading model artifact from {runtime_model_path}...", flush=True)
    print("Running model predictions for evaluation subset...", flush=True)
    predictions, runtime_kind = _predict(runtime_model_path, relevant_inputs)
    print(f"Model runtime: {runtime_kind}", flush=True)
    print("Predictions complete. Scoring windows...", flush=True)

    recent_windows = [
        EvaluationWindow(
            name="inclusive_last_recent_days",
            start_date=recent_start,
            end_date=latest_date,
            overlaps_recent_local_adaptation=True,
        )
    ]
    if blind_end >= recent_start:
        recent_windows.append(
            EvaluationWindow(
                name="blind_recent_excluding_local_adaptation",
                start_date=recent_start,
                end_date=blind_end,
                overlaps_recent_local_adaptation=False,
            )
        )

    recent_rows: list[dict[str, object]] = []
    recent_window_summaries: list[dict[str, object]] = []
    for window in recent_windows:
        rows, summary = _score_window(
            name=window.name,
            start_date=window.start_date,
            end_date=window.end_date,
            overlaps_recent_local_adaptation=window.overlaps_recent_local_adaptation,
            horizons_minutes=args.horizons_minutes,
            sample_timestamps_local=sample_timestamps_local,
            sample_timestamps_utc=sample_timestamps_utc,
            predictions=predictions,
            baseline_log=baseline_log,
            measurements_dir=args.measurements_dir,
            merge_tolerance_minutes=args.merge_tolerance_minutes,
            sensor=args.sensor,
            quantity=args.quantity,
        )
        recent_rows.extend(rows)
        recent_window_summaries.append(summary)

    board_like_recent_rows: list[dict[str, object]] = []
    board_like_recent_window_summaries: list[dict[str, object]] = []
    for window in recent_windows:
        rows, summary = _score_window(
            name=window.name,
            start_date=window.start_date,
            end_date=window.end_date,
            overlaps_recent_local_adaptation=window.overlaps_recent_local_adaptation,
            horizons_minutes=args.horizons_minutes,
            sample_timestamps_local=sample_timestamps_local,
            sample_timestamps_utc=sample_timestamps_utc,
            predictions=predictions,
            baseline_log=board_like_baseline_log,
            measurements_dir=args.measurements_dir,
            merge_tolerance_minutes=args.merge_tolerance_minutes,
            sensor=args.sensor,
            quantity=args.quantity,
        )
        board_like_recent_rows.extend(rows)
        board_like_recent_window_summaries.append(summary)

    daily_rows, daily_summaries = _daily_walkforward_rows(
        recent_days=effective_daily_days,
        adaptation_days=adaptation_days,
        latest_date=latest_date,
        horizons_minutes=args.horizons_minutes,
        sample_timestamps_local=sample_timestamps_local,
        sample_timestamps_utc=sample_timestamps_utc,
        predictions=predictions,
        baseline_log=baseline_log,
        measurements_dir=args.measurements_dir,
        merge_tolerance_minutes=args.merge_tolerance_minutes,
        sensor=args.sensor,
        quantity=args.quantity,
    )
    board_like_daily_rows, board_like_daily_summaries = _daily_walkforward_rows(
        recent_days=effective_daily_days,
        adaptation_days=adaptation_days,
        latest_date=latest_date,
        horizons_minutes=args.horizons_minutes,
        sample_timestamps_local=sample_timestamps_local,
        sample_timestamps_utc=sample_timestamps_utc,
        predictions=predictions,
        baseline_log=board_like_baseline_log,
        measurements_dir=args.measurements_dir,
        merge_tolerance_minutes=args.merge_tolerance_minutes,
        sensor=args.sensor,
        quantity=args.quantity,
    )
    startup_rows, startup_summaries = _startup_slice_rows(
        recent_days=effective_daily_days,
        adaptation_days=adaptation_days,
        latest_date=latest_date,
        startup_hours=args.startup_hours,
        horizons_minutes=args.horizons_minutes,
        sample_timestamps_local=sample_timestamps_local,
        sample_timestamps_utc=sample_timestamps_utc,
        predictions=predictions,
        baseline_log=baseline_log,
        measurements_dir=args.measurements_dir,
        merge_tolerance_minutes=args.merge_tolerance_minutes,
        sensor=args.sensor,
        quantity=args.quantity,
    )

    board_like_startup_rows, board_like_startup_summaries = _startup_slice_rows(
        recent_days=effective_daily_days,
        adaptation_days=adaptation_days,
        latest_date=latest_date,
        startup_hours=args.startup_hours,
        horizons_minutes=args.horizons_minutes,
        sample_timestamps_local=sample_timestamps_local,
        sample_timestamps_utc=sample_timestamps_utc,
        predictions=predictions,
        baseline_log=board_like_baseline_log,
        measurements_dir=args.measurements_dir,
        merge_tolerance_minutes=args.merge_tolerance_minutes,
        sensor=args.sensor,
        quantity=args.quantity,
    )

    recent_df = pd.DataFrame(recent_rows)
    daily_df = pd.DataFrame(daily_rows)
    startup_df = pd.DataFrame(startup_rows)
    board_like_recent_df = pd.DataFrame(board_like_recent_rows)
    board_like_daily_df = pd.DataFrame(board_like_daily_rows)
    board_like_startup_df = pd.DataFrame(board_like_startup_rows)

    recent_csv = args.output_dir / "preflash_recent_windows.csv"
    daily_csv = args.output_dir / "preflash_daily_walkforward.csv"
    startup_csv = args.output_dir / "preflash_startup_slices.csv"
    if not recent_df.empty:
        recent_df.to_csv(recent_csv, index=False)
    if not daily_df.empty:
        daily_df.to_csv(daily_csv, index=False)
    if not startup_df.empty:
        startup_df.to_csv(startup_csv, index=False)

    gate = {
        "target_horizon_minutes": 360,
        "full_day_sample_count": full_day_sample_count,
        "blind_recent_window_pass": None,
        "blind_recent_window_delta_c": None,
        "blind_daily_mean_pass": None,
        "blind_daily_mean_delta_c": None,
        "blind_daily_all_pass": None,
        "blind_daily_worst_delta_c": None,
        "blind_daily_blind_day_count": 0,
        "blind_daily_partial_day_count": 0,
        "blind_daily_excluded_partial_windows": [],
        "blind_startup_mean_pass": None,
        "blind_startup_mean_delta_c": None,
        "blind_startup_all_pass": None,
        "blind_startup_worst_delta_c": None,
        "blind_startup_blind_day_count": 0,
        "short_horizon_guards": {},
        "overall_pass": False,
    }

    blind_recent_360 = recent_df[
        (recent_df["window"] == "blind_recent_excluding_local_adaptation")
        & (recent_df["horizon_minutes"] == 360)
    ]
    if not blind_recent_360.empty:
        blind_delta = float(blind_recent_360.iloc[0]["delta_cnn_minus_baseline_c"])
        gate["blind_recent_window_delta_c"] = blind_delta
        gate["blind_recent_window_pass"] = blind_delta < 0.0

    daily_sample_count_by_window = _summary_sample_count_by_window(daily_summaries)
    blind_daily_candidate = daily_df[
        (daily_df["overlaps_recent_local_adaptation"] == False)
        & (daily_df["horizon_minutes"] == 360)
    ]
    blind_daily_full_history_window_names = [
        window_name
        for window_name, sample_count in daily_sample_count_by_window.items()
        if sample_count >= full_day_sample_count
    ]
    blind_daily = blind_daily_candidate[
        blind_daily_candidate["window"].isin(blind_daily_full_history_window_names)
    ]
    excluded_partial_windows = sorted(
        {
            str(window_name)
            for window_name, sample_count in daily_sample_count_by_window.items()
            if sample_count < full_day_sample_count
            and window_name in set(blind_daily_candidate["window"].astype(str))
        }
    )
    all_pass, mean_delta, worst_delta, count = _summarize_horizon_passes(blind_daily, 360)
    gate["blind_daily_all_pass"] = all_pass
    gate["blind_daily_mean_delta_c"] = mean_delta
    gate["blind_daily_mean_pass"] = None if mean_delta is None else mean_delta < 0.0
    gate["blind_daily_worst_delta_c"] = worst_delta
    gate["blind_daily_blind_day_count"] = count
    gate["blind_daily_partial_day_count"] = len(excluded_partial_windows)
    gate["blind_daily_excluded_partial_windows"] = excluded_partial_windows

    blind_startup = startup_df[
        (startup_df["overlaps_recent_local_adaptation"] == False)
        & (startup_df["horizon_minutes"] == 360)
    ]
    startup_all_pass, startup_mean_delta, startup_worst_delta, startup_count = _summarize_horizon_passes(
        blind_startup, 360
    )
    gate["blind_startup_all_pass"] = startup_all_pass
    gate["blind_startup_mean_delta_c"] = startup_mean_delta
    gate["blind_startup_mean_pass"] = None if startup_mean_delta is None else startup_mean_delta < 0.0
    gate["blind_startup_worst_delta_c"] = startup_worst_delta
    gate["blind_startup_blind_day_count"] = startup_count

    required_checks: list[bool] = []
    for horizon_minutes in args.guard_horizons_minutes:
        guard_df = recent_df[
            (recent_df["window"] == "blind_recent_excluding_local_adaptation")
            & (recent_df["horizon_minutes"] == horizon_minutes)
        ]
        if guard_df.empty:
            gate["short_horizon_guards"][str(horizon_minutes)] = {
                "pass": None,
                "delta_c": None,
                "margin_c": args.short_horizon_guard_margin_c,
            }
            continue
        delta = float(guard_df.iloc[0]["delta_cnn_minus_baseline_c"])
        guard_pass = delta <= args.short_horizon_guard_margin_c
        gate["short_horizon_guards"][str(horizon_minutes)] = {
            "pass": guard_pass,
            "delta_c": delta,
            "margin_c": args.short_horizon_guard_margin_c,
        }
        required_checks.append(guard_pass)

    for check_name in (
        "blind_recent_window_pass",
        "blind_daily_mean_pass",
        "blind_daily_all_pass",
        "blind_startup_mean_pass",
        "blind_startup_all_pass",
    ):
        value = gate[check_name]
        required_checks.append(bool(value))

    gate["overall_pass"] = bool(required_checks) and all(required_checks)

    board_like_recent_csv = args.output_dir / "preflash_board_like_recent_windows.csv"
    board_like_daily_csv = args.output_dir / "preflash_board_like_daily_walkforward.csv"
    board_like_startup_csv = args.output_dir / "preflash_board_like_startup_slices.csv"
    if not board_like_recent_df.empty:
        board_like_recent_df.to_csv(board_like_recent_csv, index=False)
    if not board_like_daily_df.empty:
        board_like_daily_df.to_csv(board_like_daily_csv, index=False)
    if not board_like_startup_df.empty:
        board_like_startup_df.to_csv(board_like_startup_csv, index=False)

    board_like_gate = _build_gate(
        recent_df=board_like_recent_df,
        daily_df=board_like_daily_df,
        startup_df=board_like_startup_df,
        daily_summaries=board_like_daily_summaries,
        full_day_sample_count=full_day_sample_count,
        guard_horizons_minutes=args.guard_horizons_minutes,
        short_horizon_guard_margin_c=args.short_horizon_guard_margin_c,
    )
    combined_overall_pass = bool(gate["overall_pass"]) and bool(board_like_gate["overall_pass"])

    summary = {
        "latest_measurement_date": str(latest_date.date()),
        "recent_start_date": str(recent_start.date()),
        "recent_local_adaptation_days": adaptation_days,
        "startup_hours": args.startup_hours,
        "effective_daily_days": effective_daily_days,
        "recent_windows": recent_window_summaries,
        "daily_walkforward": daily_summaries,
        "startup_slices": startup_summaries,
        "board_like_recent_windows": board_like_recent_window_summaries,
        "board_like_daily_walkforward": board_like_daily_summaries,
        "board_like_startup_slices": board_like_startup_summaries,
        "gate": gate,
        "board_like_gate": board_like_gate,
        "combined_overall_pass": combined_overall_pass,
        "artifacts": {
            "recent_windows_csv": str(recent_csv),
            "daily_walkforward_csv": str(daily_csv),
            "startup_slices_csv": str(startup_csv),
            "board_like_recent_windows_csv": str(board_like_recent_csv),
            "board_like_daily_walkforward_csv": str(board_like_daily_csv),
            "board_like_startup_slices_csv": str(board_like_startup_csv),
            "runtime_model_path": str(runtime_model_path),
            "board_like_baseline_rows": int(len(board_like_baseline_log)),
            "runtime_model_kind": runtime_kind,
            "scaler_path": str(resolved_scaler_path),
        },
    }
    summary_path = args.output_dir / "preflash_evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Latest measurement date: {latest_date.date()}")
    print(f"Recent start date: {recent_start.date()}")
    print(f"Recent local adaptation days: {adaptation_days}")
    print(f"Effective daily days: {effective_daily_days}")
    if not recent_df.empty:
        print("Recent windows:")
        print(recent_df.to_string(index=False))
    if not daily_df.empty:
        print("Daily walk-forward:")
        print(daily_df.to_string(index=False))
    if not startup_df.empty:
        print("Startup slices:")
        print(startup_df.to_string(index=False))
    if not board_like_recent_df.empty:
        print("Board-like recent windows:")
        print(board_like_recent_df.to_string(index=False))
    if not board_like_daily_df.empty:
        print("Board-like daily walk-forward:")
        print(board_like_daily_df.to_string(index=False))
    if not board_like_startup_df.empty:
        print("Board-like startup slices:")
        print(board_like_startup_df.to_string(index=False))
    print("Gate summary:")
    print(json.dumps(gate, indent=2))
    print("Board-like gate summary:")
    print(json.dumps(board_like_gate, indent=2))
    print(f"Combined overall pass: {combined_overall_pass}")
    print(f"Summary JSON: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





