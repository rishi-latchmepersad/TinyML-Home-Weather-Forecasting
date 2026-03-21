from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark_vs_measurements import _align_cnn_predictions
from .config import DEFAULT_MEASUREMENTS_DIRECTORY, DEFAULT_OUTPUT_DIRECTORY
from .evaluate_vs_measurements import (
    _compute_baseline_same_grid_metrics,
    _load_measurements,
    _select_measurement_paths,
)
from .preflash_evaluation import (
    _adaptation_window_days,
    _build_dataset,
    _build_gate,
    _deduplicate_baseline_log,
    _latest_measurement_date,
    _predict,
    _resolve_runtime_model_path,
    _resolve_scaler_path,
    _sample_timestamps,
    _window_mask,
    _samples_per_day,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Search baseline-heavy board-like blend schedules using cached aligned rows "
            "so candidate evaluation is fast enough to use before flashing."
        )
    )
    parser.add_argument("--measurements-dir", type=Path, default=DEFAULT_MEASUREMENTS_DIRECTORY)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_OUTPUT_DIRECTORY / "pruned_int8.tflite")
    parser.add_argument("--training-summary-path", type=Path, default=DEFAULT_OUTPUT_DIRECTORY / "training_summary.json")
    parser.add_argument("--scaler-path", type=Path, default=DEFAULT_OUTPUT_DIRECTORY / "feature_scaler.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY / "board_like_blend_search")
    parser.add_argument("--recent-days", type=int, default=21)
    parser.add_argument("--daily-days", type=int, default=28)
    parser.add_argument("--startup-hours", type=int, default=6)
    parser.add_argument("--merge-tolerance-minutes", type=int, default=15)
    parser.add_argument("--horizons-minutes", nargs="*", type=int, default=[30, 60, 180, 360])
    parser.add_argument("--guard-horizons-minutes", nargs="*", type=int, default=[60, 180])
    parser.add_argument("--short-horizon-guard-margin-c", type=float, default=0.25)
    parser.add_argument("--sensor", default="bme280")
    parser.add_argument("--quantity", default="temperature_c")
    parser.add_argument("--alpha-short-values", nargs="*", type=float, default=[0.03, 0.04, 0.05, 0.06])
    parser.add_argument("--alpha-long-values", nargs="*", type=float, default=[0.0, 0.005, 0.01])
    parser.add_argument("--bias-short-values", nargs="*", type=float, default=[0.0, 0.005])
    parser.add_argument("--bias-long-values", nargs="*", type=float, default=[0.0, 0.005, 0.01, 0.015, 0.02])
    return parser


def _build_window_definitions(
    *,
    latest_date: pd.Timestamp,
    recent_days: int,
    daily_days: int,
    adaptation_days: int,
    startup_hours: int,
    sample_timestamps_local: pd.DatetimeIndex,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], int]:
    recent_start = latest_date - pd.Timedelta(days=recent_days - 1)
    effective_daily_days = max(daily_days, adaptation_days + 7)
    blind_recent_end = latest_date - pd.Timedelta(days=adaptation_days)
    blind_cutoff = latest_date - pd.Timedelta(days=adaptation_days - 1)

    recent_windows: list[dict[str, object]] = [
        {
            "window": "inclusive_last_recent_days",
            "start_date": recent_start,
            "end_date": latest_date,
            "overlaps_recent_local_adaptation": True,
            "generation_start_local": None,
            "generation_end_local": None,
            "kind": "recent",
        }
    ]
    if blind_recent_end >= recent_start:
        recent_windows.append(
            {
                "window": "blind_recent_excluding_local_adaptation",
                "start_date": recent_start,
                "end_date": blind_recent_end,
                "overlaps_recent_local_adaptation": False,
                "generation_start_local": None,
                "generation_end_local": None,
                "kind": "recent",
            }
        )

    daily_windows: list[dict[str, object]] = []
    startup_windows: list[dict[str, object]] = []
    daily_start = latest_date - pd.Timedelta(days=effective_daily_days - 1)
    for day in pd.date_range(daily_start, latest_date, freq="D"):
        overlaps = bool(day >= blind_cutoff)
        daily_windows.append(
            {
                "window": f"daily_walkforward_{day.date().isoformat()}",
                "start_date": day,
                "end_date": day,
                "overlaps_recent_local_adaptation": overlaps,
                "generation_start_local": None,
                "generation_end_local": None,
                "kind": "daily",
            }
        )
        day_mask = sample_timestamps_local.normalize() == day
        day_timestamps = sample_timestamps_local[day_mask]
        if len(day_timestamps) == 0:
            continue
        generation_start_local = day_timestamps.min()
        generation_end_local = generation_start_local + pd.Timedelta(hours=startup_hours)
        startup_windows.append(
            {
                "window": f"startup_slice_{day.date().isoformat()}",
                "start_date": day,
                "end_date": day,
                "overlaps_recent_local_adaptation": overlaps,
                "generation_start_local": generation_start_local,
                "generation_end_local": generation_end_local,
                "kind": "startup",
            }
        )
    return recent_windows, daily_windows, startup_windows, effective_daily_days


def _cache_window_rows(
    *,
    window: dict[str, object],
    horizons_minutes: list[int],
    sample_timestamps_local: pd.DatetimeIndex,
    sample_timestamps_utc: pd.DatetimeIndex,
    predictions: np.ndarray,
    baseline_log: pd.DataFrame,
    measurements_dir: Path,
    merge_tolerance_minutes: int,
    sensor: str,
    quantity: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    baseline_timestamps_utc = pd.to_datetime(baseline_log["timestamp_iso8601"], utc=True, errors="coerce").dropna()
    mask = _window_mask(
        sample_timestamps_local=sample_timestamps_local,
        sample_timestamps_utc=sample_timestamps_utc,
        start_date=window["start_date"],
        end_date=window["end_date"],
        baseline_start_utc=baseline_timestamps_utc.min(),
        baseline_end_utc=baseline_timestamps_utc.max(),
        generation_start_local=window["generation_start_local"],
        generation_end_local=window["generation_end_local"],
    )
    replay_indices = np.flatnonzero(mask)
    summary = {
        "window": window["window"],
        "sample_count": int(len(replay_indices)),
    }
    if len(replay_indices) == 0:
        return pd.DataFrame(), summary

    replay_sample_timestamps_local = sample_timestamps_local[replay_indices]
    replay_sample_timestamps_utc = sample_timestamps_utc[replay_indices]
    max_horizon_delta = pd.Timedelta(minutes=max(horizons_minutes))
    try:
        measurement_paths = _select_measurement_paths(
            measurements_dir,
            pd.Series(replay_sample_timestamps_utc + max_horizon_delta),
        )
    except FileNotFoundError:
        return pd.DataFrame(), summary

    measurements_df = _load_measurements(
        measurement_paths=measurement_paths,
        sensor=sensor,
        quantity=quantity,
    )

    rows: list[pd.DataFrame] = []
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
            baseline_aligned_df, baseline_mae, baseline_median_ae = _compute_baseline_same_grid_metrics(
                replay_timestamps_utc=pd.Series(replay_sample_timestamps_utc),
                baseline_log=baseline_log,
                measurements_df=measurements_df,
                horizon_minutes=horizon_minutes,
                merge_tolerance_minutes=merge_tolerance_minutes,
            )
        except ValueError:
            continue

        paired_df = cnn_aligned_df.merge(
            baseline_aligned_df.loc[:, ["replay_timestamp", "predicted_temperature_c"]].rename(
                columns={
                    "replay_timestamp": "timestamp_forecast_gen",
                    "predicted_temperature_c": "baseline_predicted_temperature_c",
                }
            ),
            on="timestamp_forecast_gen",
            how="inner",
        )
        if paired_df.empty:
            continue

        paired_df = paired_df.assign(
            window=window["window"],
            start_date=str(window["start_date"].date()),
            end_date=str(window["end_date"].date()),
            overlaps_recent_local_adaptation=window["overlaps_recent_local_adaptation"],
            horizon_minutes=horizon_minutes,
            baseline_mae_c=float(baseline_mae),
            baseline_median_ae_c=float(baseline_median_ae),
            kind=window["kind"],
        )
        rows.append(paired_df)

    if not rows:
        return pd.DataFrame(), summary
    return pd.concat(rows, ignore_index=True), summary


def _schedule_value(horizon_minutes: int, min_horizon: int, max_horizon: int, start: float, end: float) -> float:
    if max_horizon == min_horizon:
        return float(end)
    fraction = (float(horizon_minutes) - float(min_horizon)) / (float(max_horizon) - float(min_horizon))
    return float(start + fraction * (end - start))


def _score_cached_rows(
    *,
    cached_rows: pd.DataFrame,
    alpha_short: float,
    alpha_long: float,
    bias_short: float,
    bias_long: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if cached_rows.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    working = cached_rows.copy()
    min_horizon = int(working["horizon_minutes"].min())
    max_horizon = int(working["horizon_minutes"].max())

    working["alpha"] = working["horizon_minutes"].apply(
        lambda h: _schedule_value(int(h), min_horizon, max_horizon, alpha_short, alpha_long)
    )
    working["bias_c"] = working["horizon_minutes"].apply(
        lambda h: _schedule_value(int(h), min_horizon, max_horizon, bias_short, bias_long)
    )
    working["final_predicted_temperature_c"] = (
        working["baseline_predicted_temperature_c"]
        + working["alpha"] * (working["predicted_temperature_c"] - working["baseline_predicted_temperature_c"])
        - working["bias_c"]
    )
    working["absolute_error"] = (
        working["final_predicted_temperature_c"] - working["measured_temperature_c"]
    ).abs()

    grouped = (
        working.groupby(
            ["kind", "window", "start_date", "end_date", "overlaps_recent_local_adaptation", "horizon_minutes"],
            as_index=False,
        )
        .agg(
            paired_rows=("absolute_error", "size"),
            cnn_mae_c=("absolute_error", "mean"),
            cnn_median_ae_c=("absolute_error", "median"),
            baseline_mae_c=("baseline_mae_c", "first"),
            baseline_median_ae_c=("baseline_median_ae_c", "first"),
        )
    )
    grouped["delta_cnn_minus_baseline_c"] = grouped["cnn_mae_c"] - grouped["baseline_mae_c"]

    recent_df = grouped[grouped["kind"] == "recent"].drop(columns=["kind"]).reset_index(drop=True)
    daily_df = grouped[grouped["kind"] == "daily"].drop(columns=["kind"]).reset_index(drop=True)
    startup_df = grouped[grouped["kind"] == "startup"].drop(columns=["kind"]).reset_index(drop=True)
    return recent_df, daily_df, startup_df


def main() -> int:
    args = build_argument_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    latest_date = _latest_measurement_date(args.measurements_dir)
    adaptation_days = _adaptation_window_days(args.training_summary_path)
    resolved_scaler_path = _resolve_scaler_path(args.training_summary_path, args.scaler_path)
    config, dataset = _build_dataset(args.measurements_dir, None, resolved_scaler_path)
    sample_timestamps_local, sample_timestamps_utc = _sample_timestamps(dataset, config)
    recent_windows, daily_windows, startup_windows, effective_daily_days = _build_window_definitions(
        latest_date=latest_date,
        recent_days=args.recent_days,
        daily_days=args.daily_days,
        adaptation_days=adaptation_days,
        startup_hours=args.startup_hours,
        sample_timestamps_local=sample_timestamps_local,
    )

    evaluation_start = min(
        latest_date - pd.Timedelta(days=args.recent_days - 1),
        latest_date - pd.Timedelta(days=effective_daily_days - 1),
    )
    relevant_mask = (
        (sample_timestamps_local.normalize() >= evaluation_start)
        & (sample_timestamps_local.normalize() <= latest_date)
    )
    sample_timestamps_local = sample_timestamps_local[relevant_mask]
    sample_timestamps_utc = sample_timestamps_utc[relevant_mask]
    relevant_inputs = dataset.input_sequences[relevant_mask]

    runtime_model_path = _resolve_runtime_model_path(args.model_path, args.training_summary_path)
    predictions, runtime_kind = _predict(runtime_model_path, relevant_inputs)

    baseline_log = pd.read_csv(
        args.measurements_dir / "baseline_forecast.log",
        parse_dates=["timestamp_iso8601"],
        on_bad_lines="skip",
    )
    board_like_baseline_log = _deduplicate_baseline_log(baseline_log)

    all_windows = recent_windows + daily_windows + startup_windows
    cached_frames: list[pd.DataFrame] = []
    daily_summaries: list[dict[str, object]] = []
    for window in all_windows:
        window_rows, summary = _cache_window_rows(
            window=window,
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
        if not window_rows.empty:
            cached_frames.append(window_rows)
        if window["kind"] == "daily":
            daily_summaries.append(summary)

    cached_rows = pd.concat(cached_frames, ignore_index=True) if cached_frames else pd.DataFrame()
    full_day_sample_count = _samples_per_day(config.resample_frequency)

    candidate_rows: list[dict[str, object]] = []
    best_summary: dict[str, object] | None = None
    best_rank: tuple[int, float, float, float] | None = None
    best_recent_df = pd.DataFrame()
    best_daily_df = pd.DataFrame()
    best_startup_df = pd.DataFrame()

    for alpha_short in args.alpha_short_values:
        for alpha_long in args.alpha_long_values:
            if alpha_long > alpha_short:
                continue
            for bias_short in args.bias_short_values:
                for bias_long in args.bias_long_values:
                    recent_df, daily_df, startup_df = _score_cached_rows(
                        cached_rows=cached_rows,
                        alpha_short=alpha_short,
                        alpha_long=alpha_long,
                        bias_short=bias_short,
                        bias_long=bias_long,
                    )
                    gate = _build_gate(
                        recent_df=recent_df,
                        daily_df=daily_df,
                        startup_df=startup_df,
                        daily_summaries=daily_summaries,
                        full_day_sample_count=full_day_sample_count,
                        guard_horizons_minutes=args.guard_horizons_minutes,
                        short_horizon_guard_margin_c=args.short_horizon_guard_margin_c,
                    )
                    record = {
                        "alpha_short": alpha_short,
                        "alpha_long": alpha_long,
                        "bias_short": bias_short,
                        "bias_long": bias_long,
                        "overall_pass": gate["overall_pass"],
                        "blind_recent_360_delta_c": gate["blind_recent_window_delta_c"],
                        "blind_daily_mean_360_delta_c": gate["blind_daily_mean_delta_c"],
                        "blind_daily_worst_360_delta_c": gate["blind_daily_worst_delta_c"],
                        "blind_startup_mean_360_delta_c": gate["blind_startup_mean_delta_c"],
                        "blind_startup_worst_360_delta_c": gate["blind_startup_worst_delta_c"],
                        "guard60_delta_c": gate["short_horizon_guards"].get("60", {}).get("delta_c"),
                        "guard180_delta_c": gate["short_horizon_guards"].get("180", {}).get("delta_c"),
                    }
                    candidate_rows.append(record)

                    rank = (
                        0 if gate["overall_pass"] else 1,
                        float(gate["blind_daily_worst_delta_c"] if gate["blind_daily_worst_delta_c"] is not None else 1e9),
                        float(gate["blind_startup_worst_delta_c"] if gate["blind_startup_worst_delta_c"] is not None else 1e9),
                        float(gate["blind_recent_window_delta_c"] if gate["blind_recent_window_delta_c"] is not None else 1e9),
                    )
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                        best_recent_df = recent_df.copy()
                        best_daily_df = daily_df.copy()
                        best_startup_df = startup_df.copy()
                        best_summary = {
                            "alpha_short": alpha_short,
                            "alpha_long": alpha_long,
                            "bias_short": bias_short,
                            "bias_long": bias_long,
                            "gate": gate,
                        }

    candidates_df = pd.DataFrame(candidate_rows).sort_values(
        ["overall_pass", "blind_daily_worst_360_delta_c", "blind_startup_worst_360_delta_c", "blind_recent_360_delta_c"],
        ascending=[False, True, True, True],
    )

    candidates_csv = args.output_dir / "candidates.csv"
    candidates_df.to_csv(candidates_csv, index=False)
    if not best_recent_df.empty:
        best_recent_df.to_csv(args.output_dir / "best_recent_windows.csv", index=False)
    if not best_daily_df.empty:
        best_daily_df.to_csv(args.output_dir / "best_daily_walkforward.csv", index=False)
    if not best_startup_df.empty:
        best_startup_df.to_csv(args.output_dir / "best_startup_slices.csv", index=False)

    summary = {
        "latest_measurement_date": str(latest_date.date()),
        "recent_local_adaptation_days": adaptation_days,
        "effective_daily_days": effective_daily_days,
        "runtime_model_path": str(runtime_model_path),
        "runtime_model_kind": runtime_kind,
        "scaler_path": str(resolved_scaler_path),
        "best_candidate": best_summary,
        "artifacts": {
            "candidates_csv": str(candidates_csv),
            "best_recent_windows_csv": str(args.output_dir / "best_recent_windows.csv"),
            "best_daily_walkforward_csv": str(args.output_dir / "best_daily_walkforward.csv"),
            "best_startup_slices_csv": str(args.output_dir / "best_startup_slices.csv"),
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(candidates_df.head(20).to_string(index=False))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

