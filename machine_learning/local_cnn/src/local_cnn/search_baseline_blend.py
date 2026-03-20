from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark_vs_measurements import _align_cnn_predictions
from .config import DEFAULT_MEASUREMENTS_DIRECTORY, DEFAULT_OUTPUT_DIRECTORY
from .evaluate_vs_measurements import _compute_baseline_same_grid_metrics, _load_measurements, _select_measurement_paths
from .preflash_evaluation import (
    _adaptation_window_days,
    _build_dataset,
    _latest_measurement_date,
    _predict,
    _resolve_runtime_model_path,
    _resolve_scaler_path,
    _sample_timestamps,
)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search a baseline/model blending schedule against the pre-flash gate.")
    parser.add_argument("--measurements-dir", type=Path, default=DEFAULT_MEASUREMENTS_DIRECTORY)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_OUTPUT_DIRECTORY / "best_pruned_model.keras")
    parser.add_argument("--training-summary-path", type=Path, default=DEFAULT_OUTPUT_DIRECTORY / "training_summary.json")
    parser.add_argument("--scaler-path", type=Path, default=DEFAULT_OUTPUT_DIRECTORY / "feature_scaler.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY / "blend_search")
    parser.add_argument("--recent-days", type=int, default=21)
    parser.add_argument("--daily-days", type=int, default=28)
    parser.add_argument("--startup-hours", type=int, default=6)
    parser.add_argument("--merge-tolerance-minutes", type=int, default=15)
    parser.add_argument("--feature-column", action="append", dest="feature_columns")
    parser.add_argument("--horizons-minutes", nargs="*", type=int, default=[30, 60, 180, 360])
    parser.add_argument("--guard-horizons-minutes", nargs="*", type=int, default=[60, 180])
    parser.add_argument("--short-horizon-guard-margin-c", type=float, default=0.25)
    parser.add_argument("--sensor", default="bme280")
    parser.add_argument("--quantity", default="temperature_c")
    return parser


def _blend_alpha(horizon_minutes: int, min_horizon: int, max_horizon: int, alpha_short: float, alpha_long: float) -> float:
    if max_horizon == min_horizon:
        return float(alpha_long)
    fraction = (float(horizon_minutes) - float(min_horizon)) / (float(max_horizon) - float(min_horizon))
    return float(alpha_short + fraction * (alpha_long - alpha_short))


def _score_window_blended(
    *,
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
    alpha_short: float,
    alpha_long: float,
    generation_start_local: pd.Timestamp | None = None,
    generation_end_local: pd.Timestamp | None = None,
) -> list[dict[str, object]]:
    baseline_timestamps_utc = pd.to_datetime(baseline_log["timestamp_iso8601"], utc=True, errors="coerce").dropna()
    local_dates = sample_timestamps_local.normalize()
    mask = (
        (local_dates >= start_date)
        & (local_dates <= end_date)
        & (sample_timestamps_utc >= baseline_timestamps_utc.min())
        & (sample_timestamps_utc <= baseline_timestamps_utc.max())
    )
    if generation_start_local is not None:
        mask &= sample_timestamps_local >= generation_start_local
    if generation_end_local is not None:
        mask &= sample_timestamps_local < generation_end_local

    replay_indices = np.flatnonzero(mask)
    if len(replay_indices) == 0:
        return []

    replay_sample_timestamps_local = sample_timestamps_local[replay_indices]
    replay_sample_timestamps_utc = sample_timestamps_utc[replay_indices]
    max_horizon_delta = pd.Timedelta(minutes=max(horizons_minutes))
    try:
        measurement_paths = _select_measurement_paths(
            measurements_dir,
            pd.Series(replay_sample_timestamps_utc + max_horizon_delta),
        )
    except FileNotFoundError:
        return []
    measurements_df = _load_measurements(measurement_paths=measurement_paths, sensor=sensor, quantity=quantity)

    rows: list[dict[str, object]] = []
    min_horizon = min(horizons_minutes)
    max_horizon = max(horizons_minutes)
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
            baseline_aligned_df, baseline_mae, _ = _compute_baseline_same_grid_metrics(
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

        alpha = _blend_alpha(horizon_minutes, min_horizon, max_horizon, alpha_short, alpha_long)
        paired_df["blended_predicted_temperature_c"] = (
            paired_df["baseline_predicted_temperature_c"]
            + alpha * (paired_df["predicted_temperature_c"] - paired_df["baseline_predicted_temperature_c"])
        )
        blended_errors = (paired_df["blended_predicted_temperature_c"] - paired_df["measured_temperature_c"]).abs()
        blended_mae = float(blended_errors.mean())

        rows.append(
            {
                "start_date": str(start_date.date()),
                "end_date": str(end_date.date()),
                "overlaps_recent_local_adaptation": overlaps_recent_local_adaptation,
                "horizon_minutes": horizon_minutes,
                "paired_rows": int(len(paired_df)),
                "cnn_mae_c": blended_mae,
                "baseline_mae_c": float(baseline_mae),
                "delta_cnn_minus_baseline_c": blended_mae - float(baseline_mae),
            }
        )
    return rows


def _gate_from_frames(recent_df: pd.DataFrame, daily_df: pd.DataFrame, startup_df: pd.DataFrame, guard_horizons: list[int], short_margin: float) -> dict[str, object]:
    gate: dict[str, object] = {
        "target_horizon_minutes": 360,
        "blind_recent_window_pass": None,
        "blind_recent_window_delta_c": None,
        "blind_daily_mean_pass": None,
        "blind_daily_mean_delta_c": None,
        "blind_daily_all_pass": None,
        "blind_daily_worst_delta_c": None,
        "blind_daily_blind_day_count": 0,
        "blind_startup_mean_pass": None,
        "blind_startup_mean_delta_c": None,
        "blind_startup_all_pass": None,
        "blind_startup_worst_delta_c": None,
        "blind_startup_blind_day_count": 0,
        "short_horizon_guards": {},
        "overall_pass": False,
    }

    blind_recent_360 = recent_df[(recent_df["window"] == "blind_recent_excluding_local_adaptation") & (recent_df["horizon_minutes"] == 360)]
    if not blind_recent_360.empty:
        delta = float(blind_recent_360.iloc[0]["delta_cnn_minus_baseline_c"])
        gate["blind_recent_window_delta_c"] = delta
        gate["blind_recent_window_pass"] = delta < 0.0

    blind_daily = daily_df[(daily_df["overlaps_recent_local_adaptation"] == False) & (daily_df["horizon_minutes"] == 360)]
    if not blind_daily.empty:
        deltas = blind_daily["delta_cnn_minus_baseline_c"].astype(float)
        gate["blind_daily_mean_delta_c"] = float(deltas.mean())
        gate["blind_daily_mean_pass"] = float(deltas.mean()) < 0.0
        gate["blind_daily_all_pass"] = bool((deltas < 0.0).all())
        gate["blind_daily_worst_delta_c"] = float(deltas.max())
        gate["blind_daily_blind_day_count"] = int(len(deltas))

    blind_startup = startup_df[(startup_df["overlaps_recent_local_adaptation"] == False) & (startup_df["horizon_minutes"] == 360)]
    if not blind_startup.empty:
        deltas = blind_startup["delta_cnn_minus_baseline_c"].astype(float)
        gate["blind_startup_mean_delta_c"] = float(deltas.mean())
        gate["blind_startup_mean_pass"] = float(deltas.mean()) < 0.0
        gate["blind_startup_all_pass"] = bool((deltas < 0.0).all())
        gate["blind_startup_worst_delta_c"] = float(deltas.max())
        gate["blind_startup_blind_day_count"] = int(len(deltas))

    checks: list[bool] = []
    for horizon in guard_horizons:
        guard_df = recent_df[(recent_df["window"] == "blind_recent_excluding_local_adaptation") & (recent_df["horizon_minutes"] == horizon)]
        if guard_df.empty:
            gate["short_horizon_guards"][str(horizon)] = {"pass": None, "delta_c": None, "margin_c": short_margin}
            continue
        delta = float(guard_df.iloc[0]["delta_cnn_minus_baseline_c"])
        passed = delta <= short_margin
        gate["short_horizon_guards"][str(horizon)] = {"pass": passed, "delta_c": delta, "margin_c": short_margin}
        checks.append(passed)

    for key in ("blind_recent_window_pass", "blind_daily_mean_pass", "blind_daily_all_pass", "blind_startup_mean_pass", "blind_startup_all_pass"):
        checks.append(bool(gate[key]))
    gate["overall_pass"] = bool(checks) and all(checks)
    return gate


def main() -> int:
    args = build_argument_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    latest_date = _latest_measurement_date(args.measurements_dir)
    adaptation_days = _adaptation_window_days(args.training_summary_path)
    recent_start = latest_date - pd.Timedelta(days=args.recent_days - 1)
    blind_end = latest_date - pd.Timedelta(days=adaptation_days)

    feature_columns = tuple(args.feature_columns) if args.feature_columns else None
    resolved_scaler_path = _resolve_scaler_path(args.training_summary_path, args.scaler_path)
    config, dataset = _build_dataset(args.measurements_dir, feature_columns, resolved_scaler_path)
    sample_timestamps_local, sample_timestamps_utc = _sample_timestamps(dataset, config)
    baseline_log = pd.read_csv(args.measurements_dir / "baseline_forecast.log", parse_dates=["timestamp_iso8601"], on_bad_lines="skip")

    evaluation_start = min(recent_start, latest_date - pd.Timedelta(days=args.daily_days - 1))
    relevant_mask = (
        (sample_timestamps_local.normalize() >= evaluation_start)
        & (sample_timestamps_local.normalize() <= latest_date)
    )
    sample_timestamps_local = sample_timestamps_local[relevant_mask]
    sample_timestamps_utc = sample_timestamps_utc[relevant_mask]
    relevant_inputs = dataset.input_sequences[relevant_mask]

    runtime_model_path = _resolve_runtime_model_path(args.model_path, args.training_summary_path)
    predictions, runtime_kind = _predict(runtime_model_path, relevant_inputs)

    candidate_rows: list[dict[str, object]] = []
    best_summary: dict[str, object] | None = None
    best_rank: tuple[int, float, float] | None = None
    alpha_short_values = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 1.00]
    alpha_long_values = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]

    for alpha_short in alpha_short_values:
        for alpha_long in alpha_long_values:
            if alpha_long > alpha_short:
                continue

            recent_rows: list[dict[str, object]] = []
            daily_rows: list[dict[str, object]] = []
            startup_rows: list[dict[str, object]] = []

            inclusive_rows = _score_window_blended(
                start_date=recent_start,
                end_date=latest_date,
                overlaps_recent_local_adaptation=True,
                horizons_minutes=args.horizons_minutes,
                sample_timestamps_local=sample_timestamps_local,
                sample_timestamps_utc=sample_timestamps_utc,
                predictions=predictions,
                baseline_log=baseline_log,
                measurements_dir=args.measurements_dir,
                merge_tolerance_minutes=args.merge_tolerance_minutes,
                sensor=args.sensor,
                quantity=args.quantity,
                alpha_short=alpha_short,
                alpha_long=alpha_long,
            )
            for row in inclusive_rows:
                row["window"] = "inclusive_last_recent_days"
            recent_rows.extend(inclusive_rows)

            if blind_end >= recent_start:
                blind_rows = _score_window_blended(
                    start_date=recent_start,
                    end_date=blind_end,
                    overlaps_recent_local_adaptation=False,
                    horizons_minutes=args.horizons_minutes,
                    sample_timestamps_local=sample_timestamps_local,
                    sample_timestamps_utc=sample_timestamps_utc,
                    predictions=predictions,
                    baseline_log=baseline_log,
                    measurements_dir=args.measurements_dir,
                    merge_tolerance_minutes=args.merge_tolerance_minutes,
                    sensor=args.sensor,
                    quantity=args.quantity,
                    alpha_short=alpha_short,
                    alpha_long=alpha_long,
                )
                for row in blind_rows:
                    row["window"] = "blind_recent_excluding_local_adaptation"
                recent_rows.extend(blind_rows)

            start_date = latest_date - pd.Timedelta(days=args.daily_days - 1)
            blind_cutoff = latest_date - pd.Timedelta(days=adaptation_days - 1)
            for day in pd.date_range(start_date, latest_date, freq="D"):
                overlaps = bool(day >= blind_cutoff)
                day_rows = _score_window_blended(
                    start_date=day,
                    end_date=day,
                    overlaps_recent_local_adaptation=overlaps,
                    horizons_minutes=args.horizons_minutes,
                    sample_timestamps_local=sample_timestamps_local,
                    sample_timestamps_utc=sample_timestamps_utc,
                    predictions=predictions,
                    baseline_log=baseline_log,
                    measurements_dir=args.measurements_dir,
                    merge_tolerance_minutes=args.merge_tolerance_minutes,
                    sensor=args.sensor,
                    quantity=args.quantity,
                    alpha_short=alpha_short,
                    alpha_long=alpha_long,
                )
                for row in day_rows:
                    row["window"] = f"daily_walkforward_{day.date().isoformat()}"
                daily_rows.extend(day_rows)

                day_mask = sample_timestamps_local.normalize() == day
                day_timestamps = sample_timestamps_local[day_mask]
                if len(day_timestamps) > 0:
                    generation_start_local = day_timestamps.min()
                    generation_end_local = generation_start_local + pd.Timedelta(hours=args.startup_hours)
                    slice_rows = _score_window_blended(
                        start_date=day,
                        end_date=day,
                        overlaps_recent_local_adaptation=overlaps,
                        horizons_minutes=args.horizons_minutes,
                        sample_timestamps_local=sample_timestamps_local,
                        sample_timestamps_utc=sample_timestamps_utc,
                        predictions=predictions,
                        baseline_log=baseline_log,
                        measurements_dir=args.measurements_dir,
                        merge_tolerance_minutes=args.merge_tolerance_minutes,
                        sensor=args.sensor,
                        quantity=args.quantity,
                        alpha_short=alpha_short,
                        alpha_long=alpha_long,
                        generation_start_local=generation_start_local,
                        generation_end_local=generation_end_local,
                    )
                    for row in slice_rows:
                        row["window"] = f"startup_slice_{day.date().isoformat()}"
                    startup_rows.extend(slice_rows)

            recent_df = pd.DataFrame(recent_rows)
            daily_df = pd.DataFrame(daily_rows)
            startup_df = pd.DataFrame(startup_rows)
            gate = _gate_from_frames(recent_df, daily_df, startup_df, args.guard_horizons_minutes, args.short_horizon_guard_margin_c)
            row = {
                "alpha_short": alpha_short,
                "alpha_long": alpha_long,
                "overall_pass": gate["overall_pass"],
                "blind_recent_360_delta_c": gate["blind_recent_window_delta_c"],
                "blind_daily_mean_360_delta_c": gate["blind_daily_mean_delta_c"],
                "blind_daily_worst_360_delta_c": gate["blind_daily_worst_delta_c"],
                "blind_startup_mean_360_delta_c": gate["blind_startup_mean_delta_c"],
                "blind_startup_worst_360_delta_c": gate["blind_startup_worst_delta_c"],
            }
            candidate_rows.append(row)

            rank = (
                0 if gate["overall_pass"] else 1,
                float(gate["blind_daily_worst_delta_c"] if gate["blind_daily_worst_delta_c"] is not None else 1e9),
                float(gate["blind_recent_window_delta_c"] if gate["blind_recent_window_delta_c"] is not None else 1e9),
            )
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_summary = {
                    "alpha_short": alpha_short,
                    "alpha_long": alpha_long,
                    "gate": gate,
                    "runtime_model_path": str(runtime_model_path),
                    "runtime_model_kind": runtime_kind,
        "scaler_path": str(resolved_scaler_path),
                }

    candidates_df = pd.DataFrame(candidate_rows).sort_values(
        ["overall_pass", "blind_daily_worst_360_delta_c", "blind_recent_360_delta_c"],
        ascending=[False, True, True],
    )
    candidates_csv = args.output_dir / "blend_search_candidates.csv"
    candidates_df.to_csv(candidates_csv, index=False)

    summary = {
        "latest_measurement_date": str(latest_date.date()),
        "recent_local_adaptation_days": adaptation_days,
        "runtime_model_path": str(runtime_model_path),
        "runtime_model_kind": runtime_kind,
        "scaler_path": str(resolved_scaler_path),
        "best_candidate": best_summary,
        "artifacts": {
            "candidates_csv": str(candidates_csv),
        },
    }
    summary_path = args.output_dir / "blend_search_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(candidates_df.to_string(index=False))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
