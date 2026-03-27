from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MEASUREMENTS_DIR = PROJECT_ROOT / "measurements"
ARTIFACTS_DIR = PROJECT_ROOT / "machine_learning" / "local_cnn" / "artifacts"
DEFAULT_TIMEZONE = "America/Port_of_Spain"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot baseline forecasts versus true measurements over a date range.",
    )
    parser.add_argument(
        "--start-date",
        required=True,
        help="First UTC date to include, in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="Last UTC date to include, in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--measurements-dir",
        type=Path,
        default=MEASUREMENTS_DIR,
        help="Directory containing baseline_forecast.log and measurements_YYYY-MM-DD.csv files.",
    )
    parser.add_argument(
        "--baseline-log",
        type=Path,
        default=None,
        help="Optional explicit baseline log path. Defaults to measurements-dir/baseline_forecast.log.",
    )
    parser.add_argument(
        "--timezone",
        default=DEFAULT_TIMEZONE,
        help="IANA timezone for plotting.",
    )
    parser.add_argument(
        "--horizon-minutes",
        type=int,
        default=360,
        help="Forecast horizon in minutes to plot. Defaults to 360.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output PNG path prefix. Separate _temperature and _error files will be created.",
    )
    parser.add_argument(
        "--start-local",
        default=None,
        help="Optional local plot-window start in YYYY-MM-DDTHH:MM format.",
    )
    parser.add_argument(
        "--end-local",
        default=None,
        help="Optional local plot-window end in YYYY-MM-DDTHH:MM format.",
    )
    return parser.parse_args()


def daterange(start_day: date, end_day: date) -> list[date]:
    total_days = (end_day - start_day).days
    return [start_day + timedelta(days=offset) for offset in range(total_days + 1)]


def load_baseline(measurements_dir: Path, baseline_log: Path | None = None) -> pd.DataFrame:
    path = baseline_log or (measurements_dir / "baseline_forecast.log")
    baseline = pd.read_csv(path, parse_dates=["timestamp_iso8601"], on_bad_lines="skip")
    if baseline.empty:
        raise ValueError(f"Baseline log {path} is empty.")
    return baseline.rename(columns={"timestamp_iso8601": "timestamp"}).sort_values("timestamp")


def load_measurements(measurements_dir: Path, days: list[date]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for day in days + [days[-1] + timedelta(days=1)]:
        path = measurements_dir / f"measurements_{day.isoformat()}.csv"
        if not path.exists():
            continue
        raw = pd.read_csv(path, parse_dates=["timestamp_iso8601"], on_bad_lines="skip")
        filtered = raw.loc[
            (raw["sensor"] == "bme280") & (raw["quantity"] == "temperature_c"),
            ["timestamp_iso8601", "value"],
        ].copy()
        if filtered.empty:
            continue
        filtered = filtered.rename(
            columns={"timestamp_iso8601": "timestamp", "value": "measured_temperature_c"}
        )
        filtered["measured_temperature_c"] = pd.to_numeric(
            filtered["measured_temperature_c"], errors="coerce"
        )
        filtered = filtered.dropna(subset=["timestamp", "measured_temperature_c"])
        frames.append(filtered)
    if not frames:
        raise ValueError("No measurement rows found for requested date range.")
    return pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def build_aligned_frame(
    baseline: pd.DataFrame,
    measurements: pd.DataFrame,
    days: list[date],
    horizon_minutes: int,
) -> pd.DataFrame:
    baseline_column = f"forecast_t+{horizon_minutes}m_c"
    if baseline_column not in baseline.columns:
        raise KeyError(f"Missing baseline column {baseline_column!r}")

    start_ts = pd.Timestamp(days[0].isoformat(), tz="UTC")
    end_ts = pd.Timestamp((days[-1] + timedelta(days=1)).isoformat(), tz="UTC")
    current = baseline.loc[
        (baseline["timestamp"] >= start_ts) & (baseline["timestamp"] < end_ts),
        ["timestamp", baseline_column],
    ].copy()
    current = current.rename(columns={baseline_column: "baseline_pred_c"})
    current["forecasted_timestamp"] = current["timestamp"] + pd.Timedelta(minutes=horizon_minutes)

    aligned = pd.merge_asof(
        current.sort_values("forecasted_timestamp"),
        measurements[["timestamp", "measured_temperature_c"]].sort_values("timestamp"),
        left_on="forecasted_timestamp",
        right_on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=15),
        suffixes=("", "_measurement"),
    )
    aligned = aligned.dropna(subset=["measured_temperature_c"]).copy()
    aligned["baseline_error_c"] = aligned["baseline_pred_c"] - aligned["measured_temperature_c"]
    aligned["baseline_abs_error"] = aligned["baseline_error_c"].abs()
    return aligned


def plot_range(
    aligned: pd.DataFrame,
    start_day: date,
    end_day: date,
    horizon_minutes: int,
    plot_timezone: ZoneInfo,
    output_prefix: Path,
    start_local: datetime | None,
    end_local: datetime | None,
) -> tuple[Path, Path]:
    plot_frame = aligned.copy()
    plot_frame["forecasted_timestamp_local"] = plot_frame["forecasted_timestamp"].dt.tz_convert(
        plot_timezone
    )
    if start_local is not None:
        plot_frame = plot_frame.loc[plot_frame["forecasted_timestamp_local"] >= start_local]
    if end_local is not None:
        plot_frame = plot_frame.loc[plot_frame["forecasted_timestamp_local"] <= end_local]
    if plot_frame.empty:
        raise ValueError("No aligned rows remain after applying the local-time window.")

    temp_output_path = output_prefix.with_name(f"{output_prefix.stem}_temperature{output_prefix.suffix}")
    error_output_path = output_prefix.with_name(f"{output_prefix.stem}_error{output_prefix.suffix}")

    fig, temp_axis = plt.subplots(
        1,
        1,
        figsize=(14, 5.2),
    )

    temp_axis.plot(
        plot_frame["forecasted_timestamp_local"],
        plot_frame["measured_temperature_c"],
        label="True measurement",
        color="#1f2937",
        linewidth=2.0,
    )
    temp_axis.plot(
        plot_frame["forecasted_timestamp_local"],
        plot_frame["baseline_pred_c"],
        label="Baseline",
        color="#b91c1c",
        linewidth=1.5,
    )
    temp_axis.set_ylabel("Temperature (C)")
    temp_axis.grid(True, alpha=0.25)
    temp_axis.legend(loc="upper right", frameon=False)
    temp_axis.set_xlabel("Forecasted timestamp")
    temp_axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=plot_timezone))
    temp_axis.tick_params(axis="x", rotation=20)
    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.98))
    fig.savefig(temp_output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, error_axis = plt.subplots(
        1,
        1,
        figsize=(14, 4.8),
    )
    error_axis.plot(
        plot_frame["forecasted_timestamp_local"],
        plot_frame["baseline_abs_error"],
        color="#7c3aed",
        linewidth=1.4,
        label="Absolute error",
    )
    error_axis.axhline(
        plot_frame["baseline_abs_error"].mean(),
        color="#374151",
        linestyle="--",
        linewidth=1.2,
        label=f"Mean AE ({plot_frame['baseline_abs_error'].mean():.3f} C)",
    )
    error_axis.set_ylabel("Absolute error (C)")
    error_axis.set_xlabel("Forecasted timestamp")
    error_axis.grid(True, alpha=0.25)
    error_axis.legend(loc="upper right", frameon=False)

    day_boundaries = [start_day + timedelta(days=offset) for offset in range((end_day - start_day).days + 2)]
    for boundary_day in day_boundaries[1:-1]:
        boundary_ts = pd.Timestamp(boundary_day.isoformat(), tz="UTC").tz_convert(plot_timezone)
        error_axis.axvline(boundary_ts, color="#9ca3af", linestyle=":", linewidth=1.0)

    error_axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=plot_timezone))
    error_axis.tick_params(axis="x", rotation=20)
    fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.98))
    fig.savefig(error_output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return temp_output_path, error_output_path


def main() -> int:
    args = parse_args()
    start_day = date.fromisoformat(args.start_date)
    end_day = date.fromisoformat(args.end_date)
    days = daterange(start_day, end_day)
    plot_timezone = ZoneInfo(args.timezone)
    output_prefix = args.output or (
        ARTIFACTS_DIR
        / f"baseline_vs_true_{start_day.isoformat()}_{end_day.isoformat()}_plus{args.horizon_minutes}m.png"
    )
    start_local = None
    end_local = None
    if args.start_local is not None:
        start_local = datetime.fromisoformat(args.start_local).replace(tzinfo=plot_timezone)
    if args.end_local is not None:
        end_local = datetime.fromisoformat(args.end_local).replace(tzinfo=plot_timezone)

    baseline = load_baseline(args.measurements_dir, args.baseline_log)
    measurements = load_measurements(args.measurements_dir, days)
    aligned = build_aligned_frame(
        baseline=baseline,
        measurements=measurements,
        days=days,
        horizon_minutes=args.horizon_minutes,
    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    temperature_output_path, error_output_path = plot_range(
        aligned=aligned,
        start_day=start_day,
        end_day=end_day,
        horizon_minutes=args.horizon_minutes,
        plot_timezone=plot_timezone,
        output_prefix=output_prefix,
        start_local=start_local,
        end_local=end_local,
    )
    print(temperature_output_path)
    print(error_output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
