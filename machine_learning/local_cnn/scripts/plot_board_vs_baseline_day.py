from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MEASUREMENTS_DIR = PROJECT_ROOT / "measurements"
ARTIFACTS_DIR = PROJECT_ROOT / "machine_learning" / "local_cnn" / "artifacts"
DEFAULT_HORIZONS = (30, 60, 180, 360)
DEFAULT_TIMEZONE = "America/Port_of_Spain"


def parse_horizon_list(raw_value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw_value.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one horizon must be provided.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot board inference vs baseline vs true measurements for one day.",
    )
    parser.add_argument(
        "--date",
        default=(datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat(),
        help="UTC date to analyze, in YYYY-MM-DD format. Defaults to yesterday.",
    )
    parser.add_argument(
        "--measurements-dir",
        type=Path,
        default=MEASUREMENTS_DIR,
        help="Directory containing inference, baseline, and measurements files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output PNG path.",
    )
    parser.add_argument(
        "--timezone",
        default=DEFAULT_TIMEZONE,
        help="IANA timezone name to use for plotting. Defaults to America/Port_of_Spain.",
    )
    parser.add_argument(
        "--horizons",
        type=parse_horizon_list,
        default=DEFAULT_HORIZONS,
        help="Comma-separated forecast horizons in minutes to plot. Defaults to 30,60,180,360.",
    )
    return parser.parse_args()


def load_measurements(measurements_dir: Path, target_date: date) -> pd.DataFrame:
    paths = [
        measurements_dir / f"measurements_{target_date.isoformat()}.csv",
        measurements_dir / f"measurements_{(target_date + timedelta(days=1)).isoformat()}.csv",
    ]
    frames: list[pd.DataFrame] = []
    for path in paths:
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
        raise FileNotFoundError("No matching measurement files found for the requested date.")
    return pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def load_inference(measurements_dir: Path, target_date: date) -> pd.DataFrame:
    path = measurements_dir / f"inference_{target_date.isoformat()}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Could not find inference file at {path}")
    inference = pd.read_csv(path, parse_dates=["timestamp_iso8601"], on_bad_lines="skip")
    if inference.empty:
        raise ValueError(f"Inference file {path.name} is empty.")
    return inference.rename(columns={"timestamp_iso8601": "timestamp"}).sort_values("timestamp")


def load_baseline(measurements_dir: Path) -> pd.DataFrame:
    path = measurements_dir / "baseline_forecast.log"
    if not path.exists():
        raise FileNotFoundError(f"Could not find baseline log at {path}")
    baseline = pd.read_csv(path, parse_dates=["timestamp_iso8601"], on_bad_lines="skip")
    if baseline.empty:
        raise ValueError(f"Baseline log {path.name} is empty.")
    return baseline.rename(columns={"timestamp_iso8601": "timestamp"}).sort_values("timestamp")


def align_baseline_to_inference(inference: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    return pd.merge_asof(
        inference[["timestamp"]].sort_values("timestamp"),
        baseline.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=3),
    )


def build_horizon_frame(
    inference: pd.DataFrame,
    baseline_on_grid: pd.DataFrame,
    measurements: pd.DataFrame,
    horizon_minutes: int,
) -> pd.DataFrame:
    inference_column = (
        f"predicted_temperature_c_plus_{horizon_minutes // 60:02d}h{horizon_minutes % 60:02d}m"
    )
    baseline_column = f"forecast_t+{horizon_minutes}m_c"
    if inference_column not in inference.columns:
        raise KeyError(f"Inference column {inference_column!r} not found.")
    if baseline_column not in baseline_on_grid.columns:
        raise KeyError(f"Baseline column {baseline_column!r} not found.")

    forecast_frame = pd.DataFrame(
        {
            "timestamp": inference["timestamp"],
            "forecasted_timestamp": inference["timestamp"] + pd.Timedelta(minutes=horizon_minutes),
            "model_pred_c": pd.to_numeric(inference[inference_column], errors="coerce"),
            "baseline_pred_c": pd.to_numeric(baseline_on_grid[baseline_column], errors="coerce"),
        }
    ).dropna(subset=["model_pred_c", "baseline_pred_c"])

    aligned = pd.merge_asof(
        forecast_frame.sort_values("forecasted_timestamp"),
        measurements[["timestamp", "measured_temperature_c"]].sort_values("timestamp"),
        left_on="forecasted_timestamp",
        right_on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta(minutes=15),
        suffixes=("", "_measurement"),
    )
    aligned = aligned.dropna(subset=["measured_temperature_c"]).copy()
    aligned["model_abs_error"] = (
        aligned["model_pred_c"] - aligned["measured_temperature_c"]
    ).abs()
    aligned["baseline_abs_error"] = (
        aligned["baseline_pred_c"] - aligned["measured_temperature_c"]
    ).abs()
    return aligned


def plot_horizons(
    target_date: date,
    aligned_by_horizon: dict[int, pd.DataFrame],
    output_path: Path,
    plot_timezone: ZoneInfo,
    horizons: tuple[int, ...],
) -> None:
    if len(horizons) == 1:
        fig, axis = plt.subplots(1, 1, figsize=(14, 5.5), sharex=False)
        axes = [axis]
    else:
        rows = (len(horizons) + 1) // 2
        fig, axes = plt.subplots(rows, 2, figsize=(14, 4.5 * rows), sharex=False)
        axes = axes.ravel().tolist()

    for axis, horizon in zip(axes, horizons):
        aligned = aligned_by_horizon[horizon].copy()
        aligned["forecasted_timestamp_local"] = aligned["forecasted_timestamp"].dt.tz_convert(
            plot_timezone
        )
        axis.plot(
            aligned["forecasted_timestamp_local"],
            aligned["measured_temperature_c"],
            label="True measurement",
            color="#1f2937",
            linewidth=2.0,
        )
        axis.plot(
            aligned["forecasted_timestamp_local"],
            aligned["model_pred_c"],
            label="Inference model",
            color="#0f766e",
            linewidth=1.6,
        )
        axis.plot(
            aligned["forecasted_timestamp_local"],
            aligned["baseline_pred_c"],
            label="Baseline",
            color="#b91c1c",
            linewidth=1.4,
        )
        axis.set_title(
            f"+{horizon} min | model MAE {aligned['model_abs_error'].mean():.3f} C | "
            f"baseline {aligned['baseline_abs_error'].mean():.3f} C"
        )
        axis.set_ylabel("Temperature (C)")
        axis.grid(True, alpha=0.25)
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=plot_timezone))
        axis.set_xlabel(f"Forecasted timestamp ({plot_timezone.key})")

    for axis in axes:
        axis.tick_params(axis="x", rotation=20)

    for axis in axes[len(horizons) :]:
        axis.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle(
        "Board Model vs Baseline vs True Measurements\n"
        f"Forecasts generated on {target_date.isoformat()} | plotted in {plot_timezone.key}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.02, 0.06, 1.0, 0.93))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    target_date = date.fromisoformat(args.date)
    plot_timezone = ZoneInfo(args.timezone)
    output_path = args.output or ARTIFACTS_DIR / f"board_vs_baseline_{target_date.isoformat()}.png"

    inference = load_inference(args.measurements_dir, target_date)
    baseline = load_baseline(args.measurements_dir)
    measurements = load_measurements(args.measurements_dir, target_date)
    baseline_on_grid = align_baseline_to_inference(inference, baseline)

    aligned_by_horizon: dict[int, pd.DataFrame] = {}
    for horizon in DEFAULT_HORIZONS:
        aligned_by_horizon[horizon] = build_horizon_frame(
            inference=inference,
            baseline_on_grid=baseline_on_grid,
            measurements=measurements,
            horizon_minutes=horizon,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_horizons(
        target_date=target_date,
        aligned_by_horizon=aligned_by_horizon,
        output_path=output_path,
        plot_timezone=plot_timezone,
        horizons=args.horizons,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
