from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None


ROOT = Path(__file__).resolve().parents[1]
MEASUREMENTS_DIR = ROOT / "measurements"
OUTPUT_PATH = ROOT / "diagnostics" / "recent_window_forecasts_1h_6h_2026-03-28_1700_to_now.png"


@dataclass
class ForecastSeries:
    times: list[datetime]
    values: list[float]


def get_local_tz():
    if ZoneInfo is not None:
        try:
            return ZoneInfo("America/Port_of_Spain")
        except Exception:
            pass
    return timezone(timedelta(hours=-4))


LOCAL_TZ = get_local_tz()


def parse_iso8601(value: str) -> datetime:
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def to_local(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(LOCAL_TZ)


def read_measurements(paths: list[Path], start: datetime, end: datetime) -> ForecastSeries:
    times = []
    values = []

    for path in paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("quantity") != "temperature_c":
                    continue
                dt = to_local(parse_iso8601(row["timestamp_iso8601"]))
                if dt < start or dt > end:
                    continue
                times.append(dt)
                values.append(float(row["value"]))

    paired = sorted(zip(times, values), key=lambda item: item[0])
    if paired:
        times, values = map(list, zip(*paired))
    else:
        times, values = [], []
    return ForecastSeries(times, values)


def read_forecasts(
    path: Path,
    column: str,
    lead_minutes: int,
    start: datetime,
    end: datetime,
) -> ForecastSeries:
    times = []
    values = []
    if not path.exists():
        return ForecastSeries(times, values)

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dt = to_local(parse_iso8601(row["timestamp_iso8601"])) + timedelta(minutes=lead_minutes)
            if dt < start or dt > end:
                continue
            value = row.get(column)
            if value in (None, ""):
                continue
            times.append(dt)
            values.append(float(value))

    paired = sorted(zip(times, values), key=lambda item: item[0])
    if paired:
        times, values = map(list, zip(*paired))
    else:
        times, values = [], []
    return ForecastSeries(times, values)


def merge_series(*series: ForecastSeries) -> ForecastSeries:
    times = []
    values = []
    for s in series:
        times.extend(s.times)
        values.extend(s.values)
    paired = sorted(zip(times, values), key=lambda item: item[0])
    if paired:
        times, values = map(list, zip(*paired))
    else:
        times, values = [], []
    return ForecastSeries(times, values)


def common_time_window(*series: ForecastSeries) -> tuple[datetime, datetime] | None:
    starts = [min(s.times) for s in series if s.times]
    ends = [max(s.times) for s in series if s.times]
    if not starts or not ends:
        return None
    window_start = max(starts)
    window_end = min(ends)
    if window_start >= window_end:
        return None
    return window_start, window_end


def main() -> None:
    now_local = datetime.now(tz=LOCAL_TZ)
    start_local = (now_local - timedelta(days=1)).replace(hour=17, minute=0, second=0, microsecond=0)

    measured_temp = read_measurements(
        [
            MEASUREMENTS_DIR / "measurements_2026-03-28.csv",
            MEASUREMENTS_DIR / "measurements_2026-03-29.csv",
        ],
        start_local,
        now_local,
    )
    baseline_1h = read_forecasts(
        MEASUREMENTS_DIR / "baseline_forecast.log",
        "forecast_t+60m_c",
        60,
        start_local,
        now_local,
    )
    baseline_6h = read_forecasts(
        MEASUREMENTS_DIR / "baseline_forecast.log",
        "forecast_t+360m_c",
        360,
        start_local,
        now_local,
    )
    rnn_1h = merge_series(
        read_forecasts(
            MEASUREMENTS_DIR / "inference_2026-03-28.csv",
            "predicted_temperature_c_plus_01h00m",
            60,
            start_local,
            now_local,
        ),
        read_forecasts(
            MEASUREMENTS_DIR / "inference_2026-03-29.csv",
            "predicted_temperature_c_plus_01h00m",
            60,
            start_local,
            now_local,
        ),
    )
    rnn_6h = merge_series(
        read_forecasts(
            MEASUREMENTS_DIR / "inference_2026-03-28.csv",
            "predicted_temperature_c_plus_06h00m",
            360,
            start_local,
            now_local,
        ),
        read_forecasts(
            MEASUREMENTS_DIR / "inference_2026-03-29.csv",
            "predicted_temperature_c_plus_06h00m",
            360,
            start_local,
            now_local,
        ),
    )
    window_1h = None
    window_6h = common_time_window(measured_temp, baseline_6h, rnn_6h)

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=False, constrained_layout=True)
    fig.suptitle(
        f"Forecasts from {start_local:%Y-%m-%d %H:%M} local to {now_local:%Y-%m-%d %H:%M} local",
        fontsize=14,
    )

    ax = axes[0]
    if measured_temp.times:
        ax.plot(measured_temp.times, measured_temp.values, color="#4c4c4c", linewidth=1.2, alpha=0.85, label="Measured temp")
    if baseline_1h.times:
        ax.plot(baseline_1h.times, baseline_1h.values, color="#ff7f0e", linewidth=1.3, label="Baseline +1h")
    if rnn_1h.times:
        ax.plot(rnn_1h.times, rnn_1h.values, color="#2ca02c", linewidth=1.3, label="RNN +1h")
    ax.set_ylabel("Temp (C)")
    ax.set_title("+1h horizon")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    if measured_temp.times:
        ax.set_xlim(measured_temp.times[0], measured_temp.times[-1])

    ax = axes[1]
    if measured_temp.times:
        ax.plot(measured_temp.times, measured_temp.values, color="#4c4c4c", linewidth=1.2, alpha=0.85, label="Measured temp")
    if baseline_6h.times:
        ax.plot(baseline_6h.times, baseline_6h.values, color="#ff7f0e", linewidth=1.3, label="Baseline +6h")
    if rnn_6h.times:
        ax.plot(rnn_6h.times, rnn_6h.values, color="#2ca02c", linewidth=1.3, label="RNN +6h")
    ax.set_ylabel("Temp (C)")
    ax.set_title("+6h horizon")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    if window_6h is not None:
        ax.set_xlim(window_6h[0], window_6h[1])

    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.setp(axes[-1].get_xticklabels(), rotation=30, ha="right")
    axes[-1].set_xlabel("Local time")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=160)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
