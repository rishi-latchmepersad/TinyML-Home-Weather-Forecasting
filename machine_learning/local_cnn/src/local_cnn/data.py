from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from .config import PipelineConfig

logger = logging.getLogger(__name__)


def _read_measurement_csv(csv_path: Path) -> pd.DataFrame:
    for encoding, engine, on_bad_lines in (
        ("utf-8", "c", "error"),
        ("latin-1", "c", "error"),
        ("utf-8", "python", "skip"),
        ("latin-1", "python", "skip"),
    ):
        try:
            dataframe = pd.read_csv(
                csv_path,
                parse_dates=["timestamp_iso8601"],
                encoding=encoding,
                engine=engine,
                on_bad_lines=on_bad_lines,
            )
            dataframe.attrs["source_encoding"] = encoding
            dataframe.attrs["parser_engine"] = engine
            return dataframe
        except (pd.errors.ParserError, UnicodeDecodeError):
            continue

    raise RuntimeError(f"Failed to decode measurement CSV: {csv_path}")


def load_local_sensor_measurements(
    measurements_directory: str | Path,
    resample_frequency: str,
) -> pd.DataFrame:
    measurements_path = Path(measurements_directory)
    csv_files = sorted(measurements_path.glob("measurements_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No measurement CSV files found in {measurements_path}")

    daily_frames: list[pd.DataFrame] = []
    for csv_path in csv_files:
        daily_frame = _read_measurement_csv(csv_path)
        if daily_frame.attrs.get("source_encoding") == "latin-1":
            logger.warning("Loaded %s with latin-1 fallback", csv_path.name)
        if daily_frame.attrs.get("parser_engine") == "python":
            logger.warning("Loaded %s with python CSV parser fallback", csv_path.name)
        daily_frames.append(daily_frame)

    raw_measurements = pd.concat(daily_frames, ignore_index=True)
    filtered_measurements = raw_measurements.loc[
        raw_measurements["quantity"] != "is_raining"
    ].copy()
    filtered_measurements["timestamp_iso8601"] = pd.to_datetime(
        filtered_measurements["timestamp_iso8601"],
        utc=True,
        errors="coerce",
    )
    filtered_measurements["value"] = pd.to_numeric(
        filtered_measurements["value"],
        errors="coerce",
    )
    filtered_measurements["quantity"] = filtered_measurements["quantity"].astype(str)
    filtered_measurements = filtered_measurements[
        filtered_measurements["quantity"].isin(
            {"temperature_c", "humidity_pct", "pressure_pa", "lux_lx"}
        )
    ]
    filtered_measurements = filtered_measurements.dropna(
        subset=["timestamp_iso8601", "quantity", "value"]
    )

    pivoted_measurements = filtered_measurements.pivot_table(
        index="timestamp_iso8601",
        columns="quantity",
        values="value",
        aggfunc="mean",
    )
    renamed_columns = pivoted_measurements.rename(
        columns={
            "temperature_c": "temperature",
            "humidity_pct": "humidity",
            "pressure_pa": "pressure",
            "lux_lx": "illuminance_lux",
        }
    )

    if isinstance(renamed_columns.columns, pd.MultiIndex):
        renamed_columns.columns = renamed_columns.columns.get_level_values(-1)
    else:
        renamed_columns.columns = renamed_columns.columns.astype(str)
    renamed_columns = renamed_columns.rename_axis(None, axis=1)

    renamed_columns.index = (
        renamed_columns.index.tz_convert("America/Port_of_Spain").tz_localize(None)
    )
    renamed_columns.index.name = "timestamp_local"

    resampled_dataframe = renamed_columns.resample(resample_frequency).mean()
    weather_dataframe = resampled_dataframe.sort_index().interpolate(limit_direction="both")

    available_columns = [
        column
        for column in ["temperature", "humidity", "pressure", "illuminance_lux"]
        if column in weather_dataframe.columns
    ]
    return weather_dataframe[available_columns]


def load_open_meteo_hourly(
    latitude: float,
    longitude: float,
    timezone_str: str,
    resample_frequency: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    if end_date is None:
        end_date = pd.Timestamp.now(tz=timezone_str).date().isoformat()
    if start_date is None:
        start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=730)).date().isoformat()

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "shortwave_radiation",
        ],
        "timezone": timezone_str,
    }

    response = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params=params,
        timeout=30,
    )
    response.raise_for_status()

    payload = response.json()
    hourly_payload = payload.get("hourly")
    if not hourly_payload:
        raise ValueError("Open-Meteo response did not include an 'hourly' block")

    hourly_dataframe = pd.DataFrame(hourly_payload)
    required_fields = [
        "time",
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "shortwave_radiation",
    ]
    missing_fields = [field for field in required_fields if field not in hourly_dataframe.columns]
    if missing_fields:
        raise ValueError(f"Open-Meteo hourly payload missing fields: {missing_fields}")

    hourly_dataframe["timestamp_local"] = (
        pd.to_datetime(hourly_dataframe["time"])
        .dt.tz_localize(timezone_str)
        .dt.tz_convert(timezone_str)
        .dt.tz_localize(None)
    )
    hourly_dataframe = hourly_dataframe.set_index("timestamp_local")
    hourly_dataframe = hourly_dataframe.rename(
        columns={
            "temperature_2m": "temperature",
            "relative_humidity_2m": "humidity",
        }
    )
    hourly_dataframe["pressure"] = hourly_dataframe["surface_pressure"] * 100.0
    hourly_dataframe["illuminance_lux"] = hourly_dataframe["shortwave_radiation"] * 126.7

    available_columns = [
        column
        for column in ["temperature", "humidity", "pressure", "illuminance_lux"]
        if column in hourly_dataframe.columns
    ]
    resampled_dataframe = (
        hourly_dataframe[available_columns]
        .sort_index()
        .resample(resample_frequency)
        .interpolate("time")
    )
    resampled_dataframe.index.name = "timestamp_local"
    return resampled_dataframe


def merge_local_and_open_meteo(
    local_df: pd.DataFrame,
    open_meteo_df: pd.DataFrame,
    resample_frequency: str,
) -> pd.DataFrame:
    common_columns = sorted(set(local_df.columns) | set(open_meteo_df.columns))
    local_aligned = local_df.reindex(columns=common_columns)
    remote_aligned = open_meteo_df.reindex(columns=common_columns)

    combined = local_aligned.combine_first(remote_aligned)
    combined = combined.sort_index().resample(resample_frequency).interpolate("time")
    combined.index.name = "timestamp_local"
    return combined


def build_weather_dataframe(config: PipelineConfig) -> pd.DataFrame:
    local_dataframe = load_local_sensor_measurements(
        measurements_directory=config.measurements_directory,
        resample_frequency=config.resample_frequency,
    )
    logger.info(
        "Loaded local measurements: shape=%s coverage=%s to %s",
        local_dataframe.shape,
        local_dataframe.index.min(),
        local_dataframe.index.max(),
    )

    if not config.include_open_meteo:
        return local_dataframe

    try:
        open_meteo_dataframe = load_open_meteo_hourly(
            latitude=config.latitude,
            longitude=config.longitude,
            timezone_str=config.timezone,
            resample_frequency=config.resample_frequency,
            start_date=config.open_meteo_start_date,
            end_date=config.open_meteo_end_date,
        )
    except Exception:
        if config.strict_open_meteo:
            raise
        logger.warning("Open-Meteo fetch failed; continuing with local-only data.", exc_info=True)
        return local_dataframe

    logger.info(
        "Loaded Open-Meteo data: shape=%s coverage=%s to %s",
        open_meteo_dataframe.shape,
        open_meteo_dataframe.index.min(),
        open_meteo_dataframe.index.max(),
    )
    combined_dataframe = merge_local_and_open_meteo(
        local_df=local_dataframe,
        open_meteo_df=open_meteo_dataframe,
        resample_frequency=config.resample_frequency,
    )
    logger.info(
        "Merged weather dataframe: shape=%s coverage=%s to %s",
        combined_dataframe.shape,
        combined_dataframe.index.min(),
        combined_dataframe.index.max(),
    )
    return combined_dataframe
