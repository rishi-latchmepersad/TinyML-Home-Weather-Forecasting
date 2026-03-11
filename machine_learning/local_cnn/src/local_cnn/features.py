from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import PipelineConfig

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(slots=True)
class PreparedDataset:
    weather_dataframe: pd.DataFrame
    selected_features_dataframe: pd.DataFrame
    combined_dataframe: pd.DataFrame
    feature_columns: list[str]
    target_column_names: list[str]
    scaler: StandardScaler
    input_sequences: np.ndarray
    target_sequences: np.ndarray
    X_train: np.ndarray
    y_train: np.ndarray
    X_validate: np.ndarray
    y_validate: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    quantization_calibration_sequences: np.ndarray


def plot_weather_features(weather_dataframe: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        nrows=len(weather_dataframe.columns),
        ncols=1,
        figsize=(12, 15),
        sharex=True,
    )

    if len(weather_dataframe.columns) == 1:
        axes = [axes]

    for axis, column_name in zip(axes, weather_dataframe.columns, strict=True):
        axis.plot(weather_dataframe.index, weather_dataframe[column_name])
        axis.set_ylabel(column_name)
        axis.grid(True)

    plt.xlabel("Time")
    fig.suptitle("Couva Weather Data Over Time", y=1.02, fontsize=16)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def engineer_features(
    weather_dataframe: pd.DataFrame,
    selected_feature_columns: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    engineered_dataframe = weather_dataframe.copy()

    if "temperature" in engineered_dataframe.columns:
        engineered_dataframe["delta_T"] = (
            engineered_dataframe["temperature"] - engineered_dataframe["temperature"].shift(1)
        )
        engineered_dataframe["temp_mean_6h"] = engineered_dataframe["temperature"].rolling(
            window=12,
            min_periods=1,
        ).mean()
    else:
        engineered_dataframe["delta_T"] = np.nan
        engineered_dataframe["temp_mean_6h"] = np.nan

    if "humidity" in engineered_dataframe.columns:
        engineered_dataframe["humidity_mean_6h"] = engineered_dataframe["humidity"].rolling(
            window=12,
            min_periods=1,
        ).mean()
    else:
        engineered_dataframe["humidity_mean_6h"] = np.nan

    hour_of_day = engineered_dataframe.index.hour
    engineered_dataframe["sin_hour"] = np.sin(2 * np.pi * hour_of_day / 24)
    engineered_dataframe["cos_hour"] = np.cos(2 * np.pi * hour_of_day / 24)

    missing_features = [
        column_name
        for column_name in selected_feature_columns
        if column_name not in engineered_dataframe.columns
    ]
    if missing_features:
        raise ValueError(f"Missing expected engineered features: {missing_features}")

    selected_features_dataframe = engineered_dataframe[list(selected_feature_columns)].copy()
    return engineered_dataframe, selected_features_dataframe


def create_sliding_windows(
    feature_matrix: np.ndarray,
    target_array: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    sequential_features: list[np.ndarray] = []
    sequential_targets: list[np.ndarray] = []

    for start_index in range(len(feature_matrix) - window_size + 1):
        end_index = start_index + window_size
        sequential_features.append(feature_matrix[start_index:end_index, :])
        sequential_targets.append(target_array[end_index - 1])

    return (
        np.asarray(sequential_features, dtype=np.float32),
        np.asarray(sequential_targets, dtype=np.float32),
    )


def prepare_dataset(
    weather_dataframe: pd.DataFrame,
    config: PipelineConfig,
) -> PreparedDataset:
    engineered_dataframe, selected_features_dataframe = engineer_features(
        weather_dataframe=weather_dataframe,
        selected_feature_columns=config.selected_feature_columns,
    )

    future_target_columns = {
        f"target_t+{offset}": engineered_dataframe[config.target_column_name].shift(-offset)
        for offset in range(1, config.forecast_horizon_slots + 1)
    }
    target_dataframe = pd.DataFrame(future_target_columns)
    target_column_names = list(target_dataframe.columns)

    combined_dataframe = pd.concat(
        [selected_features_dataframe, target_dataframe],
        axis=1,
    ).dropna()

    raw_feature_values = combined_dataframe.drop(columns=target_column_names).to_numpy(dtype=np.float32)
    target_values = combined_dataframe[target_column_names].to_numpy(dtype=np.float32)

    num_total_samples = len(raw_feature_values) - config.historical_window_slots + 1
    if num_total_samples <= 0:
        raise ValueError("No samples were generated after windowing; check the input data coverage.")

    num_train_samples = int(config.train_fraction * num_total_samples)
    num_validation_samples = int(
        (config.train_fraction + config.validation_fraction) * num_total_samples
    )
    if num_train_samples <= 0:
        raise ValueError("Training split is empty; increase the dataset size or train_fraction.")

    # Fit normalization on training-only history rows to avoid leaking future statistics.
    scaler = StandardScaler()
    train_feature_stop = num_train_samples + config.historical_window_slots - 1
    scaler.fit(raw_feature_values[:train_feature_stop])
    normalized_features = scaler.transform(raw_feature_values)

    input_sequences, target_sequences = create_sliding_windows(
        feature_matrix=normalized_features,
        target_array=target_values,
        window_size=config.historical_window_slots,
    )

    X_train = input_sequences[:num_train_samples]
    y_train = target_sequences[:num_train_samples]
    X_validate = input_sequences[num_train_samples:num_validation_samples]
    y_validate = target_sequences[num_train_samples:num_validation_samples]
    X_test = input_sequences[num_validation_samples:]
    y_test = target_sequences[num_validation_samples:]

    return PreparedDataset(
        weather_dataframe=engineered_dataframe,
        selected_features_dataframe=selected_features_dataframe,
        combined_dataframe=combined_dataframe,
        feature_columns=list(config.selected_feature_columns),
        target_column_names=target_column_names,
        scaler=scaler,
        input_sequences=input_sequences,
        target_sequences=target_sequences,
        X_train=X_train,
        y_train=y_train,
        X_validate=X_validate,
        y_validate=y_validate,
        X_test=X_test,
        y_test=y_test,
        quantization_calibration_sequences=np.concatenate([X_train, X_validate, X_test], axis=0),
    )
