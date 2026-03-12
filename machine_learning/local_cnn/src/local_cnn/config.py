from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MEASUREMENTS_DIRECTORY = REPO_ROOT / "measurements"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "artifacts"
DEFAULT_TUNER_DIRECTORY = PROJECT_ROOT / "keras_tuner_dir"
DEFAULT_FEATURE_COLUMNS = (
    "temperature",
    "humidity",
    "pressure",
    "illuminance_lux",
    "delta_T",
    "temp_mean_6h",
    "temp_mean_24h",
    "temp_std_3h",
    "humidity_mean_6h",
    "humidity_delta",
    "pressure_delta",
    "sin_hour",
    "cos_hour",
)
DEFAULT_PRUNING_CANDIDATES = (
    {"conv1_k5": 0.92, "conv2_k3": 0.82, "sepconv1_k3": 0.75, "sepconv2_k3": 0.65},
    {"conv1_k5": 0.88, "conv2_k3": 0.78, "sepconv1_k3": 0.70, "sepconv2_k3": 0.60},
    {"conv1_k5": 0.80, "conv2_k3": 0.68, "sepconv1_k3": 0.62, "sepconv2_k3": 0.52},
    {"conv1_k5": 0.70, "conv2_k3": 0.60, "sepconv1_k3": 0.55, "sepconv2_k3": 0.45},
)


@dataclass(slots=True)
class PipelineConfig:
    measurements_directory: Path = DEFAULT_MEASUREMENTS_DIRECTORY
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY
    tuner_directory: Path = DEFAULT_TUNER_DIRECTORY
    resample_frequency: str = "30min"
    target_column_name: str = "temperature"
    historical_window_slots: int = 48
    forecast_horizon_slots: int = 24
    measurement_recent_days: int | None = None
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    latitude: float = 10.4236
    longitude: float = -61.4671
    timezone: str = "America/Port_of_Spain"
    include_open_meteo: bool = True
    strict_open_meteo: bool = False
    open_meteo_start_date: str | None = None
    open_meteo_end_date: str | None = None
    tuner_max_trials: int = 3
    tuner_executions_per_trial: int = 2
    tuner_epochs: int = 80
    tuner_batch_size: int = 64
    tuner_patience: int = 12
    fine_tune_epochs: int = 40
    fine_tune_batch_size: int = 128
    fine_tune_learning_rate: float = 5e-4
    pruning_epochs: int = 24
    pruning_batch_size: int = 128
    pruning_learning_rate: float = 3e-4
    calibration_max_samples: int = 4096
    skip_quantization: bool = False
    skip_pruning: bool = False
    device_preference: str = "auto"
    gpu_memory_growth: bool = True
    enable_op_determinism: bool = True
    random_seed: int = 42
    plot_path: Path | None = None
    selected_feature_columns: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS
    pruning_candidates: tuple[dict[str, float], ...] = field(
        default_factory=lambda: tuple(dict(candidate) for candidate in DEFAULT_PRUNING_CANDIDATES)
    )

    def ensure_directories(self) -> None:
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.tuner_directory.mkdir(parents=True, exist_ok=True)
