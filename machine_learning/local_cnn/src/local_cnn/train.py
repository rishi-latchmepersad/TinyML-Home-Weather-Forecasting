from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import (
    DEFAULT_MEASUREMENTS_DIRECTORY,
    DEFAULT_OUTPUT_DIRECTORY,
    PipelineConfig,
)
from .modeling import configure_tensorflow_runtime
from .pipeline import run_pipeline


def apply_fast_profile(config: PipelineConfig) -> PipelineConfig:
    config.tuner_max_trials = 1
    config.tuner_executions_per_trial = 1
    config.tuner_epochs = 8
    config.tuner_patience = 2
    config.fine_tune_epochs = 4
    config.fine_tune_batch_size = 64
    config.skip_pruning = True
    config.skip_quantization = True
    config.enable_op_determinism = False
    return config


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train, prune, and quantize the local microclimate CNN pipeline.",
    )
    parser.add_argument(
        "--measurements-dir",
        type=Path,
        default=DEFAULT_MEASUREMENTS_DIRECTORY,
        help="Directory containing measurements_*.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory for model artifacts and summaries.",
    )
    parser.add_argument(
        "--skip-open-meteo",
        action="store_true",
        help="Disable Open-Meteo augmentation and use only local measurements.",
    )
    parser.add_argument(
        "--strict-open-meteo",
        action="store_true",
        help="Fail instead of falling back to local-only data when Open-Meteo is unavailable.",
    )
    parser.add_argument(
        "--open-meteo-start-date",
        help="Optional Open-Meteo historical start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--open-meteo-end-date",
        help="Optional Open-Meteo historical end date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save the weather feature time-series plot into the output directory.",
    )
    parser.add_argument(
        "--skip-quantization",
        action="store_true",
        help="Skip TFLite export and save only the trained Keras artifacts.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use a faster training profile for iterative experiments.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "gpu", "cpu"],
        help="Execution device preference for TensorFlow.",
    )
    parser.add_argument(
        "--disable-gpu-memory-growth",
        action="store_true",
        help="Disable TensorFlow GPU memory growth and allow eager pre-allocation.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging verbosity.",
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    plot_path = args.output_dir / "weather_features.png" if args.plot else None
    config = PipelineConfig(
        measurements_directory=args.measurements_dir,
        output_directory=args.output_dir,
        include_open_meteo=not args.skip_open_meteo,
        strict_open_meteo=args.strict_open_meteo,
        open_meteo_start_date=args.open_meteo_start_date,
        open_meteo_end_date=args.open_meteo_end_date,
        skip_quantization=args.skip_quantization,
        device_preference=args.device,
        gpu_memory_growth=not args.disable_gpu_memory_growth,
        plot_path=plot_path,
    )
    if args.fast:
        config = apply_fast_profile(config)
    configure_tensorflow_runtime(config)
    artifacts = run_pipeline(config)
    logging.getLogger(__name__).info("Saved Keras model to %s", artifacts.keras_model_path)
    logging.getLogger(__name__).info("Saved summary to %s", artifacts.summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
