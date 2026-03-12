from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark_vs_measurements import main as benchmark_main
from .config import DEFAULT_FEATURE_COLUMNS, DEFAULT_OUTPUT_DIRECTORY, PipelineConfig
from .pipeline import run_pipeline
from .runtime import configure_tensorflow_runtime
from .train import apply_fast_profile


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run targeted local_cnn feature ablations and benchmark them against measurements.",
    )
    parser.add_argument(
        "--drop-feature",
        action="append",
        required=True,
        help="Feature to remove from the input set. May be supplied multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY / "ablation_run",
        help="Directory for ablation artifacts.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "gpu", "cpu"],
        default="gpu",
        help="Execution device preference for TensorFlow.",
    )
    parser.add_argument(
        "--measurement-recent-days",
        type=int,
        default=120,
        help="Restrict training to the most recent N dated measurement files.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=500,
        help="Bootstrap samples for the ablation benchmark.",
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
        "--horizons-minutes",
        type=int,
        nargs="+",
        default=None,
        help="Optional explicit benchmark horizons in minutes.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use the reduced fast-profile training loop instead of the full pipeline.",
    )
    parser.add_argument(
        "--skip-pruning",
        action="store_true",
        help="Skip pruning and keep the unpruned Keras model.",
    )
    parser.add_argument(
        "--skip-quantization",
        action="store_true",
        help="Skip TFLite export and save only Keras artifacts.",
    )
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()

    remaining_features = tuple(
        feature for feature in DEFAULT_FEATURE_COLUMNS if feature not in set(args.drop_feature)
    )
    if "temperature" not in remaining_features:
        raise ValueError("The target temperature feature cannot be removed.")

    config = PipelineConfig(
        output_directory=args.output_dir,
        include_open_meteo=not args.skip_open_meteo,
        strict_open_meteo=args.strict_open_meteo,
        skip_quantization=args.skip_quantization,
        skip_pruning=args.skip_pruning,
        selected_feature_columns=remaining_features,
        device_preference=args.device,
        measurement_recent_days=args.measurement_recent_days,
    )
    if args.fast:
        config = apply_fast_profile(config)
        config.measurement_recent_days = args.measurement_recent_days
        config.selected_feature_columns = remaining_features

    configure_tensorflow_runtime(config)
    artifacts = run_pipeline(config)

    benchmark_args = [
        "--split",
        "test",
        "--bootstrap-samples",
        str(args.bootstrap_samples),
        "--model-path",
        str(artifacts.keras_model_path),
        "--output-dir",
        str(args.output_dir),
    ]
    if args.horizons_minutes:
        benchmark_args.extend(["--horizons-minutes", *[str(value) for value in args.horizons_minutes]])
    for feature in remaining_features:
        benchmark_args.extend(["--feature-column", feature])
    import sys

    original_argv = sys.argv
    try:
        sys.argv = ["benchmark_vs_measurements", *benchmark_args]
        benchmark_main()
    finally:
        sys.argv = original_argv

    summary = {
        "dropped_features": list(args.drop_feature),
        "remaining_features": list(remaining_features),
        "output_dir": str(args.output_dir),
        "training_summary_path": str(args.output_dir / "training_summary.json"),
        "benchmark_summary_path": str(args.output_dir / "benchmark_vs_measurements.json"),
    }
    (args.output_dir / "feature_ablation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
