from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from .config import DEFAULT_FEATURE_COLUMNS, DEFAULT_OUTPUT_DIRECTORY

BASE_FEATURES = ("temperature", "humidity", "pressure", "illuminance_lux")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Greedy forward feature selection using held-out replay MAE at a target horizon.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY / "forward_selection_60m",
        help="Directory for forward-selection artifacts.",
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
        help="Bootstrap samples for the replay benchmark.",
    )
    parser.add_argument(
        "--selection-horizon-minutes",
        type=int,
        default=60,
        help="Benchmark horizon used to score feature additions.",
    )
    parser.add_argument(
        "--target-mae-threshold",
        type=float,
        default=1.4,
        help="Only accept a feature if the resulting MAE is below this threshold.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Optional maximum number of forward-selection rounds.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use the reduced fast-profile training loop for each candidate run.",
    )
    parser.add_argument(
        "--skip-open-meteo",
        action="store_true",
        help="Disable Open-Meteo augmentation during feature-selection runs.",
    )
    parser.add_argument(
        "--strict-open-meteo",
        action="store_true",
        help="Fail instead of falling back to local-only data when Open-Meteo is unavailable.",
    )
    return parser


def sanitize_feature_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def benchmark_csv_path(output_dir: Path) -> Path:
    return output_dir / "benchmark_vs_measurements.csv"


def training_summary_path(output_dir: Path) -> Path:
    return output_dir / "training_summary.json"


def read_mae_at_horizon(path: Path, horizon_minutes: int) -> tuple[float, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if int(row["horizon_minutes"]) == horizon_minutes:
                return float(row["cnn_mae_c"]), float(row["baseline_mae_c"])
    raise ValueError(f"Horizon {horizon_minutes} not found in {path}")


def run_feature_set(
    output_dir: Path,
    selected_features: tuple[str, ...],
    measurement_recent_days: int,
    bootstrap_samples: int,
    selection_horizon_minutes: int,
    device: str,
    fast: bool,
    skip_open_meteo: bool,
    strict_open_meteo: bool,
) -> dict[str, object]:
    drop_features = [feature for feature in DEFAULT_FEATURE_COLUMNS if feature not in selected_features]
    command = [
        sys.executable,
        "-m",
        "local_cnn.feature_ablation",
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        "--measurement-recent-days",
        str(measurement_recent_days),
        "--bootstrap-samples",
        str(bootstrap_samples),
        "--horizons-minutes",
        str(selection_horizon_minutes),
    ]
    if fast:
        command.append("--fast")
    if skip_open_meteo:
        command.append("--skip-open-meteo")
    if strict_open_meteo:
        command.append("--strict-open-meteo")
    for feature in drop_features:
        command.extend(["--drop-feature", feature])
    subprocess.run(command, check=True)

    cnn_mae_c, baseline_mae_c = read_mae_at_horizon(
        benchmark_csv_path(output_dir),
        selection_horizon_minutes,
    )
    training_summary = json.loads(training_summary_path(output_dir).read_text(encoding="utf-8"))
    return {
        "output_dir": str(output_dir),
        "selected_features": list(selected_features),
        "dropped_features": drop_features,
        "cnn_mae_c": cnn_mae_c,
        "baseline_mae_c": baseline_mae_c,
        "delta_cnn_minus_baseline_c": cnn_mae_c - baseline_mae_c,
        "keras_test_mae": training_summary["metrics"]["keras_test_mae"],
        "keras_val_mae": training_summary["metrics"]["keras_val_mae"],
    }


def main() -> int:
    args = build_argument_parser().parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    current_features = BASE_FEATURES
    round_index = 0
    history: list[dict[str, object]] = []

    base_output_dir = args.output_root / "round_0_base"
    base_result = run_feature_set(
        output_dir=base_output_dir,
        selected_features=current_features,
        measurement_recent_days=args.measurement_recent_days,
        bootstrap_samples=args.bootstrap_samples,
        selection_horizon_minutes=args.selection_horizon_minutes,
        device=args.device,
        fast=args.fast,
        skip_open_meteo=args.skip_open_meteo,
        strict_open_meteo=args.strict_open_meteo,
    )
    base_result["round"] = round_index
    base_result["added_feature"] = None
    base_result["accepted"] = base_result["cnn_mae_c"] < args.target_mae_threshold
    history.append(base_result)

    current_best_mae = float(base_result["cnn_mae_c"])
    current_best_features = current_features

    remaining_features = [
        feature for feature in DEFAULT_FEATURE_COLUMNS if feature not in current_best_features
    ]

    while remaining_features:
        if args.max_rounds is not None and round_index >= args.max_rounds:
            break
        round_index += 1
        candidate_results: list[dict[str, object]] = []
        for feature in remaining_features:
            selected_features = (*current_best_features, feature)
            output_dir = args.output_root / f"round_{round_index}_add_{sanitize_feature_name(feature)}"
            result = run_feature_set(
                output_dir=output_dir,
                selected_features=selected_features,
                measurement_recent_days=args.measurement_recent_days,
                bootstrap_samples=args.bootstrap_samples,
                selection_horizon_minutes=args.selection_horizon_minutes,
                device=args.device,
                fast=args.fast,
                skip_open_meteo=args.skip_open_meteo,
                strict_open_meteo=args.strict_open_meteo,
            )
            result["round"] = round_index
            result["added_feature"] = feature
            result["accepted"] = False
            candidate_results.append(result)

        candidate_results.sort(key=lambda item: float(item["cnn_mae_c"]))
        best_candidate = candidate_results[0]
        accept_candidate = (
            float(best_candidate["cnn_mae_c"]) < args.target_mae_threshold
            and float(best_candidate["cnn_mae_c"]) < current_best_mae
        )
        for result in candidate_results:
            result["accepted"] = accept_candidate and result["added_feature"] == best_candidate["added_feature"]
        history.extend(candidate_results)

        if not accept_candidate:
            break

        current_best_mae = float(best_candidate["cnn_mae_c"])
        current_best_features = tuple(best_candidate["selected_features"])
        remaining_features = [
            feature for feature in DEFAULT_FEATURE_COLUMNS if feature not in current_best_features
        ]

    summary = {
        "selection_horizon_minutes": args.selection_horizon_minutes,
        "target_mae_threshold": args.target_mae_threshold,
        "base_features": list(BASE_FEATURES),
        "selected_features": list(current_best_features),
        "selected_feature_count": len(current_best_features),
        "best_cnn_mae_c": current_best_mae,
        "history": history,
    }
    (args.output_root / "forward_selection_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
