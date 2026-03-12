from __future__ import annotations

import json
from pathlib import Path

import keras
import numpy as np


def _smooth_bias_vector(bias_c: np.ndarray) -> np.ndarray:
    if bias_c.ndim != 1:
        raise ValueError("Bias calibration vector must be one-dimensional.")
    if bias_c.size < 3:
        return bias_c.astype(np.float32, copy=True)

    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    padded = np.pad(bias_c.astype(np.float32), pad_width=1, mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def compute_horizon_bias_calibration(
    model: keras.Model,
    X_validate: np.ndarray,
    y_validate: np.ndarray,
) -> np.ndarray:
    predictions = model.predict(X_validate, verbose=0).astype(np.float32)
    if predictions.shape != y_validate.shape:
        raise ValueError(
            f"Prediction shape {predictions.shape} does not match validation targets {y_validate.shape}."
        )
    raw_bias_c = np.mean(predictions - y_validate, axis=0, dtype=np.float64).astype(np.float32)
    return _smooth_bias_vector(raw_bias_c)


def apply_horizon_bias_calibration(predictions: np.ndarray, bias_c: np.ndarray | None) -> np.ndarray:
    if bias_c is None:
        return predictions
    if predictions.shape[-1] != bias_c.shape[0]:
        raise ValueError(
            f"Prediction horizon {predictions.shape[-1]} does not match bias vector {bias_c.shape[0]}."
        )
    return predictions - bias_c.reshape(1, -1)


def save_horizon_bias_calibration(output_path: Path, bias_c: np.ndarray) -> Path:
    payload = {
        "bias_c": bias_c.astype(float).tolist(),
        "horizon_count": int(bias_c.shape[0]),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def load_horizon_bias_calibration(input_path: Path | None) -> np.ndarray | None:
    if input_path is None or not input_path.exists():
        return None
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    bias_c = np.asarray(payload["bias_c"], dtype=np.float32)
    if bias_c.ndim != 1:
        raise ValueError("Loaded bias calibration must be one-dimensional.")
    return bias_c
