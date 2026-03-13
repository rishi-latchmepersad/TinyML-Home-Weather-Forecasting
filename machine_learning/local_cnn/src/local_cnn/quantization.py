from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import keras
import numpy as np
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as tflite_schema

from .features import PreparedDataset
from .modeling import estimate_conv1d_macs

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class QuantizationResult:
    float32_tflite_path: Path | None
    int8_tflite_path: Path | None
    float32_size_bytes: int | None
    int8_size_bytes: int | None
    keras_val_mae: float
    tflite_fp32_val_mae: float | None
    tflite_int8_val_mae: float | None
    keras_test_mae: float
    tflite_fp32_test_mae: float | None
    tflite_int8_test_mae: float | None
    int8_weight_bytes: int | None
    macs: int


def format_bytes(number_of_bytes: int) -> str:
    unit_labels = ["B", "KB", "MB", "GB", "TB"]
    value = float(number_of_bytes)
    unit_index = 0
    while value >= 1024.0 and unit_index < len(unit_labels) - 1:
        value /= 1024.0
        unit_index += 1
    return f"{value:.2f} {unit_labels[unit_index]}"


def describe_int8_dynamic_range(
    calibration_pool: np.ndarray,
    scale: float,
    zero_point: int,
) -> None:
    qmin, qmax = np.iinfo(np.int8).min, np.iinfo(np.int8).max
    int8_min = scale * (qmin - zero_point)
    int8_max = scale * (qmax - zero_point)
    flattened = calibration_pool.reshape(-1)
    dataset_min = float(flattened.min())
    dataset_max = float(flattened.max())
    pct_1 = float(np.quantile(flattened, 0.01))
    pct_99 = float(np.quantile(flattened, 0.99))
    logger.info(
        "[TFLite] calibration 1%%-99%%=[%.4f, %.4f] int8_span=[%.4f, %.4f] calibration_span=[%.4f, %.4f]",
        pct_1,
        pct_99,
        int8_min,
        int8_max,
        dataset_min,
        dataset_max,
    )


def evaluate_tflite_regression_mae(
    tflite_model_path: str | Path,
    input_array: np.ndarray,
    target_array: np.ndarray,
    max_samples: int = 1000,
    calibration_pool: np.ndarray | None = None,
) -> float:
    interpreter = tf.lite.Interpreter(model_path=str(tflite_model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_scale = input_details["quantization"][0]
    input_zero_point = input_details["quantization"][1]
    output_scale = output_details["quantization"][0]
    output_zero_point = output_details["quantization"][1]
    is_input_int8 = input_details["dtype"] == np.int8
    is_output_int8 = output_details["dtype"] == np.int8

    logger.info(
        "[TFLite] input dtype=%s scale=%s zero_point=%s",
        input_details["dtype"],
        input_scale,
        input_zero_point,
    )
    logger.info(
        "[TFLite] output dtype=%s scale=%s zero_point=%s",
        output_details["dtype"],
        output_scale,
        output_zero_point,
    )
    if is_input_int8 and calibration_pool is not None:
        describe_int8_dynamic_range(calibration_pool, input_scale, input_zero_point)

    number_of_samples = min(max_samples, input_array.shape[0])
    per_sample_mae: list[float] = []
    for index in range(number_of_samples):
        sample_input = input_array[index : index + 1]
        true_values = np.asarray(target_array[index], dtype=np.float32).reshape(-1)
        if is_input_int8:
            quantized_input = np.round(sample_input / input_scale + input_zero_point).astype(np.int8)
            interpreter.set_tensor(input_details["index"], quantized_input)
        else:
            interpreter.set_tensor(input_details["index"], sample_input.astype(np.float32))

        interpreter.invoke()
        model_output = interpreter.get_tensor(output_details["index"])
        if is_output_int8:
            dequantized_output = output_scale * (model_output.astype(np.float32) - output_zero_point)
        else:
            dequantized_output = model_output.astype(np.float32)

        predicted_values = np.asarray(dequantized_output).reshape(-1)
        if predicted_values.shape != true_values.shape:
            raise ValueError(
                f"Shape mismatch between predictions {predicted_values.shape} and targets {true_values.shape}"
            )
        per_sample_mae.append(float(np.mean(np.abs(predicted_values - true_values))))

    return float(np.mean(per_sample_mae))


def evaluate_keras_mae(keras_model: keras.Model, X: np.ndarray, y: np.ndarray) -> float:
    evaluation = keras_model.evaluate(X, y, verbose=0, return_dict=True)
    if "mae" in evaluation:
        return float(evaluation["mae"])
    if "mean_absolute_error" in evaluation:
        return float(evaluation["mean_absolute_error"])

    for metric_name, metric_value in evaluation.items():
        if metric_name.endswith("mae") and metric_name not in {"horizon_60m_mae", "horizon_360m_mae"}:
            return float(metric_value)

    raise ValueError(f"Could not find a scalar MAE metric in evaluation results: {evaluation}")


def build_representative_dataset(
    calibration_pool: np.ndarray,
    max_calibration_samples: int,
) -> Callable[[], Iterable[list[np.ndarray]]]:
    def representative_dataset() -> Iterable[list[np.ndarray]]:
        total_sequences = calibration_pool.shape[0]
        if total_sequences == 0:
            raise ValueError("No calibration sequences available for INT8 quantization.")

        max_samples = min(max_calibration_samples, total_sequences)
        temperature_feature_index = 0
        sequence_temperatures = calibration_pool[:, :, temperature_feature_index]
        sequence_max = sequence_temperatures.max(axis=1)
        sequence_min = sequence_temperatures.min(axis=1)
        high_cutoff = float(np.quantile(sequence_max, 0.98))
        low_cutoff = float(np.quantile(sequence_min, 0.02))

        extreme_indices = np.unique(
            np.concatenate(
                [
                    np.where(sequence_max >= high_cutoff)[0],
                    np.where(sequence_min <= low_cutoff)[0],
                ]
            )
        )
        rng = np.random.default_rng(seed=42)
        remaining_candidates = np.setdiff1d(
            np.arange(total_sequences),
            extreme_indices,
            assume_unique=True,
        )
        num_remaining = max(max_samples - len(extreme_indices), 0)
        sampled_remaining = rng.permutation(remaining_candidates)[:num_remaining]
        ordered_indices = np.concatenate([extreme_indices, sampled_remaining])[:max_samples]

        for index in ordered_indices:
            normalized_sequence = calibration_pool[index : index + 1].astype(np.float32)
            yield [normalized_sequence]

    return representative_dataset


def convert_to_tflite_float32(
    keras_model: keras.Model,
    working_directory: Path,
) -> bytes:
    with tempfile.TemporaryDirectory(dir=working_directory) as temporary_directory:
        export_directory = Path(temporary_directory) / "saved_model"
        keras_model.export(export_directory)
        float_converter = tf.lite.TFLiteConverter.from_saved_model(str(export_directory))
        return float_converter.convert()


def convert_to_tflite_int8(
    keras_model: keras.Model,
    representative_dataset: Callable[[], Iterable[list[np.ndarray]]],
    working_directory: Path,
) -> bytes:
    with tempfile.TemporaryDirectory(dir=working_directory) as temporary_directory:
        export_directory = Path(temporary_directory) / "saved_model"
        keras_model.export(export_directory)
        int8_converter = tf.lite.TFLiteConverter.from_saved_model(str(export_directory))
        int8_converter.optimizations = [tf.lite.Optimize.DEFAULT]
        int8_converter.representative_dataset = representative_dataset
        int8_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        int8_converter.inference_input_type = tf.int8
        int8_converter.inference_output_type = tf.int8
        return int8_converter.convert()


def estimate_tflite_weight_bytes(tflite_model_path: str | Path) -> int:
    model_buffer = Path(tflite_model_path).read_bytes()
    model = tflite_schema.Model.GetRootAsModel(model_buffer, 0)
    referenced_buffers: set[int] = set()

    for subgraph_index in range(model.SubgraphsLength()):
        subgraph = model.Subgraphs(subgraph_index)
        for tensor_index in range(subgraph.TensorsLength()):
            tensor = subgraph.Tensors(tensor_index)
            buffer_index = tensor.Buffer()
            if buffer_index != 0:
                referenced_buffers.add(buffer_index)

    weight_bytes = 0
    for buffer_index in referenced_buffers:
        buffer = model.Buffers(buffer_index)
        data = buffer.DataAsNumpy()
        if isinstance(data, np.ndarray):
            weight_bytes += data.nbytes
    return weight_bytes


def export_quantized_models(
    keras_model: keras.Model,
    dataset: PreparedDataset,
    output_directory: Path,
    calibration_max_samples: int,
) -> QuantizationResult:
    output_directory.mkdir(parents=True, exist_ok=True)
    representative_dataset = build_representative_dataset(
        calibration_pool=dataset.quantization_calibration_sequences,
        max_calibration_samples=calibration_max_samples,
    )
    tflite_float32_bytes = convert_to_tflite_float32(
        keras_model=keras_model,
        working_directory=output_directory,
    )
    tflite_int8_bytes = convert_to_tflite_int8(
        keras_model=keras_model,
        representative_dataset=representative_dataset,
        working_directory=output_directory,
    )

    float32_tflite_path = output_directory / "pruned_float32.tflite"
    int8_tflite_path = output_directory / "pruned_int8.tflite"
    float32_tflite_path.write_bytes(tflite_float32_bytes)
    int8_tflite_path.write_bytes(tflite_int8_bytes)

    keras_val_mae = evaluate_keras_mae(keras_model, dataset.X_validate, dataset.y_validate)
    keras_test_mae = evaluate_keras_mae(keras_model, dataset.X_test, dataset.y_test)
    tflite_fp32_val_mae = evaluate_tflite_regression_mae(
        tflite_model_path=float32_tflite_path,
        input_array=dataset.X_validate,
        target_array=dataset.y_validate,
    )
    tflite_int8_val_mae = evaluate_tflite_regression_mae(
        tflite_model_path=int8_tflite_path,
        input_array=dataset.X_validate,
        target_array=dataset.y_validate,
        calibration_pool=dataset.quantization_calibration_sequences,
    )
    tflite_fp32_test_mae = evaluate_tflite_regression_mae(
        tflite_model_path=float32_tflite_path,
        input_array=dataset.X_test,
        target_array=dataset.y_test,
    )
    tflite_int8_test_mae = evaluate_tflite_regression_mae(
        tflite_model_path=int8_tflite_path,
        input_array=dataset.X_test,
        target_array=dataset.y_test,
        calibration_pool=dataset.quantization_calibration_sequences,
    )

    logger.info(
        "TFLite sizes: fp32=%s int8=%s reduction=%.2f%%",
        format_bytes(len(tflite_float32_bytes)),
        format_bytes(len(tflite_int8_bytes)),
        (1 - len(tflite_int8_bytes) / len(tflite_float32_bytes)) * 100,
    )

    return QuantizationResult(
        float32_tflite_path=float32_tflite_path,
        int8_tflite_path=int8_tflite_path,
        float32_size_bytes=len(tflite_float32_bytes),
        int8_size_bytes=len(tflite_int8_bytes),
        keras_val_mae=keras_val_mae,
        tflite_fp32_val_mae=tflite_fp32_val_mae,
        tflite_int8_val_mae=tflite_int8_val_mae,
        keras_test_mae=keras_test_mae,
        tflite_fp32_test_mae=tflite_fp32_test_mae,
        tflite_int8_test_mae=tflite_int8_test_mae,
        int8_weight_bytes=estimate_tflite_weight_bytes(int8_tflite_path),
        macs=estimate_conv1d_macs(keras_model),
    )


def summarize_keras_model_without_tflite(
    keras_model: keras.Model,
    dataset: PreparedDataset,
) -> QuantizationResult:
    return QuantizationResult(
        float32_tflite_path=None,
        int8_tflite_path=None,
        float32_size_bytes=None,
        int8_size_bytes=None,
        keras_val_mae=evaluate_keras_mae(keras_model, dataset.X_validate, dataset.y_validate),
        tflite_fp32_val_mae=None,
        tflite_int8_val_mae=None,
        keras_test_mae=evaluate_keras_mae(keras_model, dataset.X_test, dataset.y_test),
        tflite_fp32_test_mae=None,
        tflite_int8_test_mae=None,
        int8_weight_bytes=None,
        macs=estimate_conv1d_macs(keras_model),
    )
