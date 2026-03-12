from __future__ import annotations

import logging
from dataclasses import dataclass

import keras
import numpy as np
from keras import layers

from .config import PipelineConfig
from .features import PreparedDataset
from .modeling import estimate_conv1d_macs, estimate_model_size_bytes, horizon_60m_mae

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PruningResult:
    best_pruned_model: keras.Model
    best_keep_map: dict[str, float]
    best_validation_mae: float
    final_test_mae: float
    best_model_size_bytes: int
    unpruned_size_bytes: int
    pruning_score: float
    original_macs: int
    pruned_macs: int


def l1_ranked_channel_indices(weight_tensor: np.ndarray, channel_keep_ratio: float) -> np.ndarray:
    per_output_channel_l1 = np.sum(np.abs(weight_tensor), axis=(0, 1))
    number_of_output_channels = per_output_channel_l1.shape[0]
    number_to_keep = max(1, int(round(channel_keep_ratio * number_of_output_channels)))
    kept_indices = np.argsort(per_output_channel_l1)[-number_to_keep:]
    kept_indices.sort()
    return kept_indices


def prune_conv1d_layer(
    original_conv_layer: layers.Conv1D,
    channel_keep_ratio: float,
    previous_layer_kept_output_indices: np.ndarray | None = None,
) -> tuple[layers.Conv1D, np.ndarray | None]:
    weight_tensor, bias_vector = original_conv_layer.get_weights()

    if previous_layer_kept_output_indices is not None:
        weight_tensor = weight_tensor[:, previous_layer_kept_output_indices, :]

    if channel_keep_ratio < 1.0:
        kept_output_indices = l1_ranked_channel_indices(
            weight_tensor=weight_tensor,
            channel_keep_ratio=channel_keep_ratio,
        )
        pruned_weight_tensor = weight_tensor[:, :, kept_output_indices]
        pruned_bias_vector = bias_vector[kept_output_indices] if bias_vector is not None else None
        filters = len(kept_output_indices)
        layer_name = f"{original_conv_layer.name}_pruned"
    else:
        kept_output_indices = None
        pruned_weight_tensor = weight_tensor
        pruned_bias_vector = bias_vector
        filters = original_conv_layer.filters
        layer_name = (
            f"{original_conv_layer.name}_aligned"
            if previous_layer_kept_output_indices is not None
            else original_conv_layer.name
        )

    new_pruned_conv_layer = layers.Conv1D(
        filters=filters,
        kernel_size=original_conv_layer.kernel_size[0],
        strides=original_conv_layer.strides[0],
        padding=original_conv_layer.padding,
        dilation_rate=original_conv_layer.dilation_rate[0],
        groups=original_conv_layer.groups,
        use_bias=original_conv_layer.use_bias,
        activation=original_conv_layer.activation,
        name=layer_name,
    )

    if pruned_bias_vector is not None:
        new_pruned_conv_layer._init_weights = (pruned_weight_tensor, pruned_bias_vector)  # type: ignore[attr-defined]
    else:
        new_pruned_conv_layer._init_weights = (pruned_weight_tensor,)  # type: ignore[attr-defined]
    new_pruned_conv_layer._out_keep_indices = kept_output_indices  # type: ignore[attr-defined]
    return new_pruned_conv_layer, kept_output_indices


def l1_ranked_separable_output_indices(pointwise_kernel: np.ndarray, channel_keep_ratio: float) -> np.ndarray:
    per_output_channel_l1 = np.sum(np.abs(pointwise_kernel), axis=(0, 1))
    number_of_output_channels = per_output_channel_l1.shape[0]
    number_to_keep = max(1, int(round(channel_keep_ratio * number_of_output_channels)))
    kept_indices = np.argsort(per_output_channel_l1)[-number_to_keep:]
    kept_indices.sort()
    return kept_indices


def prune_or_align_separable_conv1d_layer(
    original_sep_layer: layers.SeparableConv1D,
    channel_keep_ratio: float,
    previous_layer_kept_output_indices: np.ndarray | None,
) -> tuple[layers.SeparableConv1D, np.ndarray | None]:
    weights_list = original_sep_layer.get_weights()
    depthwise_kernel = weights_list[0]
    pointwise_kernel = weights_list[1]
    has_bias = original_sep_layer.use_bias
    bias_vector = weights_list[2] if has_bias else None
    depth_multiplier = original_sep_layer.depth_multiplier

    if previous_layer_kept_output_indices is not None:
        depthwise_kernel = depthwise_kernel[:, previous_layer_kept_output_indices, :]
        pointwise_input_indices = np.concatenate(
            [
                np.arange(
                    channel_index * depth_multiplier,
                    channel_index * depth_multiplier + depth_multiplier,
                )
                for channel_index in previous_layer_kept_output_indices
            ]
        )
        pointwise_kernel = pointwise_kernel[:, pointwise_input_indices, :]

    if channel_keep_ratio < 1.0:
        kept_output_indices = l1_ranked_separable_output_indices(pointwise_kernel, channel_keep_ratio)
        pointwise_kernel = pointwise_kernel[:, :, kept_output_indices]
        if bias_vector is not None:
            bias_vector = bias_vector[kept_output_indices]
        filters = len(kept_output_indices)
        layer_name = f"{original_sep_layer.name}_pruned"
    else:
        kept_output_indices = None
        filters = original_sep_layer.filters
        layer_name = (
            f"{original_sep_layer.name}_aligned"
            if previous_layer_kept_output_indices is not None
            else original_sep_layer.name
        )

    new_aligned_sep_layer = layers.SeparableConv1D(
        filters=filters,
        kernel_size=original_sep_layer.kernel_size[0],
        strides=original_sep_layer.strides[0],
        padding=original_sep_layer.padding,
        dilation_rate=original_sep_layer.dilation_rate[0],
        depth_multiplier=depth_multiplier,
        use_bias=has_bias,
        activation=original_sep_layer.activation,
        name=layer_name,
    )

    if has_bias:
        new_aligned_sep_layer._init_weights = (  # type: ignore[attr-defined]
            depthwise_kernel,
            pointwise_kernel,
            bias_vector,
        )
    else:
        new_aligned_sep_layer._init_weights = (depthwise_kernel, pointwise_kernel)  # type: ignore[attr-defined]
    new_aligned_sep_layer._out_keep_indices = kept_output_indices  # type: ignore[attr-defined]
    return new_aligned_sep_layer, kept_output_indices


def _keras_history_layer_name(tensor: keras.KerasTensor) -> str:
    history = tensor._keras_history  # type: ignore[attr-defined]
    if hasattr(history, "operation"):
        return history.operation.name
    return history[0].name


def _as_tensor_list(tensor_or_tensors):
    if isinstance(tensor_or_tensors, (list, tuple)):
        return list(tensor_or_tensors)
    return [tensor_or_tensors]


def rebuild_pruned_with_align(
    original_model: keras.Model,
    layer_keep_ratio_map: dict[str, float],
    default_channel_keep_ratio: float = 1.0,
) -> keras.Model:
    new_pruned_model_input_tensor = keras.Input(
        shape=original_model.inputs[0].shape[1:],
        name="pruned_model_input",
    )
    old_to_new_layer: dict[str, layers.Layer] = {}
    kept_output_indices_by_layer_name: dict[str, np.ndarray] = {}
    new_tensor_by_original_tensor_id: dict[int, keras.KerasTensor] = {
        id(original_model.inputs[0]): new_pruned_model_input_tensor
    }

    for original_layer in original_model.layers[1:]:
        original_inputs = _as_tensor_list(original_layer.input)
        new_inputs = [new_tensor_by_original_tensor_id[id(tensor)] for tensor in original_inputs]
        inbound_layer_name = _keras_history_layer_name(original_inputs[0]) if len(original_inputs) == 1 else None
        inbound_kept_indices = (
            kept_output_indices_by_layer_name.get(inbound_layer_name)
            if inbound_layer_name is not None
            else None
        )
        channel_keep_ratio = layer_keep_ratio_map.get(
            original_layer.name,
            default_channel_keep_ratio,
        )

        if isinstance(original_layer, layers.Conv1D):
            new_layer, kept_output_indices = prune_conv1d_layer(
                original_conv_layer=original_layer,
                channel_keep_ratio=channel_keep_ratio,
                previous_layer_kept_output_indices=inbound_kept_indices,
            )
        elif isinstance(original_layer, layers.SeparableConv1D):
            new_layer, kept_output_indices = prune_or_align_separable_conv1d_layer(
                original_sep_layer=original_layer,
                channel_keep_ratio=channel_keep_ratio,
                previous_layer_kept_output_indices=inbound_kept_indices,
            )
        else:
            new_layer = original_layer.__class__.from_config(original_layer.get_config())
            kept_output_indices = None

        new_output = new_layer(new_inputs[0] if len(new_inputs) == 1 else new_inputs)
        old_to_new_layer[original_layer.name] = new_layer

        original_outputs = _as_tensor_list(original_layer.output)
        new_outputs = _as_tensor_list(new_output)
        for original_tensor, new_tensor in zip(original_outputs, new_outputs):
            new_tensor_by_original_tensor_id[id(original_tensor)] = new_tensor

        if kept_output_indices is not None:
            kept_output_indices_by_layer_name[original_layer.name] = kept_output_indices
        else:
            kept_output_indices_by_layer_name.pop(original_layer.name, None)

    mapped_outputs = [new_tensor_by_original_tensor_id[id(tensor)] for tensor in original_model.outputs]
    pruned_model = keras.Model(
        inputs=new_pruned_model_input_tensor,
        outputs=mapped_outputs[0] if len(mapped_outputs) == 1 else mapped_outputs,
        name=f"{original_model.name}_pruned",
    )

    for original_layer in original_model.layers:
        if original_layer.name not in old_to_new_layer:
            continue

        new_layer = old_to_new_layer[original_layer.name]
        try:
            if hasattr(new_layer, "_init_weights"):
                new_layer.set_weights(new_layer._init_weights)  # type: ignore[attr-defined]
            else:
                new_layer.set_weights(original_layer.get_weights())
        except Exception:
            logger.warning("Skipping weight transfer for %s", new_layer.name, exc_info=True)

    return pruned_model


def run_pruning_sweep(
    model: keras.Model,
    dataset: PreparedDataset,
    config: PipelineConfig,
) -> PruningResult:
    unpruned_size_bytes = estimate_model_size_bytes(model)
    best_validation_mae_value = float("inf")
    best_pruning_score = float("inf")
    best_keep_map: dict[str, float] | None = None
    best_pruned_model: keras.Model | None = None
    best_model_size_bytes = unpruned_size_bytes

    for candidate_keep_map in config.pruning_candidates:
        logger.info("Trying pruning candidate: %s", candidate_keep_map)
        pruned_candidate_model = rebuild_pruned_with_align(
            original_model=model,
            layer_keep_ratio_map=dict(candidate_keep_map),
            default_channel_keep_ratio=1.0,
        )
        pruned_candidate_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=config.pruning_learning_rate),
            loss=model.loss,
            metrics=[keras.metrics.MeanAbsoluteError(name="mae"), horizon_60m_mae],
        )
        pruned_candidate_model.fit(
            dataset.X_train,
            dataset.y_train,
            validation_data=(dataset.X_validate, dataset.y_validate),
            epochs=config.pruning_epochs,
            batch_size=config.pruning_batch_size,
            verbose=0,
            callbacks=[
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_horizon_60m_mae",
                    mode="min",
                    factor=0.5,
                    patience=4,
                    min_lr=5e-6,
                    verbose=0,
                ),
                keras.callbacks.EarlyStopping(
                    monitor="val_horizon_60m_mae",
                    mode="min",
                    patience=8,
                    restore_best_weights=True,
                    verbose=0,
                ),
            ],
        )

        evaluation = pruned_candidate_model.evaluate(
            dataset.X_validate,
            dataset.y_validate,
            verbose=0,
            return_dict=True,
        )
        validation_mae_value = float(evaluation["mae"])
        candidate_size_bytes = estimate_model_size_bytes(pruned_candidate_model)
        size_ratio = candidate_size_bytes / unpruned_size_bytes
        pruning_score = float(validation_mae_value) * size_ratio

        if pruning_score < best_pruning_score:
            best_pruning_score = pruning_score
            best_validation_mae_value = float(validation_mae_value)
            best_keep_map = dict(candidate_keep_map)
            best_pruned_model = pruned_candidate_model
            best_model_size_bytes = candidate_size_bytes

    if best_pruned_model is None or best_keep_map is None:
        raise RuntimeError("No pruning candidate produced a valid model.")

    final_test_mae = float(
        best_pruned_model.evaluate(dataset.X_test, dataset.y_test, verbose=0, return_dict=True)["mae"]
    )
    original_macs = estimate_conv1d_macs(model)
    pruned_macs = estimate_conv1d_macs(best_pruned_model)
    logger.info(
        "Selected pruning candidate: keep_map=%s val_mae=%.4f test_mae=%.4f mac_reduction=%.2f%%",
        best_keep_map,
        best_validation_mae_value,
        final_test_mae,
        (1 - pruned_macs / original_macs) * 100 if original_macs else 0.0,
    )
    return PruningResult(
        best_pruned_model=best_pruned_model,
        best_keep_map=best_keep_map,
        best_validation_mae=best_validation_mae_value,
        final_test_mae=float(final_test_mae),
        best_model_size_bytes=best_model_size_bytes,
        unpruned_size_bytes=unpruned_size_bytes,
        pruning_score=best_pruning_score,
        original_macs=original_macs,
        pruned_macs=pruned_macs,
    )


def summarize_unpruned_model(
    model: keras.Model,
    dataset: PreparedDataset,
) -> PruningResult:
    validation_mae = float(model.evaluate(dataset.X_validate, dataset.y_validate, verbose=0, return_dict=True)["mae"])
    test_mae = float(model.evaluate(dataset.X_test, dataset.y_test, verbose=0, return_dict=True)["mae"])
    model_size_bytes = estimate_model_size_bytes(model)
    model_macs = estimate_conv1d_macs(model)
    return PruningResult(
        best_pruned_model=model,
        best_keep_map={},
        best_validation_mae=validation_mae,
        final_test_mae=test_mae,
        best_model_size_bytes=model_size_bytes,
        unpruned_size_bytes=model_size_bytes,
        pruning_score=validation_mae,
        original_macs=model_macs,
        pruned_macs=model_macs,
    )
