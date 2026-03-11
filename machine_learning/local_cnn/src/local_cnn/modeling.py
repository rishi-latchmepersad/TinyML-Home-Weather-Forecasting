from __future__ import annotations

import logging
import os
from dataclasses import dataclass

os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import keras_tuner as kt
import numpy as np
import tensorflow as tf
from keras import layers
from keras import ops

from .config import PipelineConfig
from .features import PreparedDataset

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TrainingResult:
    tuned_model: keras.Model
    tuned_validation_mae: float
    fine_tuned_model: keras.Model
    fine_tuned_validation_mae: float


@keras.saving.register_keras_serializable(package="local_cnn")
def horizon_60m_mae(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return ops.mean(ops.abs(y_true[:, 1] - y_pred[:, 1]))


def build_regression_loss() -> keras.losses.Loss:
    return keras.losses.Huber(delta=1.5)


def configure_tensorflow_runtime(config: PipelineConfig) -> None:
    available_gpus = tf.config.list_physical_devices("GPU")
    available_cpus = tf.config.list_physical_devices("CPU")

    if config.device_preference == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError:
            logger.warning("Could not disable GPU visibility after TensorFlow initialization.", exc_info=True)
        logger.info("TensorFlow runtime configured for CPU-only execution.")
        logger.info("Visible CPUs: %s", [device.name for device in available_cpus])
        return

    if config.device_preference == "gpu" and not available_gpus:
        raise RuntimeError("GPU execution was requested, but TensorFlow did not detect any GPUs.")

    if available_gpus and config.gpu_memory_growth:
        for device in available_gpus:
            try:
                tf.config.experimental.set_memory_growth(device, True)
            except RuntimeError:
                logger.warning("Could not enable memory growth for %s.", device.name, exc_info=True)

    if available_gpus:
        logger.info("TensorFlow will use GPU device(s): %s", [device.name for device in available_gpus])
    else:
        logger.info("TensorFlow did not detect a GPU. Training will run on CPU.")
    logger.info("Visible CPUs: %s", [device.name for device in available_cpus])


def seed_everything(seed: int, enable_op_determinism: bool = True) -> None:
    keras.utils.set_random_seed(seed)
    if enable_op_determinism:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            logger.debug("TensorFlow op determinism could not be enabled.", exc_info=True)


def build_one_dimensional_convolutional_model(
    input_window_length: int,
    number_of_input_features: int,
    forecast_horizon_slots: int = 24,
    dropout_rate: float = 0.10,
    first_block_filters: int = 24,
    second_block_filters: int = 64,
    projection_filters: int = 12,
    dense_units: int = 96,
) -> keras.Model:
    input_layer = keras.Input(
        shape=(input_window_length, number_of_input_features),
        name="input_window",
    )
    convolution_1 = layers.Conv1D(
        filters=first_block_filters,
        kernel_size=5,
        padding="same",
        activation="relu",
        name="conv1_k5",
    )(input_layer)
    convolution_2 = layers.Conv1D(
        filters=first_block_filters,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="conv2_k3",
    )(convolution_1)
    separable_convolution_1 = layers.SeparableConv1D(
        filters=second_block_filters,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="sepconv1_k3",
    )(convolution_2)
    separable_convolution_2 = layers.SeparableConv1D(
        filters=second_block_filters,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="sepconv2_k3",
    )(separable_convolution_1)
    projected_temporal_features = layers.Conv1D(
        filters=projection_filters,
        kernel_size=1,
        padding="same",
        activation="relu",
        name="conv_projection_k1",
    )(separable_convolution_2)
    flattened_temporal_features = layers.Flatten(name="flatten_temporal_features")(
        projected_temporal_features
    )
    dense_projection = layers.Dense(
        units=dense_units,
        activation="relu",
        name="dense_projection",
    )(flattened_temporal_features)
    regularized_features = layers.Dropout(
        rate=dropout_rate,
        name="dropout_regularization",
    )(dense_projection)
    output_temperature = layers.Dense(
        units=forecast_horizon_slots,
        dtype="float32",
        name="predicted_temperature_horizon",
    )(regularized_features)

    model = keras.Model(
        inputs=input_layer,
        outputs=output_temperature,
        name="lightweight_1d_cnn_forecaster",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=build_regression_loss(),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae"), horizon_60m_mae],
    )
    return model


def build_model_for_tuning(
    hp: kt.HyperParameters,
    input_window_length: int,
    number_of_input_features: int,
    forecast_horizon_slots: int,
) -> keras.Model:
    dropout_rate = hp.Float(
        "dropout_rate",
        min_value=0.05,
        max_value=0.25,
        step=0.05,
        default=0.10,
    )
    first_block_filters = hp.Choice(
        "first_block_filters",
        values=[16, 24, 32],
        default=24,
    )
    second_block_filters = hp.Choice(
        "second_block_filters",
        values=[48, 64, 96],
        default=64,
    )
    dense_units = hp.Choice(
        "dense_units",
        values=[48, 64, 96],
        default=96,
    )
    projection_filters = hp.Choice(
        "projection_filters",
        values=[8, 12, 16],
        default=12,
    )
    learning_rate = hp.Choice(
        "learning_rate",
        values=[3e-3, 1e-3, 3e-4],
        default=1e-3,
    )

    model = build_one_dimensional_convolutional_model(
        input_window_length=input_window_length,
        number_of_input_features=number_of_input_features,
        forecast_horizon_slots=forecast_horizon_slots,
        dropout_rate=dropout_rate,
        first_block_filters=first_block_filters,
        second_block_filters=second_block_filters,
        projection_filters=projection_filters,
        dense_units=dense_units,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=build_regression_loss(),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae"), horizon_60m_mae],
    )
    return model


def estimate_conv1d_macs(model: keras.Model) -> int:
    total_macs = 0
    for layer in model.layers:
        if isinstance(layer, layers.Conv1D):
            kernel = layer.kernel_size[0]
            out_ch = layer.filters
            in_ch = layer.input.shape[-1]
            out_len = layer.output.shape[1]
            total_macs += kernel * in_ch * out_ch * out_len
        elif isinstance(layer, layers.SeparableConv1D):
            kernel = layer.kernel_size[0]
            out_ch = layer.filters
            in_ch = layer.input.shape[-1]
            out_len = layer.output.shape[1]
            total_macs += kernel * in_ch * out_len
            total_macs += in_ch * out_ch * out_len
    return int(total_macs)


def estimate_model_size_bytes(model: keras.Model) -> int:
    total_bytes = 0
    for variable in model.trainable_variables:
        total_bytes += int(np.prod(variable.shape)) * tf.as_dtype(variable.dtype).size
    return int(total_bytes)


def tune_model(dataset: PreparedDataset, config: PipelineConfig) -> tuple[keras.Model, float]:
    project_name = (
        f"weather_forecaster_tuning_nfeat{dataset.X_train.shape[-1]}_win{dataset.X_train.shape[1]}"
    )
    tuner = kt.RandomSearch(
        hypermodel=lambda hp: build_model_for_tuning(
            hp,
            input_window_length=dataset.X_train.shape[1],
            number_of_input_features=dataset.X_train.shape[-1],
            forecast_horizon_slots=dataset.y_train.shape[-1],
        ),
        objective=kt.Objective("val_horizon_60m_mae", direction="min"),
        max_trials=config.tuner_max_trials,
        executions_per_trial=config.tuner_executions_per_trial,
        overwrite=True,
        directory=str(config.tuner_directory),
        project_name=project_name,
    )
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_horizon_60m_mae",
        mode="min",
        patience=config.tuner_patience,
        restore_best_weights=True,
    )
    tuner.search(
        dataset.X_train,
        dataset.y_train,
        validation_data=(dataset.X_validate, dataset.y_validate),
        epochs=config.tuner_epochs,
        batch_size=config.tuner_batch_size,
        callbacks=[early_stopping],
    )
    best_model = tuner.get_best_models(num_models=1)[0]
    evaluation = best_model.evaluate(
        dataset.X_validate,
        dataset.y_validate,
        verbose=0,
        return_dict=True,
    )
    validation_mae = float(evaluation["mae"])
    validation_60m_mae = float(evaluation["horizon_60m_mae"])
    logger.info("Tuned model validation MAE: %.4f", validation_mae)
    logger.info("Tuned model validation +60m MAE: %.4f", validation_60m_mae)
    return best_model, float(validation_mae)


def fine_tune_model(
    model: keras.Model,
    dataset: PreparedDataset,
    config: PipelineConfig,
) -> tuple[keras.Model, float]:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.fine_tune_learning_rate),
        loss=model.loss,
        metrics=[keras.metrics.MeanAbsoluteError(name="mae"), horizon_60m_mae],
    )
    callbacks = [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_horizon_60m_mae",
            mode="min",
            factor=0.5,
            patience=6,
            min_lr=5e-6,
            verbose=0,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_horizon_60m_mae",
            mode="min",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
    ]
    model.fit(
        dataset.X_train,
        dataset.y_train,
        validation_data=(dataset.X_validate, dataset.y_validate),
        epochs=config.fine_tune_epochs,
        batch_size=config.fine_tune_batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    evaluation = model.evaluate(
        dataset.X_validate,
        dataset.y_validate,
        verbose=0,
        return_dict=True,
    )
    validation_mae = float(evaluation["mae"])
    validation_60m_mae = float(evaluation["horizon_60m_mae"])
    logger.info("Fine-tuned model validation MAE: %.4f", validation_mae)
    logger.info("Fine-tuned model validation +60m MAE: %.4f", validation_60m_mae)
    return model, float(validation_mae)
