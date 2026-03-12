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


@keras.saving.register_keras_serializable(package="local_cnn")
class WeightedForecastHuber(keras.losses.Loss):
    def __init__(
        self,
        forecast_horizon_slots: int,
        primary_horizon_index: int = 1,
        delta: float = 1.5,
        name: str = "weighted_forecast_huber",
    ) -> None:
        super().__init__(name=name)
        self.forecast_horizon_slots = int(forecast_horizon_slots)
        self.primary_horizon_index = int(primary_horizon_index)
        self.delta = float(delta)

        base_weights = np.linspace(1.20, 0.80, self.forecast_horizon_slots, dtype=np.float32)
        if 0 <= self.primary_horizon_index < self.forecast_horizon_slots:
            base_weights[self.primary_horizon_index] *= 2.5
        self._weights = tf.constant(base_weights / np.mean(base_weights), dtype=tf.float32)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        error = y_true - y_pred
        abs_error = ops.abs(error)
        quadratic = ops.minimum(abs_error, self.delta)
        linear = abs_error - quadratic
        per_horizon_loss = 0.5 * ops.square(quadratic) + self.delta * linear
        return ops.mean(per_horizon_loss * self._weights, axis=-1)

    def get_config(self) -> dict[str, object]:
        return {
            "forecast_horizon_slots": self.forecast_horizon_slots,
            "primary_horizon_index": self.primary_horizon_index,
            "delta": self.delta,
            "name": self.name,
        }


@keras.saving.register_keras_serializable(package="local_cnn")
class LastTimeStep(layers.Layer):
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return inputs[:, -1, :]


@keras.saving.register_keras_serializable(package="local_cnn")
class HorizonBlend(layers.Layer):
    def __init__(self, forecast_horizon_slots: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.forecast_horizon_slots = int(forecast_horizon_slots)
        ramp = np.linspace(0.0, 1.0, self.forecast_horizon_slots, dtype=np.float32)
        self._blend_weights = tf.constant(ramp.reshape(1, -1), dtype=tf.float32)

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor] | list[tf.Tensor]) -> tf.Tensor:
        short_delta, long_delta = inputs
        return short_delta * (1.0 - self._blend_weights) + long_delta * self._blend_weights

    def get_config(self) -> dict[str, object]:
        return {
            "forecast_horizon_slots": self.forecast_horizon_slots,
            **super().get_config(),
        }


@keras.saving.register_keras_serializable(package="local_cnn")
class HybridTemperatureAnchor(layers.Layer):
    def __init__(
        self,
        forecast_horizon_slots: int,
        temperature_feature_index: int,
        temperature_feature_mean: float,
        temperature_feature_scale: float,
        temp_mean_24h_feature_index: int | None,
        temp_mean_24h_feature_mean: float,
        temp_mean_24h_feature_scale: float,
        has_climatology_anchor: bool | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.forecast_horizon_slots = int(forecast_horizon_slots)
        self.temperature_feature_index = int(temperature_feature_index)
        self.temperature_feature_mean = float(temperature_feature_mean)
        self.temperature_feature_scale = float(temperature_feature_scale)
        self.temp_mean_24h_feature_index = (
            None if temp_mean_24h_feature_index is None or temp_mean_24h_feature_index < 0
            else int(temp_mean_24h_feature_index)
        )
        self.temp_mean_24h_feature_mean = float(temp_mean_24h_feature_mean)
        self.temp_mean_24h_feature_scale = float(temp_mean_24h_feature_scale)
        inferred_has_climatology_anchor = self.temp_mean_24h_feature_index is not None
        if has_climatology_anchor is not None and bool(has_climatology_anchor) != inferred_has_climatology_anchor:
            logger.debug(
                "Ignoring serialized has_climatology_anchor=%s because temp_mean_24h_feature_index=%s.",
                has_climatology_anchor,
                self.temp_mean_24h_feature_index,
            )
        self.has_climatology_anchor = inferred_has_climatology_anchor

        ramp = np.linspace(0.0, 1.0, self.forecast_horizon_slots, dtype=np.float32)
        persistence_weights = 0.75 - 0.55 * ramp
        seasonal_weights = 0.15 + 0.60 * ramp
        if self.has_climatology_anchor:
            climatology_weights = 1.0 - persistence_weights - seasonal_weights
        else:
            climatology_weights = np.zeros_like(ramp)
            normalization = persistence_weights + seasonal_weights
            persistence_weights = persistence_weights / normalization
            seasonal_weights = seasonal_weights / normalization
        self._persistence_weights = tf.constant(persistence_weights.reshape(1, -1), dtype=tf.float32)
        self._seasonal_weights = tf.constant(seasonal_weights.reshape(1, -1), dtype=tf.float32)
        self._climatology_weights = tf.constant(climatology_weights.reshape(1, -1), dtype=tf.float32)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        last_normalized_temperature = inputs[:, -1, self.temperature_feature_index]
        last_temperature_c = (
            last_normalized_temperature * self.temperature_feature_scale + self.temperature_feature_mean
        )
        persistence_anchor = ops.repeat(
            ops.expand_dims(last_temperature_c, axis=-1),
            repeats=self.forecast_horizon_slots,
            axis=-1,
        )

        seasonal_anchor = inputs[:, : self.forecast_horizon_slots, self.temperature_feature_index]
        seasonal_anchor = (
            seasonal_anchor * self.temperature_feature_scale + self.temperature_feature_mean
        )

        if self.has_climatology_anchor:
            climatology_feature = inputs[:, -1, self.temp_mean_24h_feature_index]
            climatology_anchor = (
                climatology_feature * self.temp_mean_24h_feature_scale + self.temp_mean_24h_feature_mean
            )
            climatology_anchor = ops.repeat(
                ops.expand_dims(climatology_anchor, axis=-1),
                repeats=self.forecast_horizon_slots,
                axis=-1,
            )
        else:
            climatology_anchor = ops.zeros_like(persistence_anchor)

        return (
            persistence_anchor * self._persistence_weights
            + seasonal_anchor * self._seasonal_weights
            + climatology_anchor * self._climatology_weights
        )

    def get_config(self) -> dict[str, object]:
        return {
            "forecast_horizon_slots": self.forecast_horizon_slots,
            "temperature_feature_index": self.temperature_feature_index,
            "temperature_feature_mean": self.temperature_feature_mean,
            "temperature_feature_scale": self.temperature_feature_scale,
            "temp_mean_24h_feature_index": self.temp_mean_24h_feature_index,
            "temp_mean_24h_feature_mean": self.temp_mean_24h_feature_mean,
            "temp_mean_24h_feature_scale": self.temp_mean_24h_feature_scale,
            "has_climatology_anchor": self.has_climatology_anchor,
            **super().get_config(),
        }


@dataclass(slots=True)
class TrainingResult:
    tuned_model: keras.Model
    tuned_validation_mae: float
    fine_tuned_model: keras.Model
    fine_tuned_validation_mae: float


@keras.saving.register_keras_serializable(package="local_cnn")
def horizon_60m_mae(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return ops.mean(ops.abs(y_true[:, 1] - y_pred[:, 1]))


def build_regression_loss(forecast_horizon_slots: int, primary_horizon_index: int = 1) -> keras.losses.Loss:
    return WeightedForecastHuber(
        forecast_horizon_slots=forecast_horizon_slots,
        primary_horizon_index=primary_horizon_index,
        delta=1.5,
    )


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
    temperature_feature_mean: float = 0.0,
    temperature_feature_scale: float = 1.0,
    temperature_feature_index: int = 0,
    temp_mean_24h_feature_mean: float = 0.0,
    temp_mean_24h_feature_scale: float = 1.0,
    temp_mean_24h_feature_index: int | None = None,
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
    pooled_average_features = layers.GlobalAveragePooling1D(
        name="gap_temporal_features"
    )(projected_temporal_features)
    pooled_max_features = layers.GlobalMaxPooling1D(
        name="gmp_temporal_features"
    )(projected_temporal_features)
    latest_temporal_features = LastTimeStep(name="latest_temporal_features")(
        projected_temporal_features
    )
    temporal_summary = layers.Concatenate(name="temporal_summary_features")(
        [pooled_average_features, pooled_max_features, latest_temporal_features]
    )
    short_dense_projection = layers.Dense(
        units=dense_units,
        activation="relu",
        name="dense_short_projection",
    )(temporal_summary)
    regularized_short_features = layers.Dropout(
        rate=dropout_rate,
        name="dropout_regularization",
    )(short_dense_projection)
    short_delta_temperature_horizon = layers.Dense(
        units=forecast_horizon_slots,
        dtype="float32",
        name="predicted_temperature_delta_short_horizon",
    )(regularized_short_features)
    long_term_summary = layers.Concatenate(name="long_term_summary_features")(
        [pooled_average_features, pooled_max_features]
    )
    long_dense_projection = layers.Dense(
        units=max(dense_units // 2, 32),
        activation="relu",
        name="dense_long_projection",
    )(long_term_summary)
    delta_temperature_long_horizon = layers.Dense(
        units=forecast_horizon_slots,
        dtype="float32",
        name="predicted_temperature_delta_long_horizon",
    )(long_dense_projection)
    blended_delta_horizon = HorizonBlend(
        forecast_horizon_slots=forecast_horizon_slots,
        name="blended_temperature_delta_horizon",
    )([short_delta_temperature_horizon, delta_temperature_long_horizon])
    temperature_anchor = HybridTemperatureAnchor(
        forecast_horizon_slots=forecast_horizon_slots,
        temperature_feature_index=temperature_feature_index,
        temperature_feature_mean=temperature_feature_mean,
        temperature_feature_scale=temperature_feature_scale,
        temp_mean_24h_feature_index=temp_mean_24h_feature_index,
        temp_mean_24h_feature_mean=temp_mean_24h_feature_mean,
        temp_mean_24h_feature_scale=temp_mean_24h_feature_scale,
        name="hybrid_temperature_anchor",
    )(input_layer)
    output_temperature = layers.Add(name="predicted_temperature_horizon")(
        [temperature_anchor, blended_delta_horizon]
    )

    model = keras.Model(
        inputs=input_layer,
        outputs=output_temperature,
        name="lightweight_1d_cnn_forecaster",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=build_regression_loss(forecast_horizon_slots=forecast_horizon_slots),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae"), horizon_60m_mae],
    )
    return model


def build_model_for_tuning(
    hp: kt.HyperParameters,
    input_window_length: int,
    number_of_input_features: int,
    forecast_horizon_slots: int,
    temperature_feature_mean: float,
    temperature_feature_scale: float,
    temperature_feature_index: int,
    temp_mean_24h_feature_mean: float,
    temp_mean_24h_feature_scale: float,
    temp_mean_24h_feature_index: int | None,
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
        temperature_feature_mean=temperature_feature_mean,
        temperature_feature_scale=temperature_feature_scale,
        temperature_feature_index=temperature_feature_index,
        temp_mean_24h_feature_mean=temp_mean_24h_feature_mean,
        temp_mean_24h_feature_scale=temp_mean_24h_feature_scale,
        temp_mean_24h_feature_index=temp_mean_24h_feature_index,
        dropout_rate=dropout_rate,
        first_block_filters=first_block_filters,
        second_block_filters=second_block_filters,
        projection_filters=projection_filters,
        dense_units=dense_units,
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=build_regression_loss(forecast_horizon_slots=forecast_horizon_slots),
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
    temperature_feature_index = dataset.feature_columns.index(config.target_column_name)
    temperature_feature_mean = float(dataset.scaler.mean_[temperature_feature_index])
    temperature_feature_scale = float(dataset.scaler.scale_[temperature_feature_index])
    if "temp_mean_24h" in dataset.feature_columns:
        temp_mean_24h_feature_index = dataset.feature_columns.index("temp_mean_24h")
        temp_mean_24h_feature_mean = float(dataset.scaler.mean_[temp_mean_24h_feature_index])
        temp_mean_24h_feature_scale = float(dataset.scaler.scale_[temp_mean_24h_feature_index])
    else:
        temp_mean_24h_feature_index = None
        temp_mean_24h_feature_mean = 0.0
        temp_mean_24h_feature_scale = 1.0
    tuner = kt.RandomSearch(
        hypermodel=lambda hp: build_model_for_tuning(
            hp,
            input_window_length=dataset.X_train.shape[1],
            number_of_input_features=dataset.X_train.shape[-1],
            forecast_horizon_slots=dataset.y_train.shape[-1],
            temperature_feature_mean=temperature_feature_mean,
            temperature_feature_scale=temperature_feature_scale,
            temperature_feature_index=temperature_feature_index,
            temp_mean_24h_feature_mean=temp_mean_24h_feature_mean,
            temp_mean_24h_feature_scale=temp_mean_24h_feature_scale,
            temp_mean_24h_feature_index=temp_mean_24h_feature_index,
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
