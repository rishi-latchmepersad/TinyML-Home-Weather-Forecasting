from __future__ import annotations

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="local_cnn")
class LatestTemporalFeatures(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs[:, -1, :]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])


@tf.keras.utils.register_keras_serializable(package="local_cnn")
class SeasonalNaiveTrajectory(tf.keras.layers.Layer):
    def __init__(
        self,
        forecast_horizon_slots: int,
        temperature_feature_index: int,
        temperature_feature_mean: float,
        temperature_feature_scale: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.forecast_horizon_slots = int(forecast_horizon_slots)
        self.temperature_feature_index = int(temperature_feature_index)
        self.temperature_feature_mean = float(temperature_feature_mean)
        self.temperature_feature_scale = float(temperature_feature_scale)

    def call(self, inputs):
        trajectory = inputs[:, : self.forecast_horizon_slots, self.temperature_feature_index]
        return trajectory * self.temperature_feature_scale + self.temperature_feature_mean

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.forecast_horizon_slots)

    def get_config(self):
        return {
            **super().get_config(),
            "forecast_horizon_slots": self.forecast_horizon_slots,
            "temperature_feature_index": self.temperature_feature_index,
            "temperature_feature_mean": self.temperature_feature_mean,
            "temperature_feature_scale": self.temperature_feature_scale,
        }


@tf.keras.utils.register_keras_serializable(package="local_cnn")
class HorizonBlend(tf.keras.layers.Layer):
    def __init__(self, weights, **kwargs):
        super().__init__(**kwargs)
        self.blend_schedule = [float(value) for value in weights]

    def call(self, inputs):
        short_delta, long_delta = inputs
        schedule = tf.convert_to_tensor(self.blend_schedule, dtype=short_delta.dtype)[tf.newaxis, :]
        return short_delta * (1.0 - schedule) + long_delta * schedule

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        return {**super().get_config(), "weights": self.blend_schedule}


@tf.keras.utils.register_keras_serializable(package="local_cnn")
class HorizonTrustSchedule(tf.keras.layers.Layer):
    def __init__(self, schedule, **kwargs):
        super().__init__(**kwargs)
        self.schedule = [float(value) for value in schedule]

    def call(self, inputs):
        schedule = tf.convert_to_tensor(self.schedule, dtype=inputs.dtype)[tf.newaxis, :]
        return inputs * schedule

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {**super().get_config(), "schedule": self.schedule}
