from __future__ import annotations

import logging

import tensorflow as tf

from .config import PipelineConfig

logger = logging.getLogger(__name__)


def configure_tensorflow_runtime(config: PipelineConfig) -> None:
    logger.info("Querying TensorFlow physical GPU devices.")
    available_gpus = tf.config.list_physical_devices("GPU")
    logger.info("Detected GPU devices: %s", [device.name for device in available_gpus])
    logger.info("Querying TensorFlow physical CPU devices.")
    available_cpus = tf.config.list_physical_devices("CPU")
    logger.info("Detected CPU devices: %s", [device.name for device in available_cpus])

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
                logger.info("Enabling TensorFlow memory growth for %s.", device.name)
                tf.config.experimental.set_memory_growth(device, True)
            except RuntimeError:
                logger.warning("Could not enable memory growth for %s.", device.name, exc_info=True)

    if available_gpus:
        logger.info("TensorFlow will use GPU device(s): %s", [device.name for device in available_gpus])
    else:
        logger.info("TensorFlow did not detect a GPU. Training will run on CPU.")
    logger.info("Visible CPUs: %s", [device.name for device in available_cpus])
