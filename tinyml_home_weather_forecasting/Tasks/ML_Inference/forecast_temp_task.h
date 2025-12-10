// Prevent multiple inclusion of this header.
#ifndef FORECAST_TEMP_TASK_H_
// Define the header guard now that we know the file is being included for the first time.
#define FORECAST_TEMP_TASK_H_

// Pull in the CMSIS-RTOS definitions for thread management primitives.
#include "cmsis_os.h"
// Pull in the auto-generated network header so callers know the output shape.
#include "forecast_temp_ml_model.h"
// Pull in standard integer types used for configuration values.
#include <stdint.h>
// Pull in the standard boolean type so our API can use bool cleanly.
#include <stdbool.h>
// Pull in size_t so APIs can validate buffer capacities.
#include <stddef.h>

// Declare how many minutes of history compose a single 30-minute slot.
#ifndef FORECAST_TEMP_MINUTES_PER_SLOT
#define FORECAST_TEMP_MINUTES_PER_SLOT      (30u)
#endif

// Declare how many 30-minute slots the model predicts per inference.
#define FORECAST_TEMP_FORECAST_HORIZON_SLOTS (AI_FORECAST_TEMP_ML_MODEL_OUT_1_SIZE)

// Expose a helper that spins up the forecasting task during system initialization.
bool forecast_temp_task_start(void);
// Expose a helper that reports the latest predicted temperature vector and inference time when available.
bool forecast_temp_get_latest_prediction(float *temperatures_c_out, size_t temperatures_capacity, uint32_t *inference_time_ms_out);

// Close the header guard.
#endif  // FORECAST_TEMP_TASK_H_
