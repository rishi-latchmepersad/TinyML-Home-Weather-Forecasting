// Prevent multiple inclusion of this header.
#ifndef FORECAST_TEMP_TASK_H_
// Define the header guard now that we know the file is being included for the first time.
#define FORECAST_TEMP_TASK_H_

// Pull in the CMSIS-RTOS definitions for thread management primitives.
#include "cmsis_os.h"
// Pull in standard integer types used for configuration values.
#include <stdint.h>
// Pull in the standard boolean type so our API can use bool cleanly.
#include <stdbool.h>

// Expose a helper that spins up the forecasting task during system initialization.
bool forecast_temp_task_start(void);
// Expose a helper that reports the latest predicted temperature when available.
bool forecast_temp_get_latest_prediction(float *temperature_c_out);

// Close the header guard.
#endif  // FORECAST_TEMP_TASK_H_
