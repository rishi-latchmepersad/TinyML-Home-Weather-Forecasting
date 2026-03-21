#ifndef BASELINE_FORECAST_TASK_H_
#define BASELINE_FORECAST_TASK_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

bool baseline_forecast_task_start(void);
bool baseline_forecast_get_latest_prediction(float *temperatures_c_out,
        size_t temperatures_capacity, uint32_t *prediction_sequence_out,
        char *prediction_timestamp_iso8601_out,
        size_t prediction_timestamp_capacity);

#endif
