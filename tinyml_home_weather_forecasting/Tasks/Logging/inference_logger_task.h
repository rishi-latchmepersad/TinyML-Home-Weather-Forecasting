#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Create the inference logger background task.
 *
 * The task periodically queries the inference subsystem for the latest
 * predicted temperature and appends it to the daily inference CSV.
 *
 * @return true if the task was created successfully, false otherwise.
 */
bool inference_logger_task_create(void);

#ifdef __cplusplus
}
#endif

