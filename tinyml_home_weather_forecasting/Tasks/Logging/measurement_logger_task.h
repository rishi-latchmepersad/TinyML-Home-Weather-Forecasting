#pragma once
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#include "FreeRTOS.h"
#include "queue.h"
#include "cmsis_os.h"
#include "ff.h"

#ifdef __cplusplus
extern "C" {
#endif

/****************************************************************************************
 * Type:        measurement_logger_message_t
 * Purpose:     Describe one observation to be logged by the SD-card logger task.
 *
 * Fields:      const char *sensor_name      [in] - e.g., "bme280" (pointer to const string)
 *              const char *quantity_name    [in] - e.g., "temperature_c"
 *              float       value_numeric    [in] - numeric value
 *              const char *unit_name        [in] - e.g., "degC"
 *
 * Notes:
 *              - The strings should point to static const storage (not stack variables).
 *              - Timestamp is added by the logger task from the RTC/DS3231.
 ****************************************************************************************/
typedef struct
{
    const char *sensor_name;
    const char *quantity_name;
    float       value_numeric;
    const char *unit_name;
} measurement_logger_message_t;

/****************************************************************************************
 * Function:    measurement_logger_task_create
 * Purpose:     Create the SD-card logger task and its queue.
 *
 * Returns:     bool
 *                - true  : Task and queue created successfully.
 *                - false : Creation failed (e.g., out of memory).
 *
 * Side Effects:
 *                - Creates an RTOS task and a queue; opens/creates the daily CSV file
 *                  during task execution.
 *
 * Preconditions:
 *                - FreeRTOS kernel initialized but not yet started, or call from a
 *                  context that allows task creation.
 ****************************************************************************************/
bool measurement_logger_task_create(void);

/****************************************************************************************
 * Function:    measurement_logger_enqueue
 * Purpose:     Enqueue one observation for the logger task to persist to CSV.
 *
 * Parameters:  const measurement_logger_message_t *msg [in]
 *              uint32_t timeout_ms                     [in] - wait time for queue space.
 *
 * Returns:     bool  - true on success (queued), false on timeout/failure.
 *
 * Concurrency:
 *                - Thread-safe; may be called from any task context.
 ****************************************************************************************/
bool measurement_logger_enqueue(const measurement_logger_message_t *msg,
                                uint32_t timeout_ms);

/****************************************************************************************
 * Function:    measurement_logger_request_sync
 * Purpose:     Ask the logger task to flush buffered data to the SD card soon.
 * Parameters:  void
 * Returns:     void
 ****************************************************************************************/
void measurement_logger_request_sync(void);

#ifdef __cplusplus
}
#endif
