/* lm393_task.h */
#ifndef SENSORS_LM393_TASK_H_
#define SENSORS_LM393_TASK_H_

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initialize and start the LM393 (rain DO) polling task.
 * @notes Creates a CMSIS-RTOS2 thread that polls and reports rain state.
 */
void lm393_init_and_start_task(void);

#ifdef __cplusplus
}
#endif

#endif /* SENSORS_LM393_TASK_H_ */
