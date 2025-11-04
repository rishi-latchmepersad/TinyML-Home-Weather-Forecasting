/*
 * bme280_task.h
 *
 *  Created on: Jul 17, 2025
 *      Author: rishi_latchmepersad
 */

#ifndef BME280_BME280_TASK_H_
#define BME280_BME280_TASK_H_

// Pull in the CMSIS-RTOS definitions for mutex and thread primitives.
#include "cmsis_os.h"
// Pull in the Bosch driver definitions for register constants and structs.
#include "bme280_defs.h"
// Pull in stdbool so we can return success flags from helpers.
#include <stdbool.h>

// Public struct for sensor data
typedef struct {
	float temperature;
	float humidity;
	float pressure;
} bme280_data_t;

typedef enum {
	BME280_STATE_INIT,
	BME280_STATE_SETUP,
	BME280_STATE_TRIGGER_MEASUREMENT,
	BME280_STATE_WAIT_MEASUREMENT,
	BME280_STATE_READ_DATA,
	BME280_STATE_DELAY,
	BME280_STATE_ERROR
} bme280_state_t;

typedef struct {
	bme280_state_t state;
	struct bme280_dev dev;
	struct bme280_data data;
	int8_t result;
	uint32_t last_tick;
} bme280_task_data_t;

// Initialization function for testing
void bme280_task_init(bme280_task_data_t *task_data);

// Public task function
void bme280SensorTask(void *argument);
// Allow other tasks to grab the latest BME280 reading when available.
bool bme280_get_latest(float *temperature_c_out, float *humidity_pct_out, float *pressure_pa_out);
#endif /* BME280_TASK_H */
