/*
 * bme280_task.h
 *
 *  Created on: Jul 17, 2025
 *      Author: rishi_latchmepersad
 */

#ifndef BME280_BME280_TASK_H_
#define BME280_BME280_TASK_H_

#include "cmsis_os.h"

// Public struct for sensor data (optional)
typedef struct {
    float temperature;
    float humidity;
    float pressure;
} bme280_data_t;

// Public task function
void bme280SensorTask(void *argument);
#endif /* BME280_TASK_H */
