/*
 * bme280_task.c
 *
 *  Created on: Jul 17, 2025
 *      Author: rishi_latchmepersad
 */

#include "bme280_defs.h"
#include "bme280_support.h"
#include "bme280.h"
#include <stdio.h>
#include "main.h"
#include "cmsis_os.h"


void bme280SensorTask(void *argument)
{
    struct bme280_dev dev;
    struct bme280_data data;
    struct bme280_settings settings;
    int8_t result;

    uint8_t dev_addr = (BME280_I2C_ADDR_PRIM << 1);
    // Set up function pointers
    dev.read = bme280_i2c_read;
    dev.write = bme280_i2c_write;
    dev.delay_us = bme280_delay_us;
    dev.intf = BME280_I2C_INTF;
    dev.intf_ptr = &dev_addr;

    // Set up the BME280
    printf("Initializing BME280...\r\n");

    result = bme280_init(&dev);
    if (result != BME280_OK) {
        printf("BME280 initialization failed! Error code: %d\r\n", result);
        while (1); // Stop execution
    }

    // Set up oversampling and filter settings
    settings.osr_h = BME280_OVERSAMPLING_1X;
    settings.osr_p = BME280_OVERSAMPLING_1X;
    settings.osr_t = BME280_OVERSAMPLING_1X;
    settings.filter = BME280_FILTER_COEFF_OFF;
    settings.standby_time = BME280_STANDBY_TIME_1000_MS;

    result = bme280_set_sensor_settings(BME280_SEL_ALL_SETTINGS,&settings, &dev);
    if (result != BME280_OK) {
        printf("Failed to set BME280 settings! Error code: %d\r\n", result);
        while (1);
    }

    // Main task loop
    while (1)
    {
        // Trigger a measurement
        bme280_set_sensor_mode(BME280_ALL, &dev);

        // Wait for measurement to complete
        dev.delay_us(120000, dev.intf_ptr); // Wait ~120ms

        // Get the results
        result = bme280_get_sensor_data(BME280_ALL, &data, &dev);
        if (result == BME280_OK)
        {
            printf("Temperature: %.2f °C\r\n", data.temperature);
            printf("Pressure: %.2f Pa\r\n", data.pressure);
            printf("Humidity: %.2f %%\r\n", data.humidity);
        }
        else
        {
            printf("Failed to read BME280 data! Error code: %d\r\n", result);
        }

        // Delay before next reading
        vTaskDelay(pdMS_TO_TICKS(2000)); // 2 seconds
    }
}
