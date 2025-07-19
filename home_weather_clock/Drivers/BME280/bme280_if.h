/*
 * bme280.h
 *
 *  Created on: Jul 17, 2025
 *      Author: rishi_latchmepersad
 */

#ifndef INC_BME280_SUPPORT_H_
#define INC_BME280_SUPPORT_H_

#include "bme280.h"

BME280_INTF_RET_TYPE bme280_i2c_read(uint8_t reg_addr, uint8_t *reg_data, uint32_t length, void *intf_ptr);
BME280_INTF_RET_TYPE bme280_i2c_write(uint8_t reg_addr, uint8_t *reg_data, uint32_t length, void *intf_ptr);
void bme280_delay_us(uint32_t us, void *context);

#endif
