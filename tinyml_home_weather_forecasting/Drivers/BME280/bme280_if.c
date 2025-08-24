/*
 * bme280.c
 *
 *  Created on: Jul 17, 2025
 *      Author: rishi_latchmepersad
 */


#include "bme280_defs.h"
#include "main.h"

extern I2C_HandleTypeDef hi2c2; // Depending on which I2C we're using

// BME280 I2C read function
BME280_INTF_RET_TYPE bme280_i2c_read(uint8_t reg_addr, uint8_t *reg_data, uint32_t length, void *intf_ptr)
{
	// Retrieve the I2C address from dev->intf_ptr
	uint8_t dev_id_from_intf = *(uint8_t*)intf_ptr;
    HAL_StatusTypeDef status = HAL_I2C_Mem_Read(&hi2c2, dev_id_from_intf, reg_addr, I2C_MEMADD_SIZE_8BIT, reg_data, length, 5000);
    return (status == HAL_OK) ? 0 : -1;
}

// BME280 I2C write function
BME280_INTF_RET_TYPE bme280_i2c_write(uint8_t reg_addr, const uint8_t *reg_data, uint32_t length, void *intf_ptr)
{
	// Retrieve the I2C address from dev->intf_ptr
	uint8_t dev_id_from_intf = *(uint8_t*)intf_ptr;
    HAL_StatusTypeDef status = HAL_I2C_Mem_Write(&hi2c2, dev_id_from_intf, reg_addr, I2C_MEMADD_SIZE_8BIT, (uint8_t *) reg_data, length, 5000);
    return (status == HAL_OK) ? 0 : -1;
}

// Delay function for BME280 driver
void bme280_delay_us(uint32_t us, void *context)
{
    HAL_Delay(us / 1000); // Convert µs to ms
}
