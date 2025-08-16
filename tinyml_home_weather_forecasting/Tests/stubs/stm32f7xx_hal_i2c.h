#ifndef __STM32F7xx_HAL_I2C_H
#define __STM32F7xx_HAL_I2C_H

#include "stm32f7xx_hal.h"

#define HAL_MAX_DELAY 1000

int HAL_I2C_Mem_Read(I2C_HandleTypeDef *hi2c, uint16_t devAddr, uint16_t memAddr,
                     uint16_t memAddSize, uint8_t *pData, uint16_t size, uint32_t timeout) {
    for (int i = 0; i < size; ++i) pData[i] = 0;
    return HAL_OK;
}

#endif
