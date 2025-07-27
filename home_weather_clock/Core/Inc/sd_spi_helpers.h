/*
 * sd_spi_helpers.h
 *
 *  Created on: Jul 27, 2025
 *      Author: rishi_latchmepersad
 */

#ifndef INC_SD_SPI_HELPERS_H_
#define INC_SD_SPI_HELPERS_H_


#include "main.h"
#include "stm32f7xx_hal.h"
#define SD_DUMMY_BYTE 0xFF

// Function prototypes
void SD_Select(void);
void SD_Deselect(void);
uint8_t SD_Transmit(uint8_t data);
void SD_SendCmd(uint8_t cmd, uint32_t arg, uint8_t crc);
uint8_t SD_WaitReady(void);

#endif /* INC_SD_SPI_HELPERS_H_ */
