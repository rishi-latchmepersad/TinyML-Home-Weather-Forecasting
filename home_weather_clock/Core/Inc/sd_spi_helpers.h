/*
 * sd_spi_helpers.h - FIXED VERSION
 *
 *  Created on: Jul 27, 2025
 *      Author: rishi_latchmepersad
 */

#ifndef SD_SPI_HELPERS_H
#define SD_SPI_HELPERS_H

#include "stm32f7xx_hal.h"
#include <stdbool.h>

// SD Card constants
#define SD_DUMMY_BYTE   0xFF
#define SD_TOKEN_DATA   0xFE

// Function prototypes
void SD_Select(void);
void SD_Deselect(void);
uint8_t SD_Transmit(uint8_t data);
uint8_t SD_TransmitReceive(uint8_t data);  // Added this function
uint8_t SD_SendCmd(uint8_t cmd, uint32_t arg, uint8_t crc);  // Fixed return type
uint8_t SD_WaitResponse(uint8_t expected, uint32_t timeout_ms);
bool SD_WaitReady(void);  // Fixed return type

#endif /* SD_SPI_HELPERS_H */
