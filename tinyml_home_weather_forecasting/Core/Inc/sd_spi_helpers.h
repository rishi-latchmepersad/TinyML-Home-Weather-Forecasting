/* FIXED sd_spi_helpers.h */
#ifndef SD_SPI_HELPERS_H
#define SD_SPI_HELPERS_H

#include "stm32f7xx_hal.h"
#include <stdbool.h>

#define SD_DUMMY_BYTE 0xFF

// Function declarations
void SD_Select(void);
void SD_Deselect(void);
uint8_t SD_Transmit(uint8_t data);
uint8_t SD_TransmitReceive(uint8_t data);
uint8_t SD_SendCmd(uint8_t cmd, uint32_t arg, uint8_t crc);
uint8_t SD_WaitResponse(uint8_t *buffer, uint8_t count); // FIXED signature
bool SD_WaitReady(void);
void SD_DumpRegisters(void);

#endif /* SD_SPI_HELPERS_H */
