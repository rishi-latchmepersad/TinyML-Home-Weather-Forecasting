/*
 * sd_spi_helpers.c
 *
 *  Created on: Jul 27, 2025
 *      Author: rishi_latchmepersad
 */


#include "sd_spi_helpers.h"
#include "stm32f7xx_hal_gpio.h"

extern SPI_HandleTypeDef hspi1;  // From main.c or global SPI config
#define CS_GPIO_Port GPIOB
#define CS_Pin GPIO_PIN_6

void SD_Select(void) {
	HAL_GPIO_WritePin(CS_GPIO_Port, CS_Pin, GPIO_PIN_RESET);
}

void SD_Deselect(void) {
	HAL_GPIO_WritePin(CS_GPIO_Port, CS_Pin, GPIO_PIN_SET);
	uint8_t dummy = SD_DUMMY_BYTE;
	HAL_SPI_Transmit(&hspi1, &dummy, 1, HAL_MAX_DELAY);
}

uint8_t SD_Transmit(uint8_t data) {
	uint8_t resp;
	HAL_SPI_TransmitReceive(&hspi1, &data, &resp, 1, HAL_MAX_DELAY);
	return resp;
}

void SD_SendCmd(uint8_t cmd, uint32_t arg, uint8_t crc) {
	uint8_t packet[6];

	packet[0] = 0x40 | cmd;
	packet[1] = (arg >> 24) & 0xFF;
	packet[2] = (arg >> 16) & 0xFF;
	packet[3] = (arg >> 8) & 0xFF;
	packet[4] = arg & 0xFF;
	packet[5] = crc;

	SD_Select();
	for (int i = 0; i < 6; i++) {
		SD_Transmit(packet[i]);
	}
}

uint8_t SD_WaitReady(void) {
	uint8_t res;
	uint32_t timeout = HAL_GetTick();
	do {
		res = SD_Transmit(SD_DUMMY_BYTE);
		if (res == 0xFF)
			return 1;
	} while ((HAL_GetTick() - timeout) < 500);
	return 0;
}
