/*
 * sd_spi_helpers.c - FIXED VERSION
 *
 *  Created on: Jul 27, 2025
 *      Author: rishi_latchmepersad
 */

#include "sd_spi_helpers.h"
#include "stm32f7xx_hal_gpio.h"
#include <stdio.h>

extern SPI_HandleTypeDef hspi1;  // From main.c or global SPI config
#define CS_GPIO_Port GPIOB
#define CS_Pin GPIO_PIN_6

void SD_Select(void) {
    HAL_GPIO_WritePin(CS_GPIO_Port, CS_Pin, GPIO_PIN_RESET);
    HAL_Delay(1); // Small delay after CS assertion - CRITICAL FIX
}

void SD_Deselect(void) {
    HAL_GPIO_WritePin(CS_GPIO_Port, CS_Pin, GPIO_PIN_SET);
    // Send at least 8 clocks after deselecting - CRITICAL FIX
    uint8_t dummy = SD_DUMMY_BYTE;
    HAL_SPI_Transmit(&hspi1, &dummy, 1, HAL_MAX_DELAY);
    HAL_Delay(1); // Small delay after deselection
}

uint8_t SD_Transmit(uint8_t data) {
    uint8_t resp;
    HAL_SPI_TransmitReceive(&hspi1, &data, &resp, 1, HAL_MAX_DELAY);
    return resp;
}

// FIXED: This function was incomplete - it doesn't wait for or return response
uint8_t SD_SendCmd(uint8_t cmd, uint32_t arg, uint8_t crc) {
    uint8_t packet[6];
    uint8_t response = 0xFF;
    uint8_t dummy = SD_DUMMY_BYTE;

    // CRITICAL: Wait for card to be ready before sending command
    if (!SD_WaitReady()) {
        printf("Card not ready before CMD%d\n", cmd);
        return 0xFF;
    }

    packet[0] = 0x40 | cmd;
    packet[1] = (arg >> 24) & 0xFF;
    packet[2] = (arg >> 16) & 0xFF;
    packet[3] = (arg >> 8) & 0xFF;
    packet[4] = arg & 0xFF;
    packet[5] = crc;

    SD_Select();

    // Send command packet
    HAL_SPI_Transmit(&hspi1, packet, 6, 1000);

    // For CMD12, send extra stuff byte
    if (cmd == 12) {
        HAL_SPI_Transmit(&hspi1, &dummy, 1, 1000);
    }

    // Wait for response (up to 10 bytes for some commands)
    for (int i = 0; i < 10; i++) {
        HAL_SPI_TransmitReceive(&hspi1, &dummy, &response, 1, 1000);
        if ((response & 0x80) == 0) { // Valid response has bit 7 = 0
            break;
        }
    }

    return response;
}

uint8_t SD_WaitResponse(uint8_t expected, uint32_t timeout_ms) {
    uint8_t resp;
    uint32_t start = HAL_GetTick();
    do {
        resp = SD_Transmit(SD_DUMMY_BYTE);
        if (resp == expected) return resp;
    } while ((HAL_GetTick() - start) < timeout_ms);
    return 0xFF; // Timeout
}

// FIXED: Return type should be bool (or uint8_t with proper values)
bool SD_WaitReady(void) {
    uint8_t res;
    uint32_t timeout = HAL_GetTick();
    do {
        res = SD_Transmit(SD_DUMMY_BYTE);
        if (res == 0xFF)
            return true;  // Card is ready
        if ((HAL_GetTick() - timeout) > 5000) // Increased timeout
            return false; // Timeout
        HAL_Delay(1); // Small delay between checks
    } while (1);
}

// ADD: New function for transmit/receive that's used in your diskio
uint8_t SD_TransmitReceive(uint8_t data) {
    uint8_t result = 0xFF;
    HAL_SPI_TransmitReceive(&hspi1, &data, &result, 1, 1000);
    return result;
}
