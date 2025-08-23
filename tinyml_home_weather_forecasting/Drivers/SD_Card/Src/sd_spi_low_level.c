#include "sd_spi_low_level.h"
#include "main.h"
#include "stm32f7xx_hal.h"

/* Configure SPI and chip-select according to your CubeMX pins */
extern SPI_HandleTypeDef hspi1;
#define SD_SPI_HANDLE  hspi1
#define SD_CS_PORT     SD_Card_CS_GPIO_Port
#define SD_CS_PIN      SD_Card_CS_Pin

#define SD_SPI_TIMEOUT 200  /* ms */

/* ===== SPI byte I/O ===== */
void SDLL_SendByte(uint8_t data) {
    HAL_SPI_Transmit(&SD_SPI_HANDLE, &data, 1, SD_SPI_TIMEOUT);
}

uint8_t SDLL_ReadByte(void) {
    uint8_t d = 0xFF;
    HAL_SPI_TransmitReceive(&SD_SPI_HANDLE, &d, &d, 1, SD_SPI_TIMEOUT);
    return d;
}

void SDLL_CS_Low(void)  { HAL_GPIO_WritePin(SD_CS_PORT, SD_CS_PIN, GPIO_PIN_RESET); }
void SDLL_CS_High(void) { HAL_GPIO_WritePin(SD_CS_PORT, SD_CS_PIN, GPIO_PIN_SET); }

uint8_t SDLL_WaitReady(uint32_t timeout_ms) {
    uint32_t t = HAL_GetTick() + timeout_ms;
    do {
        if (SDLL_ReadByte() == 0xFF) return 1;
    } while (HAL_GetTick() < t);
    return 0;
}

/* ===== SD command (SPI mode) ===== */
uint8_t SDLL_SendCommand(uint8_t cmd, uint32_t arg) {
    uint8_t n, res;

    if (cmd & 0x80) {             /* ACMD<n> is the command sequence of CMD55-CMD<n> */
        cmd &= 0x7F;
        res = SDLL_SendCommand(SD_CMD55, 0);
        if (res > 1) return res;
    }

    /* Select the card and wait for ready */
    SDLL_CS_High();
    SDLL_SendByte(0xFF);
    SDLL_CS_Low();

    if (!SDLL_WaitReady(SD_SPI_TIMEOUT)) return 0xFF;

    /* Send command packet */
    SDLL_SendByte(0x40 | cmd);           /* Start + Command index */
    SDLL_SendByte((uint8_t)(arg >> 24)); /* Argument[31..24] */
    SDLL_SendByte((uint8_t)(arg >> 16)); /* Argument[23..16] */
    SDLL_SendByte((uint8_t)(arg >> 8));  /* Argument[15..8] */
    SDLL_SendByte((uint8_t)arg);         /* Argument[7..0] */

    /* CRC: only meaningful for CMD0 and CMD8 in SPI mode */
    n = 0x01;
    if (cmd == SD_CMD0) n = 0x95;
    if (cmd == SD_CMD8) n = 0x87;
    SDLL_SendByte(n);

    /* Receive command response */
    if (cmd == SD_CMD12) SDLL_ReadByte(); /* Skip a stuff byte when stop reading */

    n = 10; /* Wait for a valid response within 10 attempts */
    do {
        res = SDLL_ReadByte();
    } while ((res & 0x80) && --n);

    return res;
}

/* ===== Data block I/O ===== */
uint8_t SDLL_ReadDataBlock(uint8_t *buff, uint32_t btr) {
    uint8_t token;
    uint32_t t = HAL_GetTick() + SD_SPI_TIMEOUT;

    /* Wait for data packet */
    do {
        token = SDLL_ReadByte();
    } while ((token == 0xFF) && (HAL_GetTick() < t));

    if (token != 0xFE) return 0; /* Invalid token */

    /* Receive data */
    do {
        *buff++ = SDLL_ReadByte();
    } while (--btr);

    SDLL_ReadByte(); /* discard CRC */
    SDLL_ReadByte();
    return 1;
}

uint8_t SDLL_WriteDataBlock(const uint8_t *buff, uint8_t token) {
    uint8_t resp;
    uint32_t bc = 512U;

    if (!SDLL_WaitReady(SD_SPI_TIMEOUT)) return 0;

    SDLL_SendByte(token);
    if (token != 0xFD) {          /* Is data token */
        do {
            SDLL_SendByte(*buff++);
        } while (--bc);
        SDLL_SendByte(0xFF);      /* CRC (dummy) */
        SDLL_SendByte(0xFF);

        resp = SDLL_ReadByte();   /* Data response */
        if ((resp & 0x1F) != 0x05) return 0;  /* Not accepted */
    }
    return 1;
}
