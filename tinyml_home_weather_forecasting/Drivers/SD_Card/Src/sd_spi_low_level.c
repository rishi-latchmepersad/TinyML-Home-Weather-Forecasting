#include "sd_spi_low_level.h"
#include "main.h"
#include "stm32f7xx_hal.h"
#include <stdio.h>

#define LOG_PREFIX "[SD_SPI_LL] "

/* Configure SPI and chip-select according to your CubeMX pins */
extern SPI_HandleTypeDef hspi1;
#define SD_SPI_HANDLE  hspi1
#define SD_CS_PORT     SD_Card_CS_GPIO_Port
#define SD_CS_PIN      SD_Card_CS_Pin

#define SD_SPI_TIMEOUT 1000  /* ms */

/* ===== SPI byte I/O ===== */
void SDLL_SendByte(uint8_t data) {
	HAL_SPI_Transmit(&SD_SPI_HANDLE, &data, 1, SD_SPI_TIMEOUT);
}

uint8_t SDLL_ReadByte(void) {
	uint8_t d = 0xFF;
	HAL_SPI_TransmitReceive(&SD_SPI_HANDLE, &d, &d, 1, SD_SPI_TIMEOUT);
	return d;
}

void SDLL_CS_Low(void) {
	HAL_GPIO_WritePin(SD_CS_PORT, SD_CS_PIN, GPIO_PIN_RESET);
}
void SDLL_CS_High(void) {
	HAL_GPIO_WritePin(SD_CS_PORT, SD_CS_PIN, GPIO_PIN_SET);
}

uint8_t SDLL_WaitReady(uint32_t timeout_ms) {
	uint32_t t = HAL_GetTick() + timeout_ms;
	do {
		if (SDLL_ReadByte() == 0xFF)
			return 1;
	} while (HAL_GetTick() < t);
	return 0;
}

/* ===== SD command (SPI mode) ===== */
/* Barr-C Header
 * Name: SDLL_SendCommand
 * Purpose: Transmit an SD command (SPI mode) and return the R1 status.
 * Params: cmd - command index (bit7 set => ACMD flow), arg - 32-bit argument
 * Returns: uint8_t R1 code (0x00=OK, 0x01=Idle, bit7=busy/no response)
 * Side Effects: Leaves CS **LOW** on return so caller can read any extra bytes
 *               (R3/R7) or subsequent data tokens. Caller must deassert CS.
 * Preconditions: SPI running at slow speed for init or at data speed post-init.
 * Postconditions: None (CS remains low).
 * Concurrency: Not re-entrant; serialize access to the SPI bus.
 * Timing: Bounded by SD response timing + SPI shifts.
 * Errors: Returns 0xFF on timeout waiting for "ready" (non-CMD0 only).
 * Notes: Caller must finish with SDLL_CS_High(); SDLL_SendByte(0xFF).
 */
uint8_t SDLL_SendCommand(uint8_t cmd, uint32_t arg) {
    uint8_t n, res;

    if (cmd & 0x80) {                 /* ACMD<n> sequence */
        cmd &= 0x7F;
        res = SDLL_SendCommand(SD_CMD55, 0);
        if (res > 1) return res;
    }

    SDLL_CS_High();
    SDLL_SendByte(0xFF);
    SDLL_CS_Low();

    if (cmd != SD_CMD0) {
        if (!SDLL_WaitReady(SD_SPI_TIMEOUT)) {
            /* Leave CS high to avoid holding the bus */
            SDLL_CS_High();
            SDLL_SendByte(0xFF);
            return 0xFF;
        }
    } else {
        SDLL_SendByte(0xFF);          /* one extra clock helps some modules */
    }

    SDLL_SendByte(0x40 | cmd);
    SDLL_SendByte((uint8_t)(arg >> 24));
    SDLL_SendByte((uint8_t)(arg >> 16));
    SDLL_SendByte((uint8_t)(arg >> 8));
    SDLL_SendByte((uint8_t)arg);

    n = 0x01;                         /* valid CRC only for CMD0/CMD8 */
    if (cmd == SD_CMD0) n = 0x95;
    if (cmd == SD_CMD8) n = 0x87;
    SDLL_SendByte(n);

    if (cmd == SD_CMD12) SDLL_ReadByte();  /* skip stuff byte */

    n = 20;
    do {
        res = SDLL_ReadByte();
    } while ((res & 0x80) && --n);

    /* IMPORTANT: CS stays LOW here! Caller will finish the transaction. */
    return res;
}


/****************************************************************************************
 * Function:    sd_spi_power_on_sequence
 * Purpose:     Perform the SPI SD power-on handshake so the card enters SPI mode.
 *
 * Parameters:  void
 *
 * Returns:     void
 *
 * Side Effects:
 *                - Drives chip-select (CS) high.
 *                - Shifts at least 80 clock cycles with MOSI held high (0xFF bytes).
 *
 * Preconditions:
 *                - SPI peripheral is initialized at a slow baud rate suitable for
 *                  power-on (e.g., ≤ 400 kHz to a few MHz, per card tolerance).
 *                - SD card is powered and stable.
 *
 * Postconditions:
 *                - Card has received ≥ 80 clocks with CS high, ready for CMD0 with CS low.
 *
 * Concurrency:
 *                - Not reentrant; must be serialized with other SPI users.
 *
 * Timing:
 *                - Runs in microseconds; dominated by SPI clocks for 10 bytes.
 *
 * Errors:
 *                - None directly. If skipped, subsequent commands (CMD0/CMD8) may
 *                  return 0xFF (no response).
 *
 * Notes:
 *                - SD Physical Layer Simplified Spec requires ≥ 74 clock cycles with
 *                  CS high and DI (MOSI) high before the first command in SPI mode.
 ****************************************************************************************/
void sd_spi_power_on_sequence(void)
{
    uint8_t i;

    /* Ensure CS is high (card deselected), then provide ≥ 80 clocks with DI high. */
    SDLL_CS_High();
    for (i = 0U; i < 10U; i++)
    {
        SDLL_SendByte(0xFF);  /* 8 clocks each, MOSI held high */
    }
    SDLL_CS_Low();
    uint8_t spi_probe_byte = SDLL_ReadByte();  /* clocks once with CS low */
    printf(LOG_PREFIX "SPI probe after select: 0x%02X\r\n", spi_probe_byte);
    SDLL_CS_High();
}

/* ===== Data block I/O ===== */
uint8_t SDLL_ReadDataBlock(uint8_t *buff, uint32_t btr) {
	uint8_t token;
	uint32_t t = HAL_GetTick() + SD_SPI_TIMEOUT;

	/* Wait for data packet */
	do {
		token = SDLL_ReadByte();
	} while ((token == 0xFF) && (HAL_GetTick() < t));

	if (token != 0xFE)
		return 0; /* Invalid token */

	/* Receive data */
	do {
		*buff++ = SDLL_ReadByte();
	} while (--btr);

	SDLL_ReadByte(); /* discard CRC */
	SDLL_ReadByte();
	return 1;
}

/*
 * Name: SDLL_WriteDataBlock
 * Purpose: Send a single 512B data block (or STOP token) in SPI mode.
 * Params: buff - 512B data (ignored for STOP), token - data/stop token
 * Returns: 1 on success, 0 on failure or timeout
 * Side Effects: Leaves CS LOW on return; caller must deassert CS.
 * Preconditions: Card selected (CS low) and not busy.
 * Postconditions: Card is no longer busy (write completed) on success.
 * Errors: Returns 0 if data response not accepted or busy never releases.
 */
uint8_t SDLL_WriteDataBlock(const uint8_t *buff, uint8_t token)
{
    uint8_t resp;
    uint32_t bc = 512U;

    if (!SDLL_WaitReady(SD_SPI_TIMEOUT))
        return 0;

    SDLL_SendByte(token);
    if (token != 0xFD) {                 // data block
        do { SDLL_SendByte(*buff++); } while (--bc);
        SDLL_SendByte(0xFF);             // dummy CRC
        SDLL_SendByte(0xFF);

        resp = SDLL_ReadByte();          // data response
        if ((resp & 0x1F) != 0x05)
            return 0;                    // not accepted

        // >>> IMPORTANT: wait until the card releases busy (MISO==0xFF)
        if (!SDLL_WaitReady(500))
            return 0;
    } else {
        // STOP_TRAN for multi-block writes also leaves the card busy
        if (!SDLL_WaitReady(500))
            return 0;
    }
    return 1;
}

