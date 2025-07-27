/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file    user_diskio.c
 * @brief   This file includes a diskio driver skeleton to be completed by the user.
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2025 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */
/* USER CODE END Header */

#ifdef USE_OBSOLETE_USER_CODE_SECTION_0
/*
 * Warning: the user section 0 is no more in use (starting from CubeMx version 4.16.0)
 * To be suppressed in the future.
 * Kept to ensure backward compatibility with previous CubeMx versions when
 * migrating projects.
 * User code previously added there should be copied in the new user sections before
 * the section contents can be deleted.
 */
/* USER CODE BEGIN 0 */
/* USER CODE END 0 */
#endif

/* USER CODE BEGIN DECL */

/* Includes ------------------------------------------------------------------*/
#include <string.h>
#include "ff_gen_drv.h"
#include "diskio.h"
#include "stm32f7xx_hal.h"
#include "main.h"

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
#define CS_GPIO_Port GPIOB
#define CS_Pin GPIO_PIN_6
#define SD_DUMMY_BYTE 0xFF
#define SD_TOKEN_OK 0x00
#define SD_TOKEN_DATA 0xFE
/* Private variables ---------------------------------------------------------*/
/* Disk status */
static volatile DSTATUS Stat = STA_NOINIT;
extern SPI_HandleTypeDef hspi1;
/* USER CODE END DECL */

/* Private function prototypes -----------------------------------------------*/
DSTATUS USER_initialize(BYTE pdrv);
DSTATUS USER_status(BYTE pdrv);
DRESULT USER_read(BYTE pdrv, BYTE *buff, DWORD sector, UINT count);
#if _USE_WRITE == 1
DRESULT USER_write(BYTE pdrv, const BYTE *buff, DWORD sector, UINT count);
#endif /* _USE_WRITE == 1 */
#if _USE_IOCTL == 1
DRESULT USER_ioctl(BYTE pdrv, BYTE cmd, void *buff);
#endif /* _USE_IOCTL == 1 */

Diskio_drvTypeDef USER_Driver = { USER_initialize, USER_status, USER_read,
#if  _USE_WRITE
		USER_write,
#endif  /* _USE_WRITE == 1 */
#if  _USE_IOCTL == 1
		USER_ioctl,
#endif /* _USE_IOCTL == 1 */
		};

/* Private functions ---------------------------------------------------------*/
#ifndef __USER_DISKIO_SPI_HELPERS
#define __USER_DISKIO_SPI_HELPERS
/* USER CODE BEGIN FUNCTIONS */
static void SD_Select(void) {
	HAL_GPIO_WritePin(CS_GPIO_Port, CS_Pin, GPIO_PIN_RESET);
}

static void SD_Deselect(void) {
	HAL_GPIO_WritePin(CS_GPIO_Port, CS_Pin, GPIO_PIN_SET);
	uint8_t dummy = SD_DUMMY_BYTE;
	HAL_SPI_Transmit(&hspi1, &dummy, 1, HAL_MAX_DELAY);
}

static uint8_t SD_Transmit(uint8_t data) {
	uint8_t resp;
	HAL_SPI_TransmitReceive(&hspi1, &data, &resp, 1, HAL_MAX_DELAY);
	return resp;
}

static void SD_SendCmd(uint8_t cmd, uint32_t arg, uint8_t crc) {
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

static uint8_t SD_WaitReady(void) {
	uint8_t res;
	uint32_t timeout = HAL_GetTick();
	do {
		res = SD_Transmit(SD_DUMMY_BYTE);
		if (res == 0xFF)
			return 1;
	} while ((HAL_GetTick() - timeout) < 500);
	return 0;
}
/* USER CODE END FUNCTIONS */
#endif
/**
 * @brief  Initializes a Drive
 * @param  pdrv: Physical drive number (0..)
 * @retval DSTATUS: Operation status
 */
DSTATUS USER_initialize(BYTE pdrv /* Physical drive nmuber to identify the drive */
) {
	/* USER CODE BEGIN INIT */
	if (pdrv != 0)
		return STA_NOINIT;

	uint8_t resp, cmd, retry;
	uint32_t timeout;

	// 1. Initialize CS pin and send 80+ clock cycles
	SD_Deselect();
	HAL_Delay(1);
	for (int i = 0; i < 10; i++)
		SD_Transmit(SD_DUMMY_BYTE);

	// 2. Send CMD0 to reset card (GO_IDLE_STATE)
	SD_SendCmd(0, 0, 0x95);
	do {
		resp = SD_Transmit(SD_DUMMY_BYTE);
	} while ((resp != 0x01) && (HAL_GetTick() - timeout < 1000));

	if (resp != 0x01) {
		SD_Deselect();
		return STA_NOINIT;
	}

	// 3. Send CMD8 to check SDv2 (SEND_IF_COND)
	SD_SendCmd(8, 0x1AA, 0x87);
	resp = SD_Transmit(SD_DUMMY_BYTE);
	if (resp == 0x01) { // SDv2 card
		// Read 4 bytes (should echo back 0x1AA)
		uint8_t ocr[4];
		for (int i = 0; i < 4; i++)
			ocr[i] = SD_Transmit(SD_DUMMY_BYTE);

		// 4. Initialize with ACMD41 (SD_SEND_OP_COND)
		timeout = HAL_GetTick();
		do {
			SD_SendCmd(55, 0, 0x65); // CMD55 precedes ACMD
			SD_SendCmd(41, 0x40000000, 0x77); // ACMD41 with HCS bit
			resp = SD_Transmit(SD_DUMMY_BYTE);
		} while (resp != 0x00 && (HAL_GetTick() - timeout < 1000));
	} else { // SDv1 or MMC
			 // Try simpler initialization
		timeout = HAL_GetTick();
		do {
			SD_SendCmd(1, 0, 0xF9); // CMD1 (MMC) or ACMD41 (SDv1)
			resp = SD_Transmit(SD_DUMMY_BYTE);
		} while (resp != 0x00 && (HAL_GetTick() - timeout < 1000));
	}

	if (resp != 0x00) {
		SD_Deselect();
		return STA_NOINIT;
	}

	// 5. Set block length to 512 bytes (CMD16)
	SD_SendCmd(16, 512, 0x01);
	if (SD_Transmit(SD_DUMMY_BYTE) != 0x00) {
		SD_Deselect();
		return STA_NOINIT;
	}

	Stat &= ~STA_NOINIT; // Clear initialization flag
	SD_Deselect();
	return Stat;

	/* USER CODE END INIT */
}

/**
 * @brief  Gets Disk Status
 * @param  pdrv: Physical drive number (0..)
 * @retval DSTATUS: Operation status
 */
DSTATUS USER_status(BYTE pdrv /* Physical drive number to identify the drive */
) {
	/* USER CODE BEGIN STATUS */
	return (pdrv == 0) ? Stat : STA_NOINIT;
	/* USER CODE END STATUS */
}

/**
 * @brief  Reads Sector(s)
 * @param  pdrv: Physical drive number (0..)
 * @param  *buff: Data buffer to store read data
 * @param  sector: Sector address (LBA)
 * @param  count: Number of sectors to read (1..128)
 * @retval DRESULT: Operation result
 */
DRESULT USER_read(BYTE pdrv, /* Physical drive nmuber to identify the drive */
BYTE *buff, /* Data buffer to store read data */
DWORD sector, /* Sector address in LBA */
UINT count /* Number of sectors to read */
) {
	/* USER CODE BEGIN READ */
	if (pdrv != 0 || !count)
		return RES_PARERR;
	if (Stat & STA_NOINIT)
		return RES_NOTRDY;

	// FatFS expects LBA = sector, 512 bytes
	SD_SendCmd(17, sector, 0x01);

	// Wait for token
	uint32_t timeout = HAL_GetTick();
	while (SD_Transmit(SD_DUMMY_BYTE) != SD_TOKEN_DATA) {
		if ((HAL_GetTick() - timeout) > 1000) {
			SD_Deselect();
			return RES_ERROR;
		}
	}

	// Read 512 bytes
	HAL_SPI_Receive(&hspi1, buff, 512, HAL_MAX_DELAY);

	// Discard CRC
	SD_Transmit(SD_DUMMY_BYTE);
	SD_Transmit(SD_DUMMY_BYTE);

	SD_Deselect();
	return RES_OK;
	/* USER CODE END READ */
}

/**
 * @brief  Writes Sector(s)
 * @param  pdrv: Physical drive number (0..)
 * @param  *buff: Data to be written
 * @param  sector: Sector address (LBA)
 * @param  count: Number of sectors to write (1..128)
 * @retval DRESULT: Operation result
 */
#if _USE_WRITE == 1
DRESULT USER_write(BYTE pdrv, /* Physical drive nmuber to identify the drive */
const BYTE *buff, /* Data to be written */
DWORD sector, /* Sector address in LBA */
UINT count /* Number of sectors to write */
) {
	/* USER CODE BEGIN WRITE */
	/* USER CODE HERE */
	if (pdrv != 0 || !count)
		return RES_PARERR;
	if (Stat & STA_NOINIT)
		return RES_NOTRDY;

	SD_SendCmd(24, sector, 0x01);
	SD_Transmit(SD_TOKEN_DATA);
	HAL_SPI_Transmit(&hspi1, (uint8_t*) buff, 512, HAL_MAX_DELAY);

	// Dummy CRC
	SD_Transmit(SD_DUMMY_BYTE);
	SD_Transmit(SD_DUMMY_BYTE);

	uint8_t resp = SD_Transmit(SD_DUMMY_BYTE);
	if ((resp & 0x1F) != 0x05) {
		SD_Deselect();
		return RES_ERROR;
	}

	SD_WaitReady();
	SD_Deselect();
	return RES_OK;
	/* USER CODE END WRITE */
}
#endif /* _USE_WRITE == 1 */

/**
 * @brief  I/O control operation
 * @param  pdrv: Physical drive number (0..)
 * @param  cmd: Control code
 * @param  *buff: Buffer to send/receive control data
 * @retval DRESULT: Operation result
 */
#if _USE_IOCTL == 1
DRESULT USER_ioctl(BYTE pdrv, /* Physical drive nmuber (0..) */
BYTE cmd, /* Control code */
void *buff /* Buffer to send/receive control data */
) {
	/* USER CODE BEGIN IOCTL */
	if (pdrv != 0)
		return RES_PARERR;

	switch (cmd) {
	case CTRL_SYNC:
		SD_Select();
		if (SD_WaitReady()) {
			SD_Deselect();
			return RES_OK;
		} else {
			SD_Deselect();
			return RES_ERROR;
		}
	case GET_SECTOR_COUNT:
		*(DWORD*) buff = 32768; // Example: 16MB
		return RES_OK;
	case GET_SECTOR_SIZE:
		*(WORD*) buff = 512;
		return RES_OK;
	case GET_BLOCK_SIZE:
		*(DWORD*) buff = 1;
		return RES_OK;
	default:
		return RES_PARERR;
	}
	/* USER CODE END IOCTL */
}
#endif /* _USE_IOCTL == 1 */
