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
#include "stdio.h"
#include "sd_spi_helpers.h"
#include "stm32f7xx_hal.h"
#include "main.h"

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
#define SD_DUMMY_BYTE 0xFF
#define SD_TOKEN_OK 0x00
#define SD_TOKEN_DATA 0xFE
/* Private variables ---------------------------------------------------------*/
/* Disk status */
static volatile DSTATUS Stat = STA_NOINIT;
extern SPI_HandleTypeDef hspi1;
// SD Card Status
typedef enum {
    SD_OK = 0,
    SD_ERROR,
    SD_TIMEOUT,
    SD_NOT_INITIALIZED,
    SD_CMD_FAILED,
    SD_UNSUPPORTED_CARD
} SD_Status;

/* USER CODE END DECL */

/* Private function prototypes -----------------------------------------------*/
DSTATUS USER_initialize (BYTE pdrv);
DSTATUS USER_status (BYTE pdrv);
DRESULT USER_read (BYTE pdrv, BYTE *buff, DWORD sector, UINT count);
#if _USE_WRITE == 1
  DRESULT USER_write (BYTE pdrv, const BYTE *buff, DWORD sector, UINT count);
#endif /* _USE_WRITE == 1 */
#if _USE_IOCTL == 1
  DRESULT USER_ioctl (BYTE pdrv, BYTE cmd, void *buff);
#endif /* _USE_IOCTL == 1 */

Diskio_drvTypeDef  USER_Driver =
{
  USER_initialize,
  USER_status,
  USER_read,
#if  _USE_WRITE
  USER_write,
#endif  /* _USE_WRITE == 1 */
#if  _USE_IOCTL == 1
  USER_ioctl,
#endif /* _USE_IOCTL == 1 */
};

/* Private functions ---------------------------------------------------------*/

/**
  * @brief  Initializes a Drive
  * @param  pdrv: Physical drive number (0..)
  * @retval DSTATUS: Operation status
  */
DSTATUS USER_initialize (
	BYTE pdrv           /* Physical drive number to identify the drive */
)
{
  /* USER CODE BEGIN INIT */
	  if (pdrv != 0) return STA_NOINIT;

	    // 1. Power-up sequence (74+ clocks with CS high)
	    SD_Deselect();
	    HAL_Delay(10); // Minimum 1ms delay

	    // Send 80+ clocks (10 bytes)
	    for (int i = 0; i < 10; i++) {
	        SD_Transmit(SD_DUMMY_BYTE);
	    }

	    // 2. CMD0 with retry mechanism
	    uint8_t response;
	    int retry = 0;
	    do {
	        SD_SendCmd(0, 0, 0x95); // CMD0, arg=0, CRC=0x95
	        response = SD_Transmit(SD_DUMMY_BYTE);

	        // Wait for valid response (up to 8 bytes)
	        uint32_t timeout = HAL_GetTick();
	        while (response == 0xFF) {
	            response = SD_Transmit(SD_DUMMY_BYTE);
	            if (HAL_GetTick() - timeout > 500) break;
	        }

	        SD_Deselect();

	        if (response == 0x01) break;
	        if (++retry > 10) {
	            printf("CMD0 failed after 10 attempts\n");
	            return STA_NOINIT;
	        }
	        HAL_Delay(100);
	    } while (1);

	    // 3. CMD8 for SDC V2 voltage check
	    uint8_t r7[5];
	    SD_SendCmd(8, 0x1AA, 0x87); // CMD8, arg=0x1AA, CRC=0x87
	    response = SD_Transmit(SD_DUMMY_BYTE);
	    if (response == 0x01) {
	        r7[0] = response;
	        for (int i = 1; i < 5; i++) {
	            r7[i] = SD_Transmit(SD_DUMMY_BYTE);
	        }
	        printf("CMD8 response: %02X %02X %02X %02X %02X\n",
	               r7[0], r7[1], r7[2], r7[3], r7[4]);
	    }
	    SD_Deselect();

	    // 4. ACMD41 initialization with HCS bit
	    uint32_t timeout = HAL_GetTick();
	    do {
	        SD_SendCmd(55, 0, 0x01); // CMD55
	        response = SD_Transmit(SD_DUMMY_BYTE);
	        SD_Deselect();

	        SD_SendCmd(41, 0x40000000, 0x01); // ACMD41 with HCS bit
	        response = SD_Transmit(SD_DUMMY_BYTE);
	        SD_Deselect();

	        if ((HAL_GetTick() - timeout) > 2000) {
	            printf("ACMD41 timeout\n");
	            return STA_NOINIT;
	        }
	        HAL_Delay(10);
	    } while (response != 0x00);

	    // 5. Read OCR to confirm card voltage
	    uint8_t ocr[4];
	    SD_SendCmd(58, 0, 0x01); // CMD58
	    response = SD_Transmit(SD_DUMMY_BYTE);
	    if (response == 0x00) {
	        for (int i = 0; i < 4; i++) {
	            ocr[i] = SD_Transmit(SD_DUMMY_BYTE);
	        }
	        printf("OCR: %02X %02X %02X %02X\n", ocr[0], ocr[1], ocr[2], ocr[3]);
	    }
	    SD_Deselect();

	    Stat &= ~STA_NOINIT;
	    printf("SD Card initialized successfully\n");
	    return 0;
}

  /* USER CODE END INIT */

/**
  * @brief  Gets Disk Status
  * @param  pdrv: Physical drive number (0..)
  * @retval DSTATUS: Operation status
  */
DSTATUS USER_status (
	BYTE pdrv       /* Physical drive number to identify the drive */
)
{
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
DRESULT USER_read (
	BYTE pdrv,      /* Physical drive nmuber to identify the drive */
	BYTE *buff,     /* Data buffer to store read data */
	DWORD sector,   /* Sector address in LBA */
	UINT count      /* Number of sectors to read */
)
{
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
DRESULT USER_write (
	BYTE pdrv,          /* Physical drive nmuber to identify the drive */
	const BYTE *buff,   /* Data to be written */
	DWORD sector,       /* Sector address in LBA */
	UINT count          /* Number of sectors to write */
)
{
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
DRESULT USER_ioctl (
	BYTE pdrv,      /* Physical drive nmuber (0..) */
	BYTE cmd,       /* Control code */
	void *buff      /* Buffer to send/receive control data */
)
{
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

