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
	BYTE pdrv           /* Physical drive nmuber to identify the drive */
)
{
  /* USER CODE BEGIN INIT */
	  if (pdrv != 0) return STA_NOINIT;

	    // 1. Power-up delay (min 1ms for SD card)
	    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_6, GPIO_PIN_SET); // CS HIGH
	    HAL_Delay(50);

	    // 2. Send 80+ dummy clocks with CS HIGH
	    uint8_t dummy = 0xFF;
	    for (int i = 0; i < 10; i++) {
	        HAL_SPI_Transmit(&hspi1, &dummy, 1, 100);
	    }

	    // 3. Send CMD0 (GO_IDLE_STATE) with CS LOW
	    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_6, GPIO_PIN_RESET); // CS LOW
	    uint8_t cmd0[] = {0x40, 0x00, 0x00, 0x00, 0x00, 0x95};
	    HAL_SPI_Transmit(&hspi1, cmd0, 6, 100);

	    // 4. Wait for response (0x01 = success)
	    uint8_t response;
	    uint32_t timeout = HAL_GetTick();
	    do {
	        HAL_SPI_Receive(&hspi1, &response, 1, 100);
	        if (HAL_GetTick() - timeout > 500) {
	            printf("CMD0 timeout! No response.\n");
	            HAL_GPIO_WritePin(GPIOB, GPIO_PIN_6, GPIO_PIN_SET);
	            return STA_NOINIT;
	        }
	    } while (response == 0xFF);

	    printf("CMD0 response: 0x%02X\n", response);
	    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_6, GPIO_PIN_SET);

	    if (response != 0x01) {
	        printf("SD init failed. Bad response.\n");
	        return STA_NOINIT;
	    }

	    return 0;  // Success

  /* USER CODE END INIT */
}

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

