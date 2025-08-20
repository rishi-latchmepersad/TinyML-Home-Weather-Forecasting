/* USER CODE BEGIN Header */
/**
 ******************************************************************************
  * @file    user_diskio.h
  * @brief   Public API for the USER Disk I/O driver (SPI SD card).
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

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __USER_DISKIO_H
#define __USER_DISKIO_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
/* Disk I/O types (DSTATUS/DRESULT/BYTE/UINT/DWORD) & control codes */
#include "diskio.h"
/* Driver struct declaration (Diskio_drvTypeDef) and link helpers */
#include "ff_gen_drv.h"

/* Exported types ------------------------------------------------------------*/
/* none */

/* Exported constants --------------------------------------------------------*/
/* none */

/* Exported functions --------------------------------------------------------*/
/**
 * @brief Disk I/O driver instance for FatFs (link with FATFS_LinkDriverEx).
 */
extern Diskio_drvTypeDef USER_Driver;

/**
 * @brief Low-level driver entry points used by USER_Driver.
 * @note  Prototypes are exposed for unit tests or optional direct calls.
 */
DSTATUS USER_initialize (BYTE pdrv);
DSTATUS USER_status     (BYTE pdrv);
DRESULT USER_read       (BYTE pdrv, BYTE *buff, DWORD sector, UINT count);

#if _USE_WRITE == 1
DRESULT USER_write      (BYTE pdrv, const BYTE *buff, DWORD sector, UINT count);
#endif

#if _USE_IOCTL == 1
DRESULT USER_ioctl      (BYTE pdrv, BYTE cmd, void *buff);
#endif

#ifdef __cplusplus
}
#endif

#endif /* __USER_DISKIO_H */
