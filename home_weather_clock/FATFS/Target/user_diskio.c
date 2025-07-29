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
#include "stdbool.h"

/* Private typedef -----------------------------------------------------------*/
/* Private define ------------------------------------------------------------*/
#define SD_DUMMY_BYTE 0xFF
#define SD_TOKEN_OK 0x00
#define SD_TOKEN_DATA 0xFE
#define SD_MAX_RETRIES 100

/* Private variables ---------------------------------------------------------*/
/* Disk status */
static volatile DSTATUS Stat = STA_NOINIT;
extern SPI_HandleTypeDef hspi1;
static bool sd_card_type_sdhc = false;

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

// Helper function to send command and get response
static uint8_t SD_SendCommand(uint8_t cmd, uint32_t arg, uint8_t crc) {
    uint8_t response;

    // Use the helper function that now properly waits and returns response
    response = SD_SendCmd(cmd, arg, crc);

    // Don't deselect here - let the calling function handle it
    // This allows reading additional response bytes if needed

    return response;
}

// Helper function to read additional response bytes
static void SD_ReadResponse(uint8_t *buffer, uint8_t length) {
	uint8_t dummy = SD_DUMMY_BYTE;
	for (int i = 0; i < length; i++) {
		HAL_SPI_TransmitReceive(&hspi1, &dummy, &buffer[i], 1, 1000);
	}
}

/**
 * @brief  Initializes a Drive
 * @param  pdrv: Physical drive number (0..)
 * @retval DSTATUS: Operation status
 */
DSTATUS USER_initialize(BYTE pdrv /* Physical drive nmuber to identify the drive */
) {
	/* USER CODE BEGIN INIT */
	  if (pdrv != 0) return STA_NOINIT;

	    printf("Initializing SD card via diskio...\n");

	    // 1. Power-up sequence
	    SD_Deselect();
	    HAL_Delay(100);

	    // Send 80+ clocks with CS high
	    for (int i = 0; i < 10; i++) {
	        SD_Transmit(SD_DUMMY_BYTE);
	    }
	    HAL_Delay(10);

	    // 2. Send CMD0 (GO_IDLE_STATE)
	    uint8_t response;
	    int attempts = 0;

	    do {
	        response = SD_SendCommand(0, 0, 0x95);
	        SD_Deselect();

	        if (response == 0x01) {
	            printf("CMD0 successful: 0x%02X\n", response);
	            break;
	        }

	        if (++attempts >= 10) {
	            printf("CMD0 failed after %d attempts\n", attempts);
	            return STA_NOINIT;
	        }

	        HAL_Delay(100);
	    } while (response != 0x01);

	    // 3. Send CMD8 to check voltage compatibility
	    response = SD_SendCommand(8, 0x000001AA, 0x87);

	    uint8_t r7_data[4];
	    SD_ReadResponse(r7_data, 4);
	    SD_Deselect();

	    printf("CMD8 R1: 0x%02X, R7 data: %02X %02X %02X %02X\n", response,
	            r7_data[0], r7_data[1], r7_data[2], r7_data[3]);

	    bool is_sdc_v2 = (response == 0x01) && (r7_data[2] == 0x01) && (r7_data[3] == 0xAA);
	    printf("SDC V2 compatible: %s\n", is_sdc_v2 ? "Yes" : "No");

	    // 4. Initialize with ACMD41
	    uint32_t timeout = HAL_GetTick();
	    uint32_t hcs_arg = is_sdc_v2 ? 0x40000000 : 0x00000000;

	    printf("Starting ACMD41 initialization...\n");

	    do {
	        // Send CMD55 first
	        response = SD_SendCommand(55, 0, 0x01);
	        SD_Deselect();

	        if (response > 0x01) {
	            printf("CMD55 failed: 0x%02X\n", response);
	            return STA_NOINIT;
	        }

	        HAL_Delay(1);

	        // Send ACMD41
	        response = SD_SendCommand(41, hcs_arg, 0x01);
	        SD_Deselect();

	        if ((HAL_GetTick() - timeout) > 2000) {
	            printf("ACMD41 timeout\n");
	            return STA_NOINIT;
	        }

	        if (response == 0x00) {
	            printf("ACMD41 successful: 0x%02X\n", response);
	            break;
	        }

	        HAL_Delay(50);
	    } while (response != 0x00);

	    // 5. Read OCR to determine card type
	    response = SD_SendCommand(58, 0, 0x01);

	    uint8_t ocr_data[4];
	    SD_ReadResponse(ocr_data, 4);
	    SD_Deselect();

	    printf("CMD58 R1: 0x%02X, OCR: %02X %02X %02X %02X\n", response,
	            ocr_data[0], ocr_data[1], ocr_data[2], ocr_data[3]);

	    if (response == 0x00) {
	        // Check CCS bit (bit 30) to determine card type
	        sd_card_type_sdhc = (ocr_data[0] & 0x40) ? true : false;
	        printf("Card type: %s\n", sd_card_type_sdhc ? "SDHC/SDXC" : "SDSC");

	        // For SDSC cards, send CMD16 to set block size to 512 bytes
	        if (!sd_card_type_sdhc) {
	            response = SD_SendCommand(16, 512, 0x01);
	            SD_Deselect();

	            if (response != 0x00) {
	                printf("CMD16 failed: 0x%02X\n", response);
	                return STA_NOINIT;
	            }
	            printf("Block size set to 512 bytes\n");
	        }
	    } else {
	        printf("CMD58 failed: 0x%02X\n", response);
	        return STA_NOINIT;
	    }

	    // Clear the STA_NOINIT flag
	    Stat &= ~STA_NOINIT;
	    printf("SD Card initialized successfully via diskio\n");
	    return 0;
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
	 if (pdrv != 0 || !count) return RES_PARERR;
	    if (Stat & STA_NOINIT) return RES_NOTRDY;

	    printf("Reading sector %lu, count %u\n", sector, count);

	    for (UINT i = 0; i < count; i++) {
	        DWORD read_sector = sector + i;

	        // For SDSC cards, convert sector number to byte address
	        if (!sd_card_type_sdhc) {
	            read_sector *= 512;
	        }

	        // Send CMD17 (READ_SINGLE_BLOCK)
	        uint8_t response = SD_SendCommand(17, read_sector, 0x01);

	        if (response != 0x00) {
	            printf("CMD17 failed: 0x%02X for sector %lu\n", response, read_sector);
	            SD_Deselect();
	            return RES_ERROR;
	        }

	        // Wait for data token (0xFE)
	        uint32_t timeout = HAL_GetTick();
	        uint8_t token = 0xFF;

	        while ((HAL_GetTick() - timeout) < 2000) {
	            token = SD_TransmitReceive(0xFF);
	            if (token == SD_TOKEN_DATA) { // 0xFE - data token found
	                break;
	            }
	            if (token != 0xFF) { // Error token
	                printf("Data token error: 0x%02X\n", token);
	                SD_Deselect();
	                return RES_ERROR;
	            }
	        }

	        if (token != SD_TOKEN_DATA) {
	            printf("Data token timeout, last token: 0x%02X\n", token);
	            SD_Deselect();
	            return RES_ERROR;
	        }

	        // Read 512 bytes of data
	        if (HAL_SPI_Receive(&hspi1, buff + (i * 512), 512, 5000) != HAL_OK) {
	            printf("Data read failed\n");
	            SD_Deselect();
	            return RES_ERROR;
	        }

	        // Read and discard CRC (2 bytes)
	        SD_TransmitReceive(0xFF);
	        SD_TransmitReceive(0xFF);

	        SD_Deselect();

	        // Small delay between reads
	        if (i < count - 1) {
	            HAL_Delay(1);
	        }
	    }

	    printf("Read operation successful\n");
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
	if (pdrv != 0 || !count)
		return RES_PARERR;
	if (Stat & STA_NOINIT)
		return RES_NOTRDY;

	printf("Writing sector %lu, count %u\n", sector, count);

	for (UINT i = 0; i < count; i++) {
		DWORD write_sector = sector + i;

		// For SDSC cards, convert sector number to byte address
		if (!sd_card_type_sdhc) {
			write_sector *= 512;
		}

		// Send CMD24 (WRITE_SINGLE_BLOCK)
		uint8_t response = SD_SendCommand(24, write_sector, 0x01);

		if (response != 0x00) {
			printf("CMD24 failed: 0x%02X\n", response);
			SD_Deselect();
			return RES_ERROR;
		}

		// Send data token
		uint8_t token = SD_TOKEN_DATA;
		HAL_SPI_Transmit(&hspi1, &token, 1, 1000);

		// Send 512 bytes of data
		if (HAL_SPI_Transmit(&hspi1, (uint8_t*) (buff + (i * 512)), 512, 5000)
				!= HAL_OK) {
			printf("Data write failed\n");
			SD_Deselect();
			return RES_ERROR;
		}

		// Send dummy CRC (2 bytes)
		uint8_t dummy = SD_DUMMY_BYTE;
		HAL_SPI_Transmit(&hspi1, &dummy, 1, 1000);
		HAL_SPI_Transmit(&hspi1, &dummy, 1, 1000);

		// Check data response token
		HAL_SPI_TransmitReceive(&hspi1, &dummy, &response, 1, 1000);
		if ((response & 0x1F) != 0x05) {
			printf("Write data response error: 0x%02X\n", response);
			SD_Deselect();
			return RES_ERROR;
		}

		// Wait for card to finish writing (card will be busy)
		uint32_t timeout = HAL_GetTick();
		do {
			HAL_SPI_TransmitReceive(&hspi1, &dummy, &response, 1, 1000);
			if ((HAL_GetTick() - timeout) > 5000) {
				printf("Write busy timeout\n");
				SD_Deselect();
				return RES_ERROR;
			}
		} while (response != 0xFF);

		SD_Deselect();
	}

	printf("Write operation successful\n");
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
	if (Stat & STA_NOINIT)
		return RES_NOTRDY;

	switch (cmd) {
	case CTRL_SYNC:
		// Ensure all pending write operations are complete
		SD_Select();
		if (SD_WaitReady()) {
			SD_Deselect();
			return RES_OK;
		} else {
			SD_Deselect();
			return RES_ERROR;
		}

	case GET_SECTOR_COUNT: {
		// Try to read CSD register first
		uint8_t response = SD_SendCommand(9, 0, 0x01);
		if (response == 0x00) {
			// Wait for data token
			uint32_t timeout = HAL_GetTick();
			uint8_t dummy = SD_DUMMY_BYTE;

			while ((HAL_GetTick() - timeout) < 1000) {
				HAL_SPI_TransmitReceive(&hspi1, &dummy, &response, 1, 1000);
				if (response == SD_TOKEN_DATA)
					break;
			}

			if (response == SD_TOKEN_DATA) {
				// Read 16 bytes of CSD data
				uint8_t csd[16];
				HAL_SPI_Receive(&hspi1, csd, 16, 5000);

				// Read CRC (2 bytes) - discard
				HAL_SPI_TransmitReceive(&hspi1, &dummy, &response, 1, 1000);
				HAL_SPI_TransmitReceive(&hspi1, &dummy, &response, 1, 1000);

				SD_Deselect();

				// Calculate sector count based on CSD version
				DWORD sector_count;
				if ((csd[0] >> 6) == 1) { // CSD version 2.0 (SDHC/SDXC)
					uint32_t c_size = ((csd[7] & 0x3F) << 16) | (csd[8] << 8)
							| csd[9];
					sector_count = (c_size + 1) * 1024;
					printf("SDHC card, C_SIZE: %lu, sectors: %lu\n", c_size,
							sector_count);
				} else { // CSD version 1.0 (SDSC)
					uint32_t c_size = ((csd[6] & 0x03) << 10) | (csd[7] << 2)
							| ((csd[8] & 0xC0) >> 6);
					uint32_t c_size_mult = ((csd[9] & 0x03) << 1)
							| ((csd[10] & 0x80) >> 7);
					uint32_t read_bl_len = csd[5] & 0x0F;
					sector_count = (c_size + 1) * (1 << (c_size_mult + 2))
							* (1 << read_bl_len) / 512;
					printf("SDSC card, sectors: %lu\n", sector_count);
				}

				*(DWORD*) buff = sector_count;
				return RES_OK;
			}
		}

		// If CMD9 fails, use a reasonable default based on card type
		SD_Deselect();
		printf("CMD9 failed: 0x%02X, using default sector count\n", response);

		// Use reasonable defaults based on card type
		if (sd_card_type_sdhc) {
			*(DWORD*) buff = 7741440; // ~3.8GB in sectors (reasonable for SDHC)
		} else {
			*(DWORD*) buff = 1024000; // ~500MB in sectors (reasonable for SDSC)
		}
		printf("Using default sector count: %lu\n", *(DWORD*) buff);
		return RES_OK;
	}

	case GET_SECTOR_SIZE:
		*(WORD*) buff = 512;
		return RES_OK;

	case GET_BLOCK_SIZE:
		*(DWORD*) buff = 1;  // Single sector erase for SPI mode
		return RES_OK;

	default:
		return RES_PARERR;
	}
	/* USER CODE END IOCTL */
}
#endif /* _USE_IOCTL == 1 */
