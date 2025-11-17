/*
 * sd_card.c
 *
 *  Created on: Aug 20, 2025
 *      Author: rishi_latchmepersad
 */

#include "sd_card.h"
#include "fatfs.h"
#include "ff.h"
#include "ff_gen_drv.h"
#include "diskio.h"
#include "cmsis_os2.h"
#include <stdio.h>
#include <string.h>
#include "main.h"
#include <stdbool.h>
#include <app_error.h>
#include "sd_spi_low_level.h"

#define LOG_PREFIX "[SD_CARD] "

static FATFS s_fs;
static bool  s_is_mounted = false;
static osMutexId_t s_mount_mutex = NULL;
extern char USERPath[4];
// --------------------------------

extern SPI_HandleTypeDef hspi1;
// From ff_gen_drv.c (for debug visibility)
extern Disk_drvTypeDef disk;

static inline void sd_mount_lock(void) {
        if (osKernelGetState() != osKernelRunning) {
                return;
        }
        if (s_mount_mutex == NULL) {
                s_mount_mutex = osMutexNew(NULL);
        }
        if (s_mount_mutex != NULL) {
                (void) osMutexAcquire(s_mount_mutex, osWaitForever);
        }
}

static inline void sd_mount_unlock(void) {
        if ((osKernelGetState() != osKernelRunning) || (s_mount_mutex == NULL)) {
                return;
        }
        (void) osMutexRelease(s_mount_mutex);
}
// Sleep helper: works before and after the kernel starts.
static inline void sd_sleep(uint32_t ms) {
	if (osKernelGetState() == osKernelRunning)
		osDelay(ms);
	else
		HAL_Delay(ms);
}

// Ensure SPI is at the slowest prescaler for card init.
static void sd_bus_slow(void) {
	if (hspi1.Init.BaudRatePrescaler != SPI_BAUDRATEPRESCALER_256) {
		hspi1.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_256;
		HAL_SPI_Init(&hspi1);
	}
}

// Speed up after mount / once the card has left idle state.
static void sd_bus_fast(void) {
	if (hspi1.Init.BaudRatePrescaler != SPI_BAUDRATEPRESCALER_64) {
		hspi1.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_64;
		HAL_SPI_Init(&hspi1);
	}
}

// 80 dummy clocks with CS high to wake the card & align
static void sd_dummy_clocks(void) {
	uint8_t ff = 0xFF;
	HAL_GPIO_WritePin(SD_Card_CS_GPIO_Port, SD_Card_CS_Pin, GPIO_PIN_SET);
	for (int i = 0; i < 100; ++i) {
		(void) HAL_SPI_Transmit(&hspi1, &ff, 1, 10);
	}
}

//Run a basic read/write test to ensure that the MicroSD card works
void SD_TestFatFs(void)

{
	FRESULT fr;
	FIL fil;
	UINT bw, br;
	char buffer[100];

	printf(LOG_PREFIX "\r\n=== Testing FATFS functionality ===\r\n");
	fr = f_unlink("0:/test.txt"); // Attempt to delete the file

	if (fr == FR_OK) {
	    printf(LOG_PREFIX "test.txt deleted successfully\r\n");
	} else {
	   printf(LOG_PREFIX "Unable to delete text.txt\r\n");
	}
	fr = f_open(&fil, "0:/test.txt", FA_CREATE_ALWAYS | FA_WRITE);
	if (fr == FR_OK) {
		const char *test = "Hello from STM32 and SD card!\r\n";
		fr = f_write(&fil, test, (UINT) strlen(test), &bw);
		if (fr == FR_OK && bw == strlen(test)) {
			printf(LOG_PREFIX "Data written successfully (%u bytes)\r\n", bw);
		} else {
			printf(LOG_PREFIX "Write failed, fr=%d, bw=%u\r\n", fr, bw);
		}
		f_close(&fil);
	} else {
		printf(LOG_PREFIX "Failed to create file, fr=%d\r\n", fr);
		return;
	}

	fr = f_open(&fil, "0:/test.txt", FA_READ);

	if (fr == FR_OK) {
		fr = f_read(&fil, buffer, sizeof(buffer) - 1, &br);
		if (fr == FR_OK) {
			buffer[br] = '\0';
			printf(LOG_PREFIX "Data read successfully (%u bytes): %s", br, buffer);
		} else {
			printf(LOG_PREFIX "Read failed, fr=%d\r\n", fr);
		}
		f_close(&fil);
	} else {
		printf(LOG_PREFIX "Failed to open file for reading, fr=%d\r\n", fr);
	}
}

DWORD SD_GetFreeKB(void) {
	FATFS *pfs;
	DWORD free_clusters;
	if (f_getfree("0:", &free_clusters, &pfs) == FR_OK) {
		DWORD free_sectors = free_clusters * pfs->csize;
		return free_sectors / 2; // 512-byte sectors -> KB
	}
	return 0;
}

void SD_DebugFatFsState(void) {
	printf(LOG_PREFIX "=== FatFs Debug Info ===\r\n");
	printf(LOG_PREFIX "_VOLUMES: %d\r\n", _VOLUMES);
	printf(LOG_PREFIX "disk.nbr: %d\r\n", disk.nbr);

	for (int i = 0; i < disk.nbr; i++) {
		printf(LOG_PREFIX "Drive %d: drv=%p, lun=%d, init=%d\r\n", i, disk.drv[i],
				disk.lun[i], disk.is_initialized[i]);
	}
	printf(LOG_PREFIX "========================\r\n");
}

// Optional: turn this on to auto-format blank cards
#ifndef SD_ENABLE_AUTO_MKFS
#define SD_ENABLE_AUTO_MKFS 0
#endif

void SD_InvalidateMount(void) {
        s_is_mounted = false;
}

FRESULT SD_Mount(void) {
        sd_mount_lock();

        if (s_is_mounted) {
                DSTATUS st = disk_status(0);
                if ((st & STA_NOINIT) == 0U) {
                        /* Already mounted and the card still reports ready. */
                        sd_mount_unlock();
                        return FR_OK;
                }
                s_is_mounted = false;
        }

        /* Only dump the (expensive and noisy) FatFs debug snapshot when we
         * are about to perform a fresh mount attempt.  Several tasks probe
         * SD_Mount() to make sure the volume is ready, so calling the helper
         * while the media is already mounted ended up spamming the console
         * every few seconds without providing new information. */
        SD_DebugFatFsState();

        // Unmount just in case a previous run left things half-mounted
        f_mount(NULL, "0:", 0);

	const int MAX_TRIES = 5;
	FRESULT fr = FR_INT_ERR;

	for (int attempt = 1; attempt <= MAX_TRIES; ++attempt) {

		// Always start at slow speed for init and give the card dummy clocks
		sd_bus_slow();
		sd_spi_power_on_sequence();   /* clocks with CS high, then quick probe */
		sd_sleep(2000);

		// Force a fresh low-level init on drive 0
		DSTATUS st = disk_initialize(0);
		if (st & STA_NOINIT) {
			// Card not ready yet — small backoff and retry
			sd_sleep(1000U * attempt);
			continue;
		}

		fr = f_mount(&s_fs, USERPath, 1);
		printf(LOG_PREFIX "f_mount('%s') -> %d (try %d/%d)\r\n", USERPath, fr, attempt,
				MAX_TRIES);

                        if (fr == FR_OK) {
                                // Now it’s safe to go faster
                                sd_bus_fast();
                                s_is_mounted = true;
                                sd_mount_unlock();
                                return FR_OK;
                        }

#if SD_ENABLE_AUTO_MKFS && _USE_MKFS
if (fr == FR_NO_FILESYSTEM) {
    static BYTE work[4096];
  #if defined(MKFS_PARM)
    MKFS_PARM mp = (MKFS_PARM){ FM_ANY, 0, 0, 0, 0 };
    FRESULT mk = f_mkfs(USERPath, &mp, work, sizeof work);
  #else
    BYTE sfd = 0; UINT au = 0;
    FRESULT mk = f_mkfs(USERPath, sfd, au, work, sizeof work);
  #endif
    printf(LOG_PREFIX "f_mkfs -> %d\r\n", mk);
    if (mk == FR_OK) {
        FRESULT fr2 = f_mount(&s_fs, USERPath, 1);
        printf(LOG_PREFIX "f_mount after mkfs -> %d\r\n", fr2);
        if (fr2 == FR_OK) { sd_bus_fast(); return FR_OK; }
    }
}
#else
printf(LOG_PREFIX "mkfs disabled; enable SD_ENABLE_AUTO_MKFS && _USE_MKFS to format blank cards.\r\n");
#endif

		// If the card says "not ready", back off a bit more each time
		if (fr == FR_NOT_READY) {
			sd_sleep(50U * attempt);
			continue;
		}

		// For other errors (e.g., FR_DISK_ERR), a couple more retries may still help
		sd_sleep(25U * attempt);
	}

        s_is_mounted = false;
        sd_mount_unlock();
        return fr; // caller prints a friendly message
}
