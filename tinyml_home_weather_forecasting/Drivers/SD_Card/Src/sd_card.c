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

static FATFS s_fs;
extern char USERPath[4];
// --------------------------------

extern SPI_HandleTypeDef hspi1;
// From ff_gen_drv.c (for debug visibility)
extern Disk_drvTypeDef disk;
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

	printf("\r\n=== Testing FATFS functionality ===\r\n");

	fr = f_open(&fil, "0:/test.txt", FA_CREATE_ALWAYS | FA_WRITE);
	if (fr == FR_OK) {
		const char *test = "Hello from STM32 and SD card!\r\n";
		fr = f_write(&fil, test, (UINT) strlen(test), &bw);
		if (fr == FR_OK && bw == strlen(test)) {
			printf("Data written successfully (%u bytes)\r\n", bw);
		} else {
			printf("Write failed, fr=%d, bw=%u\r\n", fr, bw);
		}
		f_close(&fil);
	} else {
		printf("Failed to create file, fr=%d\r\n", fr);
		return;
	}

	fr = f_open(&fil, "0:/test.txt", FA_READ);

	if (fr == FR_OK) {
		fr = f_read(&fil, buffer, sizeof(buffer) - 1, &br);
		if (fr == FR_OK) {
			buffer[br] = '\0';
			printf("Data read successfully (%u bytes): %s", br, buffer);
		} else {
			printf("Read failed, fr=%d\r\n", fr);
		}
		f_close(&fil);
	} else {
		printf("Failed to open file for reading, fr=%d\r\n", fr);
	}
}

// map a file result to a descriptive result/error string
static const char* ff_errstr(FRESULT fr) {
    switch (fr) {
        case FR_OK: return "FR_OK";
        case FR_DISK_ERR: return "FR_DISK_ERR";
        case FR_INT_ERR: return "FR_INT_ERR";
        case FR_NOT_READY: return "FR_NOT_READY";
        case FR_NO_FILE: return "FR_NO_FILE";
        case FR_NO_PATH: return "FR_NO_PATH";
        case FR_INVALID_NAME: return "FR_INVALID_NAME";
        case FR_DENIED: return "FR_DENIED";
        case FR_EXIST: return "FR_EXIST";
        case FR_INVALID_OBJECT: return "FR_INVALID_OBJECT";
        case FR_WRITE_PROTECTED: return "FR_WRITE_PROTECTED";
        case FR_INVALID_DRIVE: return "FR_INVALID_DRIVE";
        case FR_NOT_ENABLED: return "FR_NOT_ENABLED";
        case FR_NO_FILESYSTEM: return "FR_NO_FILESYSTEM";
        case FR_MKFS_ABORTED: return "FR_MKFS_ABORTED";
        case FR_TIMEOUT: return "FR_TIMEOUT";
        case FR_LOCKED: return "FR_LOCKED";
        case FR_NOT_ENOUGH_CORE: return "FR_NOT_ENOUGH_CORE";
        case FR_TOO_MANY_OPEN_FILES: return "FR_TOO_MANY_OPEN_FILES";
        default: return "FR_???";
    }
}

void sd_card_ensure_temperature_csv_file(void) {
    const char *path = "0:/temperature.csv";
    const char *header = "timestamp,temperature_c\r\n";

    FRESULT file_result;
    FIL file;
    FILINFO file_info;

    printf("Ensuring temperature CSV exists...\r\n");

    //if it already exists, do nothing (don’t truncate!)
    file_result = f_stat(path, &file_info);
    if (file_result == FR_OK) {
        printf("Found %s (size=%lu). Leaving as-is.\r\n", path, (unsigned long)file_info.fsize);
        return;
    } else if (file_result != FR_NO_FILE) {
        printf("f_stat(%s) failed: %d (%s)\r\n", path, file_result, ff_errstr(file_result));
        error_handler_with_message("f_stat failed for temperature.csv");
        return;
    }

    // create a brand new file and write the csv header
    file_result = f_open(&file, path, FA_CREATE_NEW | FA_WRITE);
    if (file_result != FR_OK) {
        printf("f_open(FA_CREATE_NEW|FA_WRITE) failed: %d (%s)\r\n", file_result, ff_errstr(file_result));
        error_handler_with_message("Failed to create temperature.csv");
        return;
    }

    UINT bw = 0;
    file_result = f_write(&file, header, (UINT)strlen(header), &bw);
    if (file_result != FR_OK || bw != strlen(header)) {
        printf("f_write header failed: fr=%d (%s), bw=%u\r\n", file_result, ff_errstr(file_result), bw);
        f_close(&file);
        error_handler_with_message("Failed to write header to temperature.csv");
        return;
    }

    file_result = f_sync(&file); // ensure it’s physically written
    if (file_result != FR_OK) {
        printf("f_sync failed: %d (%s)\r\n", file_result, ff_errstr(file_result));
        // not fatal; proceed to close
    }

    f_close(&file);
    printf("Created %s with header.\r\n", path);
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
	printf("=== FatFs Debug Info ===\r\n");
	printf("_VOLUMES: %d\r\n", _VOLUMES);
	printf("disk.nbr: %d\r\n", disk.nbr);

	for (int i = 0; i < disk.nbr; i++) {
		printf("Drive %d: drv=%p, lun=%d, init=%d\r\n", i, disk.drv[i],
				disk.lun[i], disk.is_initialized[i]);
	}
	printf("========================\r\n");
}

// Optional: turn this on to auto-format blank cards
#ifndef SD_ENABLE_AUTO_MKFS
#define SD_ENABLE_AUTO_MKFS 1
#endif

FRESULT SD_Mount(void) {
	SD_DebugFatFsState();
	// Unmount just in case a previous run left things half-mounted
	f_mount(NULL, "0:", 0);

	const int MAX_TRIES = 5;
	FRESULT fr = FR_INT_ERR;

	for (int attempt = 1; attempt <= MAX_TRIES; ++attempt) {

		// Always start at slow speed for init and give the card dummy clocks
		sd_bus_slow();
		sd_dummy_clocks();
		sd_sleep(1000);

		// Force a fresh low-level init on drive 0
		DSTATUS st = disk_initialize(0);
		if (st & STA_NOINIT) {
			// Card not ready yet — small backoff and retry
			sd_sleep(500U * attempt);
			continue;
		}

		fr = f_mount(&s_fs, USERPath, 1);
		printf("f_mount('%s') -> %d (try %d/%d)\r\n", USERPath, fr, attempt,
				MAX_TRIES);

		if (fr == FR_OK) {
			// Now it’s safe to go faster
			sd_bus_fast();
			return FR_OK;
		}

#if _USE_MKFS   // make sure this is 1 in ffconf.h
		static BYTE work[4096]; // 4KB is safe; for plain FAT/FAT32 you can use 1024

#if defined(MKFS_PARM)
    // Newer FatFs (R0.14+)
    MKFS_PARM mp = { FM_ANY, 0, 0, 0, 0 };
    FRESULT mk = f_mkfs(USERPath, &mp, work, sizeof work);
  #else
		// Older FatFs (e.g., R0.12c)
		// f_mkfs(const TCHAR* path, BYTE sfd, UINT au, void* work, UINT len)
		BYTE sfd = 0; // 0 = create MBR partition (FDISK), 1 = super-floppy (no MBR)
		UINT au = 0;      // 0 = auto select allocation unit size
		FRESULT mk = f_mkfs(USERPath, sfd, au, work, sizeof work);
#endif

		printf("f_mkfs -> %d\r\n", mk);
		if (mk == FR_OK) {
			FRESULT fr2 = f_mount(&s_fs, USERPath, 1);
			printf("f_mount after mkfs -> %d\r\n", fr2);
			if (fr2 == FR_OK) {
				sd_bus_fast();
				return FR_OK;
			}
		}
#else
  printf("mkfs disabled: set _USE_MKFS = 1 in ffconf.h to enable auto-format.\r\n");
#endif

		// If the card says "not ready", back off a bit more each time
		if (fr == FR_NOT_READY) {
			sd_sleep(50U * attempt);
			continue;
		}

		// For other errors (e.g., FR_DISK_ERR), a couple more retries may still help
		sd_sleep(25U * attempt);
	}

	return fr; // caller prints a friendly message
}
