/*
 * sd_card.c
 *
 *  Created on: Aug 20, 2025
 *      Author: rishi_latchmepersad
 */
#include "sd_card.h"
#include "fatfs.h"
#include "ff_gen_drv.h"
#include <stdio.h>
#include <string.h>

// Local filesystem object lives here, not in main.c
static FATFS s_fs;

// Provided by fatfs.c after MX_FATFS_Init()
extern char USERPath[4];

// From ff_gen_drv.c (for debug visibility)
extern Disk_drvTypeDef disk;

FRESULT SD_Mount(void)
{
    FRESULT fr = f_mount(&s_fs, USERPath, 1);
    printf("f_mount('%s') -> %d\r\n", USERPath, fr);
    return fr;
}

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
        fr = f_write(&fil, test, (UINT)strlen(test), &bw);
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

    FATFS *pfs;
    DWORD free_clusters, free_sectors;
    fr = f_getfree("0:", &free_clusters, &pfs);
    if (fr == FR_OK) {
        free_sectors = free_clusters * pfs->csize;
        printf("Free space: %lu sectors (%lu KB)\r\n",
               free_sectors, free_sectors / 2);
    } else {
        printf("Failed to get free space, fr=%d\r\n", fr);
    }
}

DWORD SD_GetFreeKB(void)
{
    FATFS *pfs;
    DWORD free_clusters;
    if (f_getfree("0:", &free_clusters, &pfs) == FR_OK) {
        DWORD free_sectors = free_clusters * pfs->csize;
        return free_sectors / 2; // 512-byte sectors -> KB
    }
    return 0;
}

void SD_DebugFatFsState(void)
{
    printf("=== FatFs Debug Info ===\r\n");
    printf("_VOLUMES: %d\r\n", _VOLUMES);
    printf("disk.nbr: %d\r\n", disk.nbr);

    for (int i = 0; i < disk.nbr; i++) {
        printf("Drive %d: drv=%p, lun=%d, init=%d\r\n",
               i, disk.drv[i], disk.lun[i], disk.is_initialized[i]);
    }
    printf("========================\r\n");
}
