#pragma once
#include "ff.h"
#include <stdbool.h>

// Mount SD card via FatFs. Returns FR_OK on success.
FRESULT SD_Mount(void);

// Mark the cached mount state as invalid so the next SD_Mount call performs
// a full re-initialization (used after I/O errors).
void SD_InvalidateMount(void);

// Create -> write -> read-back self-test (prints to UART).
void SD_TestFatFs(void);

// Print internal FatFs driver table (useful when debugging).
void SD_DebugFatFsState(void);

// Return free space in KB (0 on error).
DWORD SD_GetFreeKB(void);
