#pragma once
#include "ff.h"
#include <stdbool.h>

// Mount SD card via FatFs. Returns FR_OK on success.
FRESULT SD_Mount(void);

// Create -> write -> read-back self-test (prints to UART).
void SD_TestFatFs(void);

// Print internal FatFs driver table (useful when debugging).
void SD_DebugFatFsState(void);

// Return free space in KB (0 on error).
DWORD SD_GetFreeKB(void);
