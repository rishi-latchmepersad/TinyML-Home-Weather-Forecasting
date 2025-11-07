#include "measurement_logger_task.h"

#include <stdio.h>
#include <string.h>

#include "ds3231.h"            /* ds3231_read_time_iso8601_utc_i2c1() */
#include "diskio.h"
#include "led_service.h"
#include "ff.h"
#include "sd_card.h"

/* === Configuration macros (override in project settings if desired) ================= */
#ifndef LOGGER_TASK_STACK_WORDS
#define LOGGER_TASK_STACK_WORDS   (600u)
#endif
#ifndef LOGGER_TASK_PRIORITY
#define LOGGER_TASK_PRIORITY      (osPriorityNormal)
#endif
#ifndef LOGGER_QUEUE_LENGTH
#define LOGGER_QUEUE_LENGTH       (32u)
#endif
#ifndef LOGGER_BUFFER_BYTES
#define LOGGER_BUFFER_BYTES       (512u)       /* batch to 1 sector */
#endif
#ifndef LOGGER_BASE_DIR
#define LOGGER_BASE_DIR           "0:/logs"    /* single logs directory */
#endif
#ifndef LOGGER_FLUSH_PERIOD_MS
#define LOGGER_FLUSH_PERIOD_MS    (30000u)      /* periodic flush cadence */
#endif
#ifndef LOGGER_BACKOFF_MS
#define LOGGER_BACKOFF_MS         (500u)       /* retry delay when errors occur */
#endif
#ifndef LOGGER_SYNC_PERIOD_MS
#define LOGGER_SYNC_PERIOD_MS   (5000u)   /* sync at least every 5 s */
#endif
extern osMutexId_t g_fs_mutex;
#ifndef FS_LOCK
#define FS_LOCK()   osMutexAcquire(g_fs_mutex, osWaitForever)
#define FS_UNLOCK() osMutexRelease(g_fs_mutex)
#endif
/* ==================================================================================== */

/* Internal state */
static QueueHandle_t g_measurement_queue = NULL;
static TaskHandle_t g_logger_task_handle = NULL;

static FIL g_active_file;
static bool g_file_open = false;
static char g_last_date_yyyy_mm_dd[11] = { 0 }; /* "YYYY-MM-DD" + NUL */
static char g_active_path[128] = { 0 };
#if defined(__GNUC__)
__attribute__((aligned(32)))
#endif
/* Place in DMA-visible SRAM (not DTCM) and align to cache line */
__attribute__((section(".sram1"), aligned(32)))
static char g_write_buffer[LOGGER_BUFFER_BYTES];
static size_t g_write_used = 0u;
static volatile bool g_sync_requested = false;
static TickType_t last_sync_tick = 0;
/* Local helpers */
static void measurement_logger_task_entry(void *argument);
static bool measurement_logger_ensure_directory_exists(const char *dir_path);
static bool measurement_logger_open_today_file(const char *date_yyyy_mm_dd);
static bool measurement_logger_flush_buffer_to_file(void);
static size_t measurement_logger_format_csv_line(
		const measurement_logger_message_t *msg, char *out_line,
		size_t out_capacity);
static void measurement_logger_checkpoint_close(void);


/****************************************************************************************
 * Function:    print_dir_listing_r012c
 * Purpose:     List a directory (R0.12c). Uses FILINFO.fname (LFN/primary) and
 *              falls back to FILINFO.altname (8.3) when needed.
 ****************************************************************************************/
static void print_dir_listing(const TCHAR *path)
{
    FRESULT fr;
    DIR     dir;
    FILINFO fno;

    memset(&fno, 0, sizeof(fno));
    printf("DIR %s\r\n", path);

    fr = f_opendir(&dir, path);
    if (fr != FR_OK) {
        printf("  <opendir failed fr=%d>\r\n", (int)fr);
        return;
    }

    for (;;) {
        fr = f_readdir(&dir, &fno);
        if (fr != FR_OK) {
            printf("  <readdir error fr=%d>\r\n", (int)fr);
            break;
        }
        if (fno.fname[0] == '\0') {
            /* End of directory */
            break;
        }

        /* Prefer primary name (LFN if enabled), else 8.3 alias */
        const TCHAR *name_ptr = (fno.fname[0] != '\0') ? fno.fname : fno.altname;

        printf("  %c %10lu %s\r\n",
               (fno.fattrib & AM_DIR) ? 'D' : 'F',
               (unsigned long)fno.fsize,
               name_ptr);
    }

    (void)f_closedir(&dir);
}

/****************************************************************************************
 * Function:    measurement_logger_debug_snapshot
 * Purpose:     Print volume info and list "0:/" and "0:/logs" using the R0.12c fields.
 ****************************************************************************************/
static void measurement_logger_debug_snapshot(void)
{
    FRESULT fr;
    FATFS  *pfs = NULL;
    DWORD   nfree = 0, totsec = 0, freesec = 0;
    TCHAR   label[32];
    DWORD   vsn = 0;

#ifdef FS_LOCK
    FS_LOCK();
#endif

    fr = f_getlabel("0:/", label, &vsn);
    printf("VOL: fr=%d label=\"%s\" vsn=%08lX\r\n",
           (int)fr, (label[0] ? label : ""), (unsigned long)vsn);

    fr = f_getfree("0:/", &nfree, &pfs);
    if (fr == FR_OK && pfs) {
        totsec  = (DWORD)(pfs->n_fatent - 2U) * (DWORD)pfs->csize;
        freesec = (DWORD)nfree * (DWORD)pfs->csize;
        printf("FS: total=%lu KB free=%lu KB\r\n",
               (unsigned long)(totsec/2U), (unsigned long)(freesec/2U));
    } else {
        printf("FS: f_getfree failed fr=%d\r\n", (int)fr);
    }

    print_dir_listing("0:/");
    print_dir_listing("0:/logs");

#ifdef FS_UNLOCK
    FS_UNLOCK();
#endif
}

/* Optional: compile-time probe so you *know* what this CU sees */
static void fatfs_build_probe(void) {
#if defined(_FFCONF)
    printf("_FFCONF=%ld ", (long)_FFCONF);
#endif
#if defined(_USE_LFN)
    printf("_USE_LFN=%d ", (int)_USE_LFN);
#endif
#if defined(_MAX_LFN)
    printf("_MAX_LFN=%d ", (int)_MAX_LFN);
#endif
    printf("\r\n");
}

/****************************************************************************************
 * Function:    measurement_logger_debug_file_size
 * Purpose:     Print current size of the active CSV to confirm growth after a flush.
 * Params:      void
 * Returns:     void
 * Notes:       Uses g_active_path; call only when g_file_open == true.
 ****************************************************************************************/
static void measurement_logger_debug_file_size(void) {
	if (!g_file_open) {
		printf("DEBUG: file not open\r\n");
		return;
	}
	FSIZE_t sz = f_size(&g_active_file);
	printf("DEBUG: size(%s) = %lu bytes\r\n", g_active_path,
			(unsigned long) sz);
}

/****************************************************************************************
 * Function:    measurement_logger_force_commit_and_reopen
 * Purpose:     Force directory entry & data to media, unmount/remount to simulate
 *              safe removal, verify file visibility via f_stat, then reopen for append.
 * Notes:       Call this ONCE after your first successful flush to prove durability.
 ****************************************************************************************/
static void measurement_logger_force_commit_and_reopen(void) {
    if (!g_file_open || g_active_path[0] == '\0') return;

    /* Finish and close the current handle */
    FS_LOCK();
    (void)f_sync(&g_active_file);
    (void)f_close(&g_active_file);
    g_file_open = false;
    FS_UNLOCK();

    /* Small settle */
    FS_LOCK();
    (void)disk_ioctl(0, CTRL_SYNC, NULL);
    FS_UNLOCK();
    vTaskDelay(pdMS_TO_TICKS(5));

    /* Reopen append WITHOUT an f_stat probe */
    FS_LOCK();
    FRESULT fr = f_open(&g_active_file, g_active_path, FA_OPEN_ALWAYS | FA_WRITE);
    if (fr == FR_OK) {
        (void)f_lseek(&g_active_file, f_size(&g_active_file));
        g_file_open = true;
        printf("REOPENED: %s size=%lu\r\n",
               g_active_path, (unsigned long)f_size(&g_active_file));
    } else {
        printf("REOPEN FAILED fr=%d on %s\r\n", (int)fr, g_active_path);
    }
    FS_UNLOCK();
}
/* --- Public API -------------------------------------------------------------------- */

bool measurement_logger_task_create(void) {
	if (g_measurement_queue == NULL) {
		g_measurement_queue = xQueueCreate((UBaseType_t)LOGGER_QUEUE_LENGTH,
				(UBaseType_t )sizeof(measurement_logger_message_t));
		if (g_measurement_queue == NULL) {
			return false;
		}
	}

	if (g_logger_task_handle == NULL) {
		const osThreadAttr_t attr = { .name = "sd_card_logger_task", .priority =
				(osPriority_t) LOGGER_TASK_PRIORITY, .stack_size =
				(uint32_t) (LOGGER_TASK_STACK_WORDS * sizeof(StackType_t)), };
		g_logger_task_handle = osThreadNew(measurement_logger_task_entry, NULL,
				&attr);
		if (g_logger_task_handle == NULL) {
			return false;
		}
	}
	return true;
}

bool measurement_logger_enqueue(const measurement_logger_message_t *msg,
		uint32_t timeout_ms) {
	if ((g_measurement_queue == NULL) || (msg == NULL)) {
		return false;
	}
	BaseType_t ok = xQueueSend(g_measurement_queue, msg,
			pdMS_TO_TICKS(timeout_ms));
	return (ok == pdPASS);
}

void measurement_logger_request_sync(void) {
	g_sync_requested = true;
}

/* --- Internal: state machine -------------------------------------------------------- */

typedef enum {
	LOGGER_STATE_STARTUP = 0,
	LOGGER_STATE_ENSURE_DIR,
	LOGGER_STATE_OPEN_TODAY,
	LOGGER_STATE_RUNNING,
	LOGGER_STATE_FLUSH,
	LOGGER_STATE_ROTATE,
	LOGGER_STATE_ERROR_BACKOFF
} logger_state_t;

/****************************************************************************************
 * Function:    measurement_logger_task_entry
 * Purpose:     Own the SD card and persist queued observations to CSV in batches.
 *              Implemented as a small state machine to make behavior explicit.
 *
 * Parameters:  void *argument - unused.
 *
 * Side Effects:
 *                - Opens/creates files on the SD card; performs periodic writes and flushes.
 *
 * Concurrency:
 *                - This task should be the only writer to FatFs/SD in the system.
 ****************************************************************************************/
static void measurement_logger_task_entry(void *argument) {
	(void) argument;
	logger_state_t state = LOGGER_STATE_STARTUP;
	TickType_t last_flush_tick = xTaskGetTickCount();

	while (1) {
		switch (state) {
		case LOGGER_STATE_STARTUP:
			/* Print DMA-visible buffer address once */
			static bool s_buf_addr_printed = false;
			if (!s_buf_addr_printed) {
				printf("g_write_buffer @ %p\r\n", (void*) g_write_buffer);
				s_buf_addr_printed = true;
			}
			if (SD_Mount() == FR_OK) {
				printf("SD card mounted successfully!\r\n");
				SD_TestFatFs();
				state = LOGGER_STATE_ENSURE_DIR;
				break;
			} else {
				printf("Unable to mount SD card!\r\n");
				state = LOGGER_STATE_ERROR_BACKOFF;
				break;
			}

		case LOGGER_STATE_ENSURE_DIR:
			if (measurement_logger_ensure_directory_exists(LOGGER_BASE_DIR)) {
				state = LOGGER_STATE_OPEN_TODAY;
			} else {
				state = LOGGER_STATE_ERROR_BACKOFF;
			}
			break;

		case LOGGER_STATE_OPEN_TODAY: {
			char ts[24];
			if (ds3231_read_time_iso8601_utc_i2c1(ts, sizeof ts) != HAL_OK) {
				printf("We weren't able to get the current time from the DS3231\n");
				(void) snprintf(ts, sizeof ts, "2000-01-01T00:00:00Z");
			}
			else{
				printf("The DS3231 seems to be responding. \n");
			}
			char date[11];
			memcpy(date, ts, 10);
			date[10] = '\0';

			if (measurement_logger_open_today_file(date)) {
				last_flush_tick = xTaskGetTickCount();
				printf("Current time: %s \r\n", ts);
				printf("Current open file: %s \r\n", g_active_path);
				measurement_logger_request_sync();
				state = LOGGER_STATE_RUNNING;
			} else {
				state = LOGGER_STATE_ERROR_BACKOFF;
			}
			break;
		}

		case LOGGER_STATE_RUNNING: {
			/* Check for date change to rotate */
			char ts_now[24];
			if (ds3231_read_time_iso8601_utc_i2c1(ts_now, sizeof ts_now)
					!= HAL_OK) {
				printf("We weren't able to get the current time from the DS3231\n");
				(void) snprintf(ts_now, sizeof ts_now, "2000-01-01T00:00:00Z");
			}
			char date_now[11];
			memcpy(date_now, ts_now, 10);
			date_now[10] = '\0';
			if (strncmp(date_now, g_last_date_yyyy_mm_dd, 10) != 0) {
				/* Day rolled */
				printf("Date roll detected: %s -> %s (rotating)\r\n",
						g_last_date_yyyy_mm_dd, date_now);
				state = LOGGER_STATE_ROTATE;
				break;
			}

			/* Service the queue */
			measurement_logger_message_t msg;
			if (xQueueReceive(g_measurement_queue, &msg,
					pdMS_TO_TICKS(200u)) == pdPASS) {
				char line[160];
				size_t n = measurement_logger_format_csv_line(&msg, line,
						sizeof line);
				if (n > 0u) {
					const size_t len = n;
					if ((g_write_used + len) > sizeof g_write_buffer) {
						state = LOGGER_STATE_FLUSH;
						break;
					}
					memcpy(&g_write_buffer[g_write_used], line, len);
					g_write_used += len;
				}
			}

			/* Periodic flush or on request */
			TickType_t now = xTaskGetTickCount();
			if (g_sync_requested
					|| ((now - last_flush_tick)
							>= pdMS_TO_TICKS(LOGGER_FLUSH_PERIOD_MS))) {
				state = LOGGER_STATE_FLUSH;
			}
			break;
		}

		case LOGGER_STATE_FLUSH:
			//get the current timestamp
			char ts_now[24];
			if (ds3231_read_time_iso8601_utc_i2c1(ts_now, sizeof ts_now)
					!= HAL_OK) {
				printf("We weren't able to get the current time from the DS3231\n");
				(void) snprintf(ts_now, sizeof ts_now, "2000-01-01T00:00:00Z");
			}
			const size_t before = g_write_used;
			bool flush_result = measurement_logger_flush_buffer_to_file();
			if (!flush_result) {
				state = LOGGER_STATE_ERROR_BACKOFF;
				printf("Buffered data failed to flush to file.\r\n");
			} else {
				g_sync_requested = false;
				last_flush_tick = xTaskGetTickCount();
				if (flush_result && before > 0u) {
					printf("Flushed %lu bytes to file %s on %s.\r\n",
							(unsigned long) before, g_active_path, ts_now);
					led_service_activity_bump(1000);
					static bool s_force_once_done = false;
					if (!s_force_once_done && g_file_open) {
						s_force_once_done = true;
						measurement_logger_force_commit_and_reopen();
						measurement_logger_checkpoint_close();
					}
				} else {
					//we don't have any data to flush
				}
				state = LOGGER_STATE_RUNNING;
			}
			break;

		case LOGGER_STATE_ROTATE:
			if (g_file_open) {
				(void) measurement_logger_flush_buffer_to_file();
				(void) f_close(&g_active_file);
				g_file_open = false;
			}
			state = LOGGER_STATE_OPEN_TODAY;
			break;

		case LOGGER_STATE_ERROR_BACKOFF:
			/* Close file if needed */
			if (g_file_open) {
				(void) f_close(&g_active_file);
				g_file_open = false;
			}
			// show the red led
			led_command_t err = { .led_identifier = led_identifier_ld3,
					.pattern_identifier = led_pattern_identifier_error_code,
					.error_code_count = 2, /* two-blink code = storage */
					.duration_ms = 0, /* persist until cleared */
					.priority_level = 10 };
			(void) led_service_set_pattern(&err);
			vTaskDelay(pdMS_TO_TICKS(LOGGER_BACKOFF_MS));
			state = LOGGER_STATE_STARTUP;
			break;
		}
	}
}

/* --- Helper implementations --------------------------------------------------------- */

/****************************************************************************************
 * Function:    measurement_logger_format_csv_line
 * Purpose:     Make one CSV line: timestamp_iso8601,sensor,quantity,value,unit\r\n
 ****************************************************************************************/
static size_t measurement_logger_format_csv_line(
		const measurement_logger_message_t *msg, char *out_line,
		size_t out_capacity) {
	char ts[24];
	if (ds3231_read_time_iso8601_utc_i2c1(ts, sizeof ts) != HAL_OK) {
		(void) snprintf(ts, sizeof ts, "TIME?");
	}

	int n = snprintf(out_line, out_capacity, "%s,%s,%s,%.6f,%s\r\n", ts,
			(msg->sensor_name != NULL) ? msg->sensor_name : "",
			(msg->quantity_name != NULL) ? msg->quantity_name : "",
			(double) msg->value_numeric,
			(msg->unit_name != NULL) ? msg->unit_name : "");
	if ((n <= 0) || ((size_t) n >= out_capacity)) {
		return 0u;
	}
	return (size_t) n;
}

/****************************************************************************************
 * Function:    measurement_logger_flush_buffer_to_file
 * Purpose:     Write buffered data to the file and optionally f_sync.
 * Returns:     bool - true on success.
 ****************************************************************************************/
static bool measurement_logger_flush_buffer_to_file(void) {
    /* Nothing to flush? */
    if (g_write_used == 0u) return true;

    /* Ensure an open file; create if missing. */
    if (!g_file_open) {
        FS_LOCK();
        FILINFO fi; memset(&fi, 0, sizeof fi);
        FRESULT fr = f_stat(g_active_path, &fi);
        if (fr == FR_OK) {
            fr = f_open(&g_active_file, g_active_path, FA_OPEN_EXISTING | FA_WRITE);
            if (fr == FR_OK) (void)f_lseek(&g_active_file, f_size(&g_active_file));
        } else if (fr == FR_NO_FILE) {
            /* Create (no truncate), then write header if this is a brand-new file */
            fr = f_open(&g_active_file, g_active_path, FA_OPEN_ALWAYS | FA_WRITE);
            if (fr == FR_OK) {
                if (f_size(&g_active_file) == 0) {
                    const char *hdr = "timestamp_iso8601,sensor,quantity,value,unit\r\n";
                    UINT bw = 0;
                    if (f_write(&g_active_file, hdr, (UINT)strlen(hdr), &bw) == FR_OK &&
                        bw == (UINT)strlen(hdr)) {
                        (void)f_sync(&g_active_file);
                    }
                }
                (void)f_lseek(&g_active_file, f_size(&g_active_file));
            }
        }
        if (fr == FR_OK) g_file_open = true;
        FS_UNLOCK();

        if (!g_file_open) {
            /* Keep buffer intact; let caller retry. */
            return false;
        }
    }

    /* Append buffered data and sync. */
    UINT bw = 0u;
    FS_LOCK();
    FRESULT wr = f_write(&g_active_file, g_write_buffer, (UINT)g_write_used, &bw);
    if (wr == FR_OK && bw == g_write_used) wr = f_sync(&g_active_file);
    FS_UNLOCK();

    if (wr != FR_OK || bw != g_write_used) {
        /* Keep buffer for retry */
        return false;
    }

    g_write_used = 0u;
    return true;
}


/****************************************************************************************
 * Function:    measurement_logger_ensure_directory_exists
 * Purpose:     Create base logs directory if it doesn't exist.
 ****************************************************************************************/
static bool measurement_logger_ensure_directory_exists(const char *dir_path) {
	FILINFO fi;
	FRESULT fr = f_stat(dir_path, &fi);
	if (fr == FR_OK) {
		return (fi.fattrib & AM_DIR) != 0;
	}
	if (fr == FR_NO_FILE) {
		fr = f_mkdir(dir_path);
		return (fr == FR_OK);
	}
	return false;
}

/****************************************************************************************
 * Function:    measurement_logger_open_today_file
 * Purpose:     Ensure today's CSV exists at LOGGER_BASE_DIR and open it for append.
 *              If not present, create with FA_CREATE_NEW, write header, f_sync + close
 *              to force the directory entry to media, then reopen and seek to EOF.
 *
 * Parameters:
 *   date_yyyy_mm_dd  [in] Pointer to 10-char "YYYY-MM-DD" string.
 *
 * Returns:     bool - true on success (file open at EOF), false on any error.
 *
 * Side Effects:
 *   - Creates LOGGER_BASE_DIR if missing.
 *   - Updates g_file_open, g_active_path, g_last_date_yyyy_mm_dd.
 *
 * Preconditions:
 *   - Volume "0:" mounted.
 *   - FS_LOCK/FS_UNLOCK are safe from this context (or no-ops).
 *
 * Notes:
 *   - FatFs R0.12c: FILINFO has .fname (primary/LFN if enabled) and .altname (8.3).
 *   - We log every step with FRESULT codes so failures are unambiguous.
 ****************************************************************************************/
static bool measurement_logger_open_today_file(const char *date_yyyy_mm_dd)
{
    FRESULT fr;
    FILINFO fi_dir; memset(&fi_dir, 0, sizeof(fi_dir));
    FILINFO fi;     memset(&fi, 0, sizeof(fi));
    char path[128];

    /* Build "0:/logs/measurements_YYYY-MM-DD.csv" */
    (void)snprintf(path, sizeof path, LOGGER_BASE_DIR "/measurements_%s.csv",
                   date_yyyy_mm_dd);

    /* Hex dump the path once to catch invisible characters */
    {
        size_t L = strlen(path);
        printf("PATH len=%u: ", (unsigned)L);
        for (size_t i = 0; i < L; ++i) printf("%02X ", (unsigned char)path[i]);
        printf("\r\n");
    }

    /* 1) Ensure the base directory exists */
    FS_LOCK();
    fr = f_stat(LOGGER_BASE_DIR, &fi_dir);
    FS_UNLOCK();
    printf("DIRCHECK f_stat(%s) -> fr=%d attr=0x%02X\r\n",
           LOGGER_BASE_DIR, (int)fr, (unsigned)fi_dir.fattrib);

    if (fr == FR_NO_FILE || ((fi_dir.fattrib & AM_DIR) == 0)) {
        FS_LOCK();
        FRESULT mr = f_mkdir(LOGGER_BASE_DIR);
        FS_UNLOCK();
        printf("mkdir(%s) -> fr=%d\r\n", LOGGER_BASE_DIR, (int)mr);
        if (mr != FR_OK && mr != FR_EXIST) {
            return false; /* parent path truly missing/invalid */
        }
    } else if (fr != FR_OK) {
        /* e.g., FR_INVALID_NAME, FR_DISK_ERR, etc. */
        return false;
    }

    /* 2) If file already exists, open+seek EOF; else create it */
    FS_LOCK();
    fr = f_stat(path, &fi);
    FS_UNLOCK();
    printf("PRESCAN f_stat(%s) -> fr=%d size=%lu\r\n",
           path, (int)fr, (unsigned long)fi.fsize);

    if (fr == FR_OK) {
        /* Exists -> open existing, seek to EOF */
        FS_LOCK();
        fr = f_open(&g_active_file, path, FA_OPEN_EXISTING | FA_WRITE);
        if (fr == FR_OK) fr = f_lseek(&g_active_file, f_size(&g_active_file));
        FS_UNLOCK();
        if (fr != FR_OK) {
            printf("open/seek existing failed for %s (fr=%d)\r\n", path, (int)fr);
            FS_LOCK(); (void)f_close(&g_active_file); FS_UNLOCK();
            return false;
        }
    } else if (fr == FR_NO_FILE) {
        /* New -> create, write header, sync+close to cement dir entry, reopen append */
        FS_LOCK();
        fr = f_open(&g_active_file, path, FA_CREATE_NEW | FA_WRITE);
        FS_UNLOCK();
        if (fr != FR_OK) {
            /* If parent was missing, CREATE_NEW returns FR_NO_FILE — double-check dir */
            printf("CREATE_NEW failed for %s (fr=%d)\r\n", path, (int)fr);
            return false;
        }

        /* Write header once */
        {
            const char *hdr = "timestamp_iso8601,sensor,quantity,value,unit\r\n";
            UINT bw = 0U;
            FS_LOCK();
            fr = f_write(&g_active_file, hdr, (UINT)strlen(hdr), &bw);
            if (fr == FR_OK && bw == (UINT)strlen(hdr)) (void)f_sync(&g_active_file);
            (void)f_close(&g_active_file);
            (void)disk_ioctl(0, CTRL_SYNC, NULL);
            FS_UNLOCK();
            if (fr != FR_OK || bw != (UINT)strlen(hdr)) {
                printf("header write/sync failed for %s (fr=%d bw=%u)\r\n",
                       path, (int)fr, (unsigned)bw);
                return false;
            }
        }

        /* Verify via f_stat and reopen for append */
        FS_LOCK();
        fr = f_stat(path, &fi);
        FS_UNLOCK();
        printf("POSTCREATE f_stat(%s) -> fr=%d size=%lu\r\n",
               path, (int)fr, (unsigned long)fi.fsize);
        if (fr != FR_OK) {
            printf("POSTCREATE stat failed (fr=%d)\r\n", (int)fr);
            return false;
        }

        FS_LOCK();
        fr = f_open(&g_active_file, path, FA_OPEN_EXISTING | FA_WRITE);
        if (fr == FR_OK) fr = f_lseek(&g_active_file, f_size(&g_active_file));
        FS_UNLOCK();
        if (fr != FR_OK) {
            printf("reopen/seek failed for %s (fr=%d)\r\n", path, (int)fr);
            return false;
        }
    } else {
        /* Some other f_stat error */
        printf("PRESCAN stat unexpected fr=%d for %s\r\n", (int)fr, path);
        return false;
    }

    g_file_open = true;
    (void)strncpy(g_last_date_yyyy_mm_dd, date_yyyy_mm_dd, sizeof g_last_date_yyyy_mm_dd);
    (void)strncpy(g_active_path, path, sizeof g_active_path - 1);
    g_active_path[sizeof g_active_path - 1] = '\0';

    printf("Daily log ready: %s size=%lu\r\n",
           g_active_path, (unsigned long)f_size(&g_active_file));
    return true;
}



static void measurement_logger_checkpoint_close(void) {
    if (!g_file_open || g_active_path[0] == '\0') return;

    FS_LOCK();
    (void)f_sync(&g_active_file);
    (void)f_close(&g_active_file);
    g_file_open = false;
    (void)disk_ioctl(0, CTRL_SYNC, NULL);
    FS_UNLOCK();

    vTaskDelay(pdMS_TO_TICKS(5));

    FS_LOCK();
    FRESULT fr = f_open(&g_active_file, g_active_path, FA_OPEN_ALWAYS | FA_WRITE);
    if (fr == FR_OK) {
        (void)f_lseek(&g_active_file, f_size(&g_active_file));
        g_file_open = true;
        printf("CHECKPOINT REOPENED: %s size=%lu\r\n", g_active_path, (unsigned long)f_size(&g_active_file));
    } else {
        printf("CHECKPOINT REOPEN FAILED fr=%d on %s\r\n", (int)fr, g_active_path);
    }
    FS_UNLOCK();
}
