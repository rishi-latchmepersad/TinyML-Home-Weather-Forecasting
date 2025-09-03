#include "measurement_logger_task.h"

#include <stdio.h>
#include <string.h>

#include "ds3231.h"            /* ds3231_read_time_iso8601_utc_i2c1() */
#include "diskio.h"
#include "led_service.h"
#include "ff.h"

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

/****************************************************************************************
 * Function:    measurement_logger_debug_snapshot
 * Purpose:     Sanity-check the mounted volume and list entries in "0:/" and "0:/logs".
 * Params:      void
 * Returns:     void
 * Side Effects:
 *              - Prints volume label/serial, free space, and directory listings.
 * Preconditions:
 *              - Volume 0 must be mounted.
 * Notes:
 *              - Use sparingly; directory walks can be slow over SPI.
 ****************************************************************************************/
static void measurement_logger_debug_snapshot(void) {
	char label[24] = { 0 };
	DWORD vsn = 0;
	FATFS *fs;
	DWORD clusters_free = 0, clusters_total = 0;
	FRESULT fr;

	fr = f_getlabel("0:", label, &vsn);
	printf("VOL: fr=%d label=\"%s\" vsn=%08lX\r\n", (int) fr, label,
			(unsigned long) vsn);

	fr = f_getfree("0:", &clusters_free, &fs);
	if (fr == FR_OK && fs) {
		clusters_total = (fs->n_fatent - 2);
		uint64_t bytes_per_cluster = (uint64_t) fs->csize * 512u;
		uint64_t bytes_free = (uint64_t) clusters_free * bytes_per_cluster;
		uint64_t bytes_total = (uint64_t) clusters_total * bytes_per_cluster;
		printf("FS: total=%lu KB free=%lu KB\r\n",
				(unsigned long) (bytes_total / 1024u),
				(unsigned long) (bytes_free / 1024u));
	} else {
		printf("FS: f_getfree failed fr=%d\r\n", (int) fr);
	}

	DIR dir;
	FILINFO fi;
	printf("DIR 0:/\r\n");
	if (f_opendir(&dir, "0:/") == FR_OK) {
		while (f_readdir(&dir, &fi) == FR_OK && fi.fname[0]) {
			printf("  %c %10lu %s\r\n", (fi.fattrib & AM_DIR) ? 'D' : 'F',
					(unsigned long) fi.fsize, fi.fname);
		}
		f_closedir(&dir);
	}

	printf("DIR 0:/logs\r\n");
	if (f_opendir(&dir, "0:/logs") == FR_OK) {
		while (f_readdir(&dir, &fi) == FR_OK && fi.fname[0]) {
			printf("  %c %10lu %s\r\n", (fi.fattrib & AM_DIR) ? 'D' : 'F',
					(unsigned long) fi.fsize, fi.fname);
		}
		f_closedir(&dir);
	}
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
static void measurement_logger_force_commit_and_reopen(void)
{
    FRESULT fr;
    FILINFO fi = {0};

    if (g_file_open) {
        (void)f_sync(&g_active_file);
        (void)f_close(&g_active_file);
        g_file_open = false;
        printf("FORCE: closed %s\r\n", g_active_path);
    }

    /* Ground truth: after close, file should exist on media now */
    fr = f_stat(g_active_path, &fi);
    printf("FORCE: f_stat(%s) -> fr=%d size=%lu\r\n",
           g_active_path, (int)fr, (unsigned long)fi.fsize);

    /* Reopen for append so logging continues */
    fr = f_open(&g_active_file, g_active_path, FA_OPEN_EXISTING | FA_WRITE);
    if (fr == FR_OK) {
        (void)f_lseek(&g_active_file, f_size(&g_active_file));
        g_file_open = true;
        printf("FORCE: reopened append, size=%lu\r\n",
               (unsigned long)f_size(&g_active_file));
    } else {
        printf("FORCE: reopen failed fr=%d\r\n", (int)fr);
    }
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
				printf("g_write_buffer @ %p\r\n", (void*)g_write_buffer);
				s_buf_addr_printed = true;
			}
			/* Filesystem is mounted by SD_Mount() (main.c). Move on. */
			state = LOGGER_STATE_ENSURE_DIR;
			break;

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
				(void) snprintf(ts, sizeof ts, "2000-01-01T00:00:00Z");
			}
			char date[11];
			memcpy(date, ts, 10);
			date[10] = '\0';

			if (measurement_logger_open_today_file(date)) {
				last_flush_tick = xTaskGetTickCount();
				printf("Current time: %s \r\n", ts);
				printf("Current open file: %s \r\n", g_active_path);
				measurement_logger_debug_snapshot();
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
				(void) snprintf(ts_now, sizeof ts_now, "2000-01-01T00:00:00Z");
			}
			const size_t before = g_write_used;
			if (!measurement_logger_flush_buffer_to_file()) {
				state = LOGGER_STATE_ERROR_BACKOFF;
				printf("Buffered data failed to flush to file.\r\n");
			} else {
				g_sync_requested = false;
				last_flush_tick = xTaskGetTickCount();
				if (before > 0u) {
					printf("Flushed %lu bytes to file %s on %s.\r\n",
							(unsigned long) before, g_active_path, ts_now);
					led_service_activity_bump(1000);
					static bool s_force_once_done = false;
					if (!s_force_once_done) {
					    s_force_once_done = true;
					    measurement_logger_force_commit_and_reopen();
					}
					measurement_logger_debug_file_size();
					/* Optional: close & reopen to force directory updates to media */
#if 0 /* set to 0 after testing */
					(void) f_close(&g_active_file);
					g_file_open = false;
					/* Reopen without header to keep appending */
					if (!measurement_logger_open_today_file(
							g_last_date_yyyy_mm_dd)) {
						printf("Reopen failed after flush\r\n");
						state = LOGGER_STATE_ERROR_BACKOFF;
						break;
					}
#endif

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
			/* Close file if needed, back off, and restart from directory ensure */
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
			state = LOGGER_STATE_ENSURE_DIR;
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
	if (!g_file_open || (g_write_used == 0u)) {
		g_write_used = 0u;
		return true;
	}
	UINT bw = 0u;
	FRESULT fr = f_write(&g_active_file, g_write_buffer, (UINT) g_write_used,
			&bw);
	if ((fr != FR_OK) || (bw != g_write_used)) {
		return false;
	}
	g_write_used = 0u;

	(void) f_sync(&g_active_file);
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
 * Purpose:     Ensure today's CSV exists at LOGGER_BASE_DIR and open it in append mode.
 *              If not present, create with FA_CREATE_NEW, write header, f_sync + close
 *              to force the directory entry and size to media, then reopen for append.
 *
 * Params:      date_yyyy_mm_dd  Pointer to 10-char "YYYY-MM-DD" string.
 *
 * Returns:     bool  true on success (file open at EOF), false on error.
 *
 * Side Effects:
 *   - Flushes and closes the previous day's file if date rolled over.
 *   - Updates g_file_open, g_active_path, g_last_date_yyyy_mm_dd.
 *   - Creates LOGGER_BASE_DIR if missing.
 *
 * Preconditions:
 *   - Volume "0:" mounted with FATFS work area in AXI/SRAM (e.g., g_fs in .sram1).
 *   - LOGGER_BASE_DIR is a valid path, e.g., "0:/logs".
 *
 * Postconditions:
 *   - On success, directory entry for new file is durable (visible after safe remove).
 *
 * Concurrency:
 *   - Not ISR-safe. Caller must serialize against any other code that mounts/unmounts
 *     or accesses the same path. Avoid remounts while this function runs.
 *
 * Timing:      Performs file I/O and an f_sync on first creation.
 *
 * Errors:
 *   - Prints FatFs FRESULT codes on all failure paths.
 *
 * Notes:
 *   - We use FA_CREATE_NEW on first create (no truncate), then close & reopen to
 *     cement the directory entry before subsequent appends.
 ****************************************************************************************/
static bool measurement_logger_open_today_file(const char *date_yyyy_mm_dd)
{
    FRESULT fr;
    FILINFO fi;
    char    path[128];

    /* If we already have today's file open, nothing to do. */
    if (g_file_open) {
        if (strncmp(g_last_date_yyyy_mm_dd, date_yyyy_mm_dd, 10) == 0) {
            return true;
        }
        /* Date changed: flush and close the previous file. */
        (void) measurement_logger_flush_buffer_to_file();
        (void) f_close(&g_active_file);
        g_file_open = false;
    }

    /* Ensure base directory exists. Ignore FR_EXIST. */
    fr = f_mkdir(LOGGER_BASE_DIR);
    if ((fr != FR_OK) && (fr != FR_EXIST)) {
        printf("mkdir(%s) failed (fr=%d)\r\n", LOGGER_BASE_DIR, (int)fr);
        return false;
    }

    /* Build today's path: "0:/logs/measurements_YYYY-MM-DD.csv" */
    (void)snprintf(path, sizeof path, LOGGER_BASE_DIR "/measurements_%s.csv", date_yyyy_mm_dd);

    /* If it already exists, open and seek to EOF. */
    fr = f_stat(path, &fi);
    if (fr == FR_OK) {
        fr = f_open(&g_active_file, path, FA_OPEN_EXISTING | FA_WRITE);
        if (fr != FR_OK) {
            printf("f_open(EXISTING) failed for %s (fr=%d)\r\n", path, (int)fr);
            return false;
        }
        fr = f_lseek(&g_active_file, f_size(&g_active_file));
        if (fr != FR_OK) {
            printf("f_lseek(EOF) failed for %s (fr=%d)\r\n", path, (int)fr);
            (void)f_close(&g_active_file);
            return false;
        }
        g_file_open = true;
        (void)strncpy(g_last_date_yyyy_mm_dd, date_yyyy_mm_dd, sizeof g_last_date_yyyy_mm_dd);
        (void)strncpy(g_active_path, path, sizeof g_active_path - 1);
        g_active_path[sizeof g_active_path - 1] = '\0';
        printf("Opened existing daily log: %s size=%lu\r\n",
               g_active_path, (unsigned long)f_size(&g_active_file));
        return true;
    }
    else if (fr != FR_NO_FILE) {
        printf("f_stat(%s) failed (fr=%d)\r\n", path, (int)fr);
        return false;
    }

    /* Brand-new file path: create, write header, sync, close, reopen-append. */
    fr = f_open(&g_active_file, path, FA_CREATE_NEW | FA_WRITE);
    if (fr != FR_OK) {
        /* Race: if another task created it, fall back to open existing. */
        if (fr == FR_EXIST) {
            fr = f_open(&g_active_file, path, FA_OPEN_EXISTING | FA_WRITE);
            if (fr != FR_OK) {
                printf("f_open(EXIST after race) failed for %s (fr=%d)\r\n", path, (int)fr);
                return false;
            }
            (void)f_lseek(&g_active_file, f_size(&g_active_file));
            g_file_open = true;
            (void)strncpy(g_last_date_yyyy_mm_dd, date_yyyy_mm_dd, sizeof g_last_date_yyyy_mm_dd);
            (void)strncpy(g_active_path, path, sizeof g_active_path - 1);
            g_active_path[sizeof g_active_path - 1] = '\0';
            printf("Opened existing (race) daily log: %s size=%lu\r\n",
                   g_active_path, (unsigned long)f_size(&g_active_file));
            return true;
        }
        printf("f_open(CREATE_NEW) failed for %s (fr=%d)\r\n", path, (int)fr);
        return false;
    }

    /* Write CSV header once. */
    {
        const char *hdr = "timestamp_iso8601,sensor,quantity,value,unit\r\n";
        UINT        bw  = 0U;

        fr = f_write(&g_active_file, hdr, (UINT)strlen(hdr), &bw);
        if ((fr != FR_OK) || (bw != (UINT)strlen(hdr))) {
            printf("header write failed for %s (fr=%d, bw=%u)\r\n", path, (int)fr, (unsigned)bw);
            (void)f_close(&g_active_file);
            return false;
        }
        fr = f_sync(&g_active_file);
        if (fr != FR_OK) {
            printf("header f_sync failed for %s (fr=%d)\r\n", path, (int)fr);
            (void)f_close(&g_active_file);
            return false;
        }
    }

    /* Close to cement dir entry + size, then reopen for append. */
    (void)f_close(&g_active_file);

    /* Ground-truth: after close, this SHOULD exist now. */
    memset(&fi, 0, sizeof fi);
    fr = f_stat(path, &fi);
    printf("AFTER OPEN: f_stat(%s) -> fr=%d size=%lu\r\n", path, (int)fr, (unsigned long)fi.fsize);

    fr = f_open(&g_active_file, path, FA_OPEN_EXISTING | FA_WRITE);
    if (fr != FR_OK) {
        /* Fallback: very defensively, try CREATE_ALWAYS once, then reopen-append. */
        printf("reopen(EXISTING) failed for %s (fr=%d) — trying CREATE_ALWAYS\r\n", path, (int)fr);
        fr = f_open(&g_active_file, path, FA_CREATE_ALWAYS | FA_WRITE);
        if (fr != FR_OK) {
            printf("CREATE_ALWAYS also failed for %s (fr=%d)\r\n", path, (int)fr);
            return false;
        }
        /* Re-write header because CREATE_ALWAYS truncates. */
        {
            const char *hdr = "timestamp_iso8601,sensor,quantity,value,unit\r\n";
            UINT        bw  = 0U;
            (void)f_write(&g_active_file, hdr, (UINT)strlen(hdr), &bw);
            (void)f_sync(&g_active_file);
        }
    }

    fr = f_lseek(&g_active_file, f_size(&g_active_file));
    if (fr != FR_OK) {
        printf("reopen lseek(EOF) failed for %s (fr=%d)\r\n", path, (int)fr);
        (void)f_close(&g_active_file);
        return false;
    }

    g_file_open = true;
    (void)strncpy(g_last_date_yyyy_mm_dd, date_yyyy_mm_dd, sizeof g_last_date_yyyy_mm_dd);
    (void)strncpy(g_active_path, path, sizeof g_active_path - 1);
    g_active_path[sizeof g_active_path - 1] = '\0';

    printf("Created new daily log: %s\r\n", g_active_path);
    return true;
}

