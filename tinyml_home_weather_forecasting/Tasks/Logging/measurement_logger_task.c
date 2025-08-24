#include "measurement_logger_task.h"

#include <stdio.h>
#include <string.h>

#include "ds3231.h"            /* ds3231_read_time_iso8601_utc_i2c1() */
#include "diskio.h"

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
#define LOGGER_FLUSH_PERIOD_MS    (1000u)      /* periodic flush cadence */
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
			/* Reset per-run state */
			g_file_open = false;
			g_write_used = 0u;
			g_last_date_yyyy_mm_dd[0] = '\0';
			state = LOGGER_STATE_ENSURE_DIR;
			last_sync_tick = xTaskGetTickCount();
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
				state = LOGGER_STATE_ROTATE;
				break;
			}

			/* Service the queue */
			measurement_logger_message_t msg;
			if (xQueueReceive(g_measurement_queue, &msg,
					pdMS_TO_TICKS(200u)) == pdPASS) {
				printf("New data received on queue\r\n");
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
			const size_t before = g_write_used;
			if (!measurement_logger_flush_buffer_to_file()) {
				state = LOGGER_STATE_ERROR_BACKOFF;
				printf("Buffered data failed to flush to file.\r\n");
			} else {
				g_sync_requested = false;
				last_flush_tick = xTaskGetTickCount();
				if (before > 0u) {
					printf("Flushed %lu bytes to file.\r\n",
							(unsigned long) before);

				} else {
					printf("Flush skipped (no pending bytes).\r\n");
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
 * Purpose:     Open today's CSV, writing header when newly created.
 * Parameters:  date_yyyy_mm_dd [in]  "YYYY-MM-DD"
 * Returns:     bool - true on success.
 ****************************************************************************************/
static bool measurement_logger_open_today_file(const char *date_yyyy_mm_dd) {
	/* Close previous file if open or if date changed */
	if (g_file_open) {
		if (strncmp(g_last_date_yyyy_mm_dd, date_yyyy_mm_dd, 10) == 0) {
			return true;
		}
		(void) measurement_logger_flush_buffer_to_file();
		(void) f_close(&g_active_file);
		g_file_open = false;
	}

	char path[128];
	(void) snprintf(path, sizeof path, LOGGER_BASE_DIR "/measurements_%s.csv",
			date_yyyy_mm_dd);

	/* Decide whether we need a header */
	FILINFO fi;
	bool need_header = false;
	FRESULT fr = f_stat(path, &fi);
	if (fr == FR_NO_FILE) {
		need_header = true;
	} else if (fr != FR_OK) {
		return false;
	}

	fr = f_open(&g_active_file, path, FA_OPEN_ALWAYS | FA_WRITE);
	if (fr != FR_OK) {
		return false;
	}

	/* Append mode */
	fr = f_lseek(&g_active_file, f_size(&g_active_file));
	if (fr != FR_OK) {
		(void) f_close(&g_active_file);
		return false;
	}

	g_file_open = true;
	(void) strncpy(g_last_date_yyyy_mm_dd, date_yyyy_mm_dd,
			sizeof g_last_date_yyyy_mm_dd);

	if (need_header || (f_size(&g_active_file) == 0u)) {
		const char *hdr = "timestamp_iso8601,sensor,quantity,value,unit\r\n";
		UINT bw = 0u;
		fr = f_write(&g_active_file, hdr, (UINT) strlen(hdr), &bw);
		if ((fr != FR_OK) || (bw != strlen(hdr))) {
			(void) f_close(&g_active_file);
			g_file_open = false;
			return false;
		}
		(void) f_sync(&g_active_file);
	}
	return true;
}
