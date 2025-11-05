#include "inference_logger_task.h"

#include <stdio.h>
#include <string.h>

#include "FreeRTOS.h"
#include "cmsis_os.h"
#include "ds3231.h"
#include "diskio.h"
#include "ff.h"
#include "forecast_temp_task.h"
#include "led_service.h"
#include "sd_card.h"
#include "task.h"

#ifndef INFERENCE_LOGGER_TASK_STACK_WORDS
#define INFERENCE_LOGGER_TASK_STACK_WORDS   (512u)
#endif
#ifndef INFERENCE_LOGGER_TASK_PRIORITY
#define INFERENCE_LOGGER_TASK_PRIORITY      (osPriorityLow)
#endif
#ifndef INFERENCE_LOGGER_BUFFER_BYTES
#define INFERENCE_LOGGER_BUFFER_BYTES       (192u)
#endif
#ifndef INFERENCE_LOGGER_BASE_DIR
#define INFERENCE_LOGGER_BASE_DIR           "0:/logs"
#endif
#ifndef INFERENCE_LOGGER_BACKOFF_MS
#define INFERENCE_LOGGER_BACKOFF_MS         (500u)
#endif
#ifndef INFERENCE_LOGGER_PERIOD_MS
#define INFERENCE_LOGGER_PERIOD_MS          (300000u)
#endif

extern osMutexId_t g_fs_mutex;
#ifndef FS_LOCK
#define FS_LOCK()   osMutexAcquire(g_fs_mutex, osWaitForever)
#define FS_UNLOCK() osMutexRelease(g_fs_mutex)
#endif

static void inference_logger_task_entry(void *argument);
static bool inference_logger_ensure_directory_exists(const char *dir_path);
static bool inference_logger_open_today_file(const char *date_yyyy_mm_dd);
static bool inference_logger_flush_buffer_to_file(void);
static size_t inference_logger_format_csv_line(const char *timestamp_iso8601,
                                               float predicted_temperature_c,
                                               char *out_line,
                                               size_t out_capacity);

static TaskHandle_t g_inference_logger_task_handle = NULL;
static FIL g_active_file;
static bool g_file_open = false;
static char g_last_date_yyyy_mm_dd[11] = {0};
static char g_active_path[128] = {0};
#if defined(__GNUC__)
__attribute__((aligned(32)))
#endif
__attribute__((section(".sram1"), aligned(32)))
static char g_write_buffer[INFERENCE_LOGGER_BUFFER_BYTES];
static size_t g_write_used = 0u;

typedef enum {
    INFERENCE_LOGGER_STATE_STARTUP = 0,
    INFERENCE_LOGGER_STATE_ENSURE_DIR,
    INFERENCE_LOGGER_STATE_OPEN_TODAY,
    INFERENCE_LOGGER_STATE_RUNNING,
    INFERENCE_LOGGER_STATE_FLUSH,
    INFERENCE_LOGGER_STATE_ROTATE,
    INFERENCE_LOGGER_STATE_ERROR_BACKOFF
} inference_logger_state_t;

bool inference_logger_task_create(void) {
    if (g_inference_logger_task_handle != NULL) {
        return true;
    }

    const osThreadAttr_t attr = {
        .name = "inference_logger_task",
        .priority = (osPriority_t) INFERENCE_LOGGER_TASK_PRIORITY,
        .stack_size = (uint32_t)(INFERENCE_LOGGER_TASK_STACK_WORDS * sizeof(StackType_t)),
    };
    g_inference_logger_task_handle =
        osThreadNew(inference_logger_task_entry, NULL, &attr);
    return (g_inference_logger_task_handle != NULL);
}

static void inference_logger_task_entry(void *argument) {
    (void)argument;
    inference_logger_state_t state = INFERENCE_LOGGER_STATE_STARTUP;
    static TickType_t s_next_poll_tick = 0;
    static bool s_poll_initialized = false;

    for (;;) {
        switch (state) {
        case INFERENCE_LOGGER_STATE_STARTUP:
            if (SD_Mount() == FR_OK) {
                state = INFERENCE_LOGGER_STATE_ENSURE_DIR;
            } else {
                state = INFERENCE_LOGGER_STATE_ERROR_BACKOFF;
            }
            break;

        case INFERENCE_LOGGER_STATE_ENSURE_DIR:
            if (inference_logger_ensure_directory_exists(INFERENCE_LOGGER_BASE_DIR)) {
                state = INFERENCE_LOGGER_STATE_OPEN_TODAY;
            } else {
                state = INFERENCE_LOGGER_STATE_ERROR_BACKOFF;
            }
            break;

        case INFERENCE_LOGGER_STATE_OPEN_TODAY: {
            char ts[24];
            if (ds3231_read_time_iso8601_utc_i2c1(ts, sizeof ts) != HAL_OK) {
                (void)snprintf(ts, sizeof ts, "2000-01-01T00:00:00Z");
            }
            char date[11];
            memcpy(date, ts, 10);
            date[10] = '\0';

            if (inference_logger_open_today_file(date)) {
                s_poll_initialized = false;
                state = INFERENCE_LOGGER_STATE_RUNNING;
            } else {
                state = INFERENCE_LOGGER_STATE_ERROR_BACKOFF;
            }
            break;
        }

        case INFERENCE_LOGGER_STATE_RUNNING: {
            if (!s_poll_initialized) {
                s_next_poll_tick = xTaskGetTickCount();
                s_poll_initialized = true;
            } else {
                vTaskDelayUntil(&s_next_poll_tick,
                                pdMS_TO_TICKS(INFERENCE_LOGGER_PERIOD_MS));
            }

            char ts_now[24];
            if (ds3231_read_time_iso8601_utc_i2c1(ts_now, sizeof ts_now) != HAL_OK) {
                (void)snprintf(ts_now, sizeof ts_now, "2000-01-01T00:00:00Z");
            }
            char date_now[11];
            memcpy(date_now, ts_now, 10);
            date_now[10] = '\0';

            if (strncmp(date_now, g_last_date_yyyy_mm_dd, 10) != 0) {
                state = INFERENCE_LOGGER_STATE_ROTATE;
                break;
            }

            float predicted_temperature_c = 0.0f;
            if (forecast_temp_get_latest_prediction(&predicted_temperature_c)) {
                char line[96];
                size_t len =
                    inference_logger_format_csv_line(ts_now, predicted_temperature_c,
                                                      line, sizeof line);
                if (len > 0u) {
                    if ((g_write_used + len) > sizeof g_write_buffer) {
                        state = INFERENCE_LOGGER_STATE_FLUSH;
                        break;
                    }
                    memcpy(&g_write_buffer[g_write_used], line, len);
                    g_write_used += len;
                    state = INFERENCE_LOGGER_STATE_FLUSH;
                    break;
                }
            }
            break;
        }

        case INFERENCE_LOGGER_STATE_FLUSH: {
            const size_t before = g_write_used;
            if (before == 0u) {
                state = INFERENCE_LOGGER_STATE_RUNNING;
                break;
            }

            if (inference_logger_flush_buffer_to_file()) {
                state = INFERENCE_LOGGER_STATE_RUNNING;
            } else {
                state = INFERENCE_LOGGER_STATE_ERROR_BACKOFF;
            }
            break;
        }

        case INFERENCE_LOGGER_STATE_ROTATE:
            if (g_file_open) {
                (void)inference_logger_flush_buffer_to_file();
                FS_LOCK();
                (void)f_close(&g_active_file);
                FS_UNLOCK();
                g_file_open = false;
            }
            state = INFERENCE_LOGGER_STATE_OPEN_TODAY;
            break;

        case INFERENCE_LOGGER_STATE_ERROR_BACKOFF:
            if (g_file_open) {
                FS_LOCK();
                (void)f_close(&g_active_file);
                FS_UNLOCK();
                g_file_open = false;
            }
            vTaskDelay(pdMS_TO_TICKS(INFERENCE_LOGGER_BACKOFF_MS));
            state = INFERENCE_LOGGER_STATE_STARTUP;
            break;
        }
    }
}

static bool inference_logger_flush_buffer_to_file(void) {
    const size_t bytes_to_write = g_write_used;

    if (bytes_to_write == 0u) {
        return true;
    }

    if (!g_file_open) {
        FS_LOCK();
        FRESULT fr = f_open(&g_active_file, g_active_path, FA_OPEN_ALWAYS | FA_WRITE);
        if (fr == FR_OK) {
            if (f_size(&g_active_file) == 0) {
                const char *hdr = "timestamp_iso8601,predicted_temperature_c\r\n";
                UINT bw = 0;
                (void)f_write(&g_active_file, hdr, (UINT)strlen(hdr), &bw);
                if (bw == (UINT)strlen(hdr)) {
                    (void)f_sync(&g_active_file);
                }
                (void)disk_ioctl(0, CTRL_SYNC, NULL);
            }
            (void)f_lseek(&g_active_file, f_size(&g_active_file));
            g_file_open = true;
        }
        FS_UNLOCK();

        if (!g_file_open) {
            return false;
        }
    }

    UINT bw = 0u;
    FS_LOCK();
    FRESULT wr = f_write(&g_active_file, g_write_buffer, (UINT)bytes_to_write, &bw);
    if (wr == FR_OK && bw == bytes_to_write) {
        wr = f_sync(&g_active_file);
    }
    FS_UNLOCK();

    if (wr != FR_OK || bw != bytes_to_write) {
        return false;
    }

    led_service_activity_bump(1000);
    g_write_used = 0u;
    return true;
}

static bool inference_logger_ensure_directory_exists(const char *dir_path) {
    FILINFO fi;
    FRESULT fr;

    FS_LOCK();
    fr = f_stat(dir_path, &fi);
    FS_UNLOCK();

    if (fr == FR_OK) {
        return ((fi.fattrib & AM_DIR) != 0);
    }

    if (fr == FR_NO_FILE) {
        FS_LOCK();
        fr = f_mkdir(dir_path);
        FS_UNLOCK();
        return (fr == FR_OK || fr == FR_EXIST);
    }

    return false;
}

static bool inference_logger_open_today_file(const char *date_yyyy_mm_dd) {
    char path[128];
    (void)snprintf(path, sizeof path,
                   INFERENCE_LOGGER_BASE_DIR "/inference__%s.csv", date_yyyy_mm_dd);

    FILINFO fi;
    FRESULT fr;

    FS_LOCK();
    fr = f_stat(path, &fi);
    FS_UNLOCK();

    if (fr == FR_OK) {
        FS_LOCK();
        fr = f_open(&g_active_file, path, FA_OPEN_EXISTING | FA_WRITE);
        if (fr == FR_OK) {
            fr = f_lseek(&g_active_file, f_size(&g_active_file));
        }
        FS_UNLOCK();
        if (fr != FR_OK) {
            return false;
        }
    } else if (fr == FR_NO_FILE) {
        FS_LOCK();
        fr = f_open(&g_active_file, path, FA_CREATE_NEW | FA_WRITE);
        if (fr == FR_OK) {
            const char *hdr = "timestamp_iso8601,predicted_temperature_c\r\n";
            UINT bw = 0U;
            fr = f_write(&g_active_file, hdr, (UINT)strlen(hdr), &bw);
            if (fr == FR_OK && bw == (UINT)strlen(hdr)) {
                (void)f_sync(&g_active_file);
            }
            (void)f_close(&g_active_file);
            (void)disk_ioctl(0, CTRL_SYNC, NULL);
        }
        FS_UNLOCK();
        if (fr != FR_OK) {
            return false;
        }

        FS_LOCK();
        fr = f_open(&g_active_file, path, FA_OPEN_EXISTING | FA_WRITE);
        if (fr == FR_OK) {
            fr = f_lseek(&g_active_file, f_size(&g_active_file));
        }
        FS_UNLOCK();
        if (fr != FR_OK) {
            return false;
        }
    } else {
        return false;
    }

    g_file_open = true;
    strncpy(g_last_date_yyyy_mm_dd, date_yyyy_mm_dd,
            sizeof g_last_date_yyyy_mm_dd);
    g_last_date_yyyy_mm_dd[sizeof g_last_date_yyyy_mm_dd - 1] = '\0';
    strncpy(g_active_path, path, sizeof g_active_path);
    g_active_path[sizeof g_active_path - 1] = '\0';
    g_write_used = 0u;
    return true;
}

static size_t inference_logger_format_csv_line(const char *timestamp_iso8601,
                                               float predicted_temperature_c,
                                               char *out_line,
                                               size_t out_capacity) {
    int n = snprintf(out_line, out_capacity, "%s,%.6f\r\n",
                     (timestamp_iso8601 != NULL) ? timestamp_iso8601 : "TIME?",
                     (double)predicted_temperature_c);
    if (n <= 0 || (size_t)n >= out_capacity) {
        return 0u;
    }
    return (size_t)n;
}

