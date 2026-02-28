#include "baseline_forecast_task.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>

#include "FreeRTOS.h"
#include "task.h"
#include "cmsis_os.h"
#include "ff.h"
#include "diskio.h"
#include "sd_card.h"
#include "ds3231.h"
#include "forecast_temp_task.h"

#define LOG_PREFIX "[BASELINE_FORECAST] "

#define BASELINE_TASK_STACK_WORDS         (2048u)
#define BASELINE_TASK_PRIORITY            (osPriorityLow)
#define BASELINE_TASK_PERIOD_MS           (300000u)
#define BASELINE_BASE_DIR                 "0:/logs"
#define BASELINE_LOG_PATH                 BASELINE_BASE_DIR "/baseline_forecast.log"
#define BASELINE_SEASON_LENGTH_SLOTS      (48u)
#define BASELINE_FORECAST_HORIZON_SLOTS   (24u)
#define BASELINE_MAX_FILES                (4u)
#define BASELINE_MAX_SLOT_HISTORY         (128u)

#if (FORECAST_TEMP_FORECAST_HORIZON_SLOTS != BASELINE_FORECAST_HORIZON_SLOTS)
#error "Baseline horizon must remain aligned to 12 hours (24 slots) to match CNN output"
#endif

extern osMutexId_t g_fs_mutex;
#define FS_LOCK()   osMutexAcquire(g_fs_mutex, osWaitForever)
#define FS_UNLOCK() osMutexRelease(g_fs_mutex)

typedef struct {
    char filename[48];
    uint32_t date_key;
} baseline_log_file_t;

typedef struct {
    float sum;
    uint16_t count;
} baseline_slot_accumulator_t;

static osThreadId_t g_baseline_task_id = NULL;
static char g_last_logged_timestamp[24] = {0};

static void baseline_forecast_task_entry(void *argument);
static bool baseline_collect_recent_temperature_slots(float *slots_out, size_t slots_capacity, size_t *slots_count_out);
static bool baseline_write_forecast_log_line(const char *timestamp_iso8601, const float *forecast_slots, size_t forecast_count, size_t history_slots);
static bool baseline_ensure_logs_directory(void);
static bool baseline_write_header_if_needed(FIL *file, size_t forecast_count);
static int baseline_compare_file_desc(const void *a, const void *b);

bool baseline_forecast_task_start(void) {
    if (g_baseline_task_id != NULL) {
        return true;
    }

    const osThreadAttr_t task_attributes = {
        .name = "baselineForecast",
        .priority = BASELINE_TASK_PRIORITY,
        .stack_size = BASELINE_TASK_STACK_WORDS * sizeof(StackType_t),
    };

    g_baseline_task_id = osThreadNew(baseline_forecast_task_entry, NULL, &task_attributes);
    return (g_baseline_task_id != NULL);
}

static int baseline_compare_file_desc(const void *a, const void *b) {
    const baseline_log_file_t *lhs = (const baseline_log_file_t *)a;
    const baseline_log_file_t *rhs = (const baseline_log_file_t *)b;
    if (lhs->date_key < rhs->date_key) {
        return 1;
    }
    if (lhs->date_key > rhs->date_key) {
        return -1;
    }
    return 0;
}

static void baseline_forecast_task_entry(void *argument) {
    (void)argument;
    TickType_t last_wake_tick = xTaskGetTickCount();

    while (1) {
        char timestamp[24] = {0};
        if (ds3231_read_time_iso8601_utc_i2c1(timestamp, sizeof(timestamp)) != HAL_OK) {
            printf(LOG_PREFIX "Unable to read DS3231 timestamp; skipping baseline cycle\r\n");
            vTaskDelayUntil(&last_wake_tick, pdMS_TO_TICKS(BASELINE_TASK_PERIOD_MS));
            continue;
        }

        if (strncmp(timestamp, g_last_logged_timestamp, sizeof(g_last_logged_timestamp) - 1u) == 0) {
            vTaskDelayUntil(&last_wake_tick, pdMS_TO_TICKS(BASELINE_TASK_PERIOD_MS));
            continue;
        }

        float history_slots[BASELINE_MAX_SLOT_HISTORY] = {0.0f};
        size_t history_count = 0u;
        if (!baseline_collect_recent_temperature_slots(history_slots, BASELINE_MAX_SLOT_HISTORY, &history_count)) {
            printf(LOG_PREFIX "Unable to read recent measurement history for baseline\r\n");
            vTaskDelayUntil(&last_wake_tick, pdMS_TO_TICKS(BASELINE_TASK_PERIOD_MS));
            continue;
        }

        if (history_count < BASELINE_SEASON_LENGTH_SLOTS) {
            printf(LOG_PREFIX "Not enough history for baseline: have %lu slots need %u\r\n",
                   (unsigned long)history_count,
                   (unsigned int)BASELINE_SEASON_LENGTH_SLOTS);
            vTaskDelayUntil(&last_wake_tick, pdMS_TO_TICKS(BASELINE_TASK_PERIOD_MS));
            continue;
        }

        float forecast_slots[BASELINE_FORECAST_HORIZON_SLOTS] = {0.0f};
        for (size_t horizon_index = 0u; horizon_index < BASELINE_FORECAST_HORIZON_SLOTS; ++horizon_index) {
            /*
             * Seasonal-naive with 24h period: y_hat(t+h) = y(t+h-48 slots).
             * With h starting at 1, the first forecast maps to history_count-47.
             */
            const size_t source_index = (history_count - BASELINE_SEASON_LENGTH_SLOTS + 1u) + horizon_index;
            if (source_index < history_count) {
                forecast_slots[horizon_index] = history_slots[source_index];
            } else {
                forecast_slots[horizon_index] = history_slots[history_count - 1u];
            }
        }

        if (baseline_write_forecast_log_line(timestamp,
                                             forecast_slots,
                                             BASELINE_FORECAST_HORIZON_SLOTS,
                                             history_count)) {
            (void)strncpy(g_last_logged_timestamp, timestamp, sizeof(g_last_logged_timestamp) - 1u);
            g_last_logged_timestamp[sizeof(g_last_logged_timestamp) - 1u] = '\0';
            printf(LOG_PREFIX "Logged seasonal baseline at %s using %lu slots\r\n",
                   timestamp,
                   (unsigned long)history_count);
        }

        vTaskDelayUntil(&last_wake_tick, pdMS_TO_TICKS(BASELINE_TASK_PERIOD_MS));
    }
}

static bool baseline_collect_recent_temperature_slots(float *slots_out, size_t slots_capacity, size_t *slots_count_out) {
    if ((slots_out == NULL) || (slots_count_out == NULL) || (slots_capacity == 0u)) {
        return false;
    }

    *slots_count_out = 0u;

    if (SD_Mount() != FR_OK) {
        return false;
    }

    baseline_log_file_t files[BASELINE_MAX_FILES] = {0};
    size_t file_count = 0u;

    DIR dir;
    FILINFO fno;
    FS_LOCK();
    FRESULT fr = f_opendir(&dir, BASELINE_BASE_DIR);
    FS_UNLOCK();
    if (fr != FR_OK) {
        return false;
    }

    while (1) {
        FS_LOCK();
        fr = f_readdir(&dir, &fno);
        FS_UNLOCK();
        if (fr != FR_OK || fno.fname[0] == '\0') {
            break;
        }
        if (fno.fattrib & AM_DIR) {
            continue;
        }

        unsigned y = 0u, m = 0u, d = 0u;
        if (sscanf(fno.fname, "measurements_%4u-%2u-%2u.csv", &y, &m, &d) == 3) {
            const uint32_t date_key = (y * 10000u) + (m * 100u) + d;
            if (file_count < BASELINE_MAX_FILES) {
                (void)snprintf(files[file_count].filename, sizeof(files[file_count].filename), "%s", fno.fname);
                files[file_count].date_key = date_key;
                ++file_count;
            } else {
                size_t oldest_index = 0u;
                for (size_t i = 1u; i < BASELINE_MAX_FILES; ++i) {
                    if (files[i].date_key < files[oldest_index].date_key) {
                        oldest_index = i;
                    }
                }
                if (date_key > files[oldest_index].date_key) {
                    (void)snprintf(files[oldest_index].filename, sizeof(files[oldest_index].filename), "%s", fno.fname);
                    files[oldest_index].date_key = date_key;
                }
            }
        }
    }

    FS_LOCK();
    (void)f_closedir(&dir);
    FS_UNLOCK();

    if (file_count == 0u) {
        return false;
    }

    qsort(files, file_count, sizeof(files[0]), baseline_compare_file_desc);

    baseline_slot_accumulator_t slot_accumulators[BASELINE_MAX_SLOT_HISTORY] = {0};
    char slot_keys[BASELINE_MAX_SLOT_HISTORY][17];
    memset(slot_keys, 0, sizeof(slot_keys));
    size_t slot_count = 0u;

    for (int32_t file_index = (int32_t)file_count - 1; file_index >= 0; --file_index) {
        char path[96];
        (void)snprintf(path, sizeof(path), BASELINE_BASE_DIR "/%s", files[file_index].filename);

        FIL file;
        FS_LOCK();
        fr = f_open(&file, path, FA_READ);
        FS_UNLOCK();
        if (fr != FR_OK) {
            continue;
        }

        bool header_skipped = false;
        while (1) {
            char line[192];
            FS_LOCK();
            char *line_ptr = f_gets(line, (int)sizeof(line), &file);
            FS_UNLOCK();
            if (line_ptr == NULL) {
                break;
            }

            if (!header_skipped) {
                header_skipped = true;
                continue;
            }

            line[strcspn(line, "\r\n")] = '\0';
            char *timestamp = strtok(line, ",");
            char *sensor = strtok(NULL, ",");
            char *quantity = strtok(NULL, ",");
            char *value_str = strtok(NULL, ",");

            if ((timestamp == NULL) || (sensor == NULL) || (quantity == NULL) || (value_str == NULL)) {
                continue;
            }
            if ((strcmp(sensor, "bme280") != 0) || (strcmp(quantity, "temperature_c") != 0)) {
                continue;
            }
            if (strlen(timestamp) < 16u) {
                continue;
            }

            char slot_key[17] = {0};
            (void)strncpy(slot_key, timestamp, 16u);
            slot_key[16] = '\0';
            const int minute_value = ((slot_key[14] - '0') * 10) + (slot_key[15] - '0');
            if (minute_value < 30) {
                slot_key[14] = '0';
                slot_key[15] = '0';
            } else {
                slot_key[14] = '3';
                slot_key[15] = '0';
            }

            size_t slot_index = slot_count;
            if ((slot_count > 0u) && (strncmp(slot_keys[slot_count - 1u], slot_key, 16u) == 0)) {
                slot_index = slot_count - 1u;
            } else {
                if (slot_count >= BASELINE_MAX_SLOT_HISTORY) {
                    memmove(&slot_accumulators[0], &slot_accumulators[1], sizeof(slot_accumulators[0]) * (BASELINE_MAX_SLOT_HISTORY - 1u));
                    memmove(&slot_keys[0], &slot_keys[1], sizeof(slot_keys[0]) * (BASELINE_MAX_SLOT_HISTORY - 1u));
                    slot_count = BASELINE_MAX_SLOT_HISTORY - 1u;
                }
                slot_index = slot_count;
                (void)strncpy(slot_keys[slot_index], slot_key, sizeof(slot_keys[slot_index]) - 1u);
                slot_keys[slot_index][sizeof(slot_keys[slot_index]) - 1u] = '\0';
                slot_accumulators[slot_index].sum = 0.0f;
                slot_accumulators[slot_index].count = 0u;
                ++slot_count;
            }

            slot_accumulators[slot_index].sum += strtof(value_str, NULL);
            slot_accumulators[slot_index].count++;
        }

        FS_LOCK();
        (void)f_close(&file);
        FS_UNLOCK();
    }

    size_t written = 0u;
    for (size_t i = 0u; i < slot_count; ++i) {
        if (slot_accumulators[i].count == 0u) {
            continue;
        }
        if (written >= slots_capacity) {
            break;
        }
        slots_out[written++] = slot_accumulators[i].sum / (float)slot_accumulators[i].count;
    }

    *slots_count_out = written;
    return (written > 0u);
}

static bool baseline_ensure_logs_directory(void) {
    FILINFO fi;
    FS_LOCK();
    FRESULT fr = f_stat(BASELINE_BASE_DIR, &fi);
    FS_UNLOCK();

    if (fr == FR_OK) {
        return ((fi.fattrib & AM_DIR) != 0u);
    }

    if (fr == FR_NO_FILE) {
        FS_LOCK();
        fr = f_mkdir(BASELINE_BASE_DIR);
        FS_UNLOCK();
        return (fr == FR_OK || fr == FR_EXIST);
    }

    return false;
}

static bool baseline_write_header_if_needed(FIL *file, size_t forecast_count) {
    if (f_size(file) > 0u) {
        return true;
    }

    char header[768];
    size_t used = 0u;
    int n = snprintf(header, sizeof(header),
                     "timestamp_iso8601,method,history_slots,season_length_slots");
    if (n <= 0 || (size_t)n >= sizeof(header)) {
        return false;
    }
    used = (size_t)n;

    for (size_t i = 0u; i < forecast_count; ++i) {
        n = snprintf(&header[used], sizeof(header) - used,
                     ",forecast_t+%lum_c",
                     (unsigned long)((i + 1u) * FORECAST_TEMP_MINUTES_PER_SLOT));
        if (n <= 0 || (size_t)n >= (sizeof(header) - used)) {
            return false;
        }
        used += (size_t)n;
    }

    if ((used + 2u) >= sizeof(header)) {
        return false;
    }
    header[used++] = '\r';
    header[used++] = '\n';

    UINT bw = 0u;
    FRESULT fr = f_write(file, header, (UINT)used, &bw);
    if (fr != FR_OK || bw != used) {
        return false;
    }

    fr = f_sync(file);
    return (fr == FR_OK);
}

static bool baseline_write_forecast_log_line(const char *timestamp_iso8601, const float *forecast_slots, size_t forecast_count, size_t history_slots) {
    if ((timestamp_iso8601 == NULL) || (forecast_slots == NULL) || (forecast_count == 0u)) {
        return false;
    }

    if (SD_Mount() != FR_OK) {
        return false;
    }

    if (!baseline_ensure_logs_directory()) {
        return false;
    }

    FIL file;
    FS_LOCK();
    FRESULT fr = f_open(&file, BASELINE_LOG_PATH, FA_OPEN_ALWAYS | FA_WRITE);
    if (fr == FR_OK) {
        fr = f_lseek(&file, f_size(&file));
    }
    if (fr == FR_OK) {
        if (!baseline_write_header_if_needed(&file, forecast_count)) {
            fr = FR_INT_ERR;
        }
    }

    char line[768];
    int n = snprintf(line, sizeof(line), "%s,seasonal_naive_24h,%lu,%u",
                     timestamp_iso8601,
                     (unsigned long)history_slots,
                     (unsigned int)BASELINE_SEASON_LENGTH_SLOTS);
    size_t used = 0u;
    if (n <= 0 || (size_t)n >= sizeof(line)) {
        fr = FR_INT_ERR;
    } else {
        used = (size_t)n;
    }
    if (fr == FR_OK) {
        for (size_t i = 0u; i < forecast_count; ++i) {
            n = snprintf(&line[used], sizeof(line) - used, ",%.4f", forecast_slots[i]);
            if (n <= 0 || (size_t)n >= (sizeof(line) - used)) {
                fr = FR_INT_ERR;
                break;
            }
            used += (size_t)n;
        }
    }

    if (fr == FR_OK) {
        if ((used + 2u) < sizeof(line)) {
            line[used++] = '\r';
            line[used++] = '\n';
        } else {
            fr = FR_INT_ERR;
        }
    }

    if (fr == FR_OK) {
        UINT bw = 0u;
        fr = f_write(&file, line, (UINT)used, &bw);
        if (fr != FR_OK || bw != used) {
            fr = FR_DISK_ERR;
        }
    }

    if (fr == FR_OK) {
        fr = f_sync(&file);
    }
    if (fr == FR_OK) {
        (void)disk_ioctl(0, CTRL_SYNC, NULL);
    }
    (void)f_close(&file);
    FS_UNLOCK();

    return (fr == FR_OK);
}
