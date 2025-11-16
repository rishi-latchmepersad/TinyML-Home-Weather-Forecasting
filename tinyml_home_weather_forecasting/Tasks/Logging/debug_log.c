#include "debug_log.h"

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "FreeRTOS.h"
#include "cmsis_os.h"
#include "ff.h"
#include "task.h"

extern osMutexId_t g_fs_mutex;
#ifndef FS_LOCK
#define FS_LOCK()   osMutexAcquire(g_fs_mutex, osWaitForever)
#define FS_UNLOCK() osMutexRelease(g_fs_mutex)
#endif

#ifndef DEBUG_LOG_TASK_STACK_WORDS
#define DEBUG_LOG_TASK_STACK_WORDS   (768u)
#endif
#ifndef DEBUG_LOG_TASK_PRIORITY
#define DEBUG_LOG_TASK_PRIORITY      (osPriorityLow)
#endif
#ifndef DEBUG_LOG_BUFFER_BYTES
#define DEBUG_LOG_BUFFER_BYTES       (4096u)
#endif
#ifndef DEBUG_LOG_WRITE_CHUNK_BYTES
#define DEBUG_LOG_WRITE_CHUNK_BYTES  (256u)
#endif
#ifndef DEBUG_LOG_MAX_BYTES
#define DEBUG_LOG_MAX_BYTES          (256u * 1024u)
#endif
#ifndef DEBUG_LOG_DIR
#define DEBUG_LOG_DIR                "0:/logs"
#endif
#ifndef DEBUG_LOG_PATH
#define DEBUG_LOG_PATH               "0:/logs/debug_log.txt"
#endif
#ifndef DEBUG_LOG_PREVIOUS_PATH
#define DEBUG_LOG_PREVIOUS_PATH      "0:/logs/debug_log.prev"
#endif
#ifndef DEBUG_LOG_FLUSH_PERIOD_MS
#define DEBUG_LOG_FLUSH_PERIOD_MS    (2000u)
#endif

typedef struct {
    uint8_t buffer[DEBUG_LOG_BUFFER_BYTES];
    size_t head;
    size_t tail;
    size_t count;
} debug_ring_t;

static debug_ring_t g_ring = { .head = 0u, .tail = 0u, .count = 0u };
static osMutexId_t g_ring_mutex = NULL;
static osSemaphoreId_t g_flush_semaphore = NULL;
static osThreadId_t g_task_handle = NULL;
static bool g_ready = false;

static FIL g_log_file;
static bool g_log_file_open = false;
static FSIZE_t g_log_file_size = 0u;

static void debug_log_task_entry(void *argument);
static void ring_push_bytes(const char *data, size_t length);
static size_t ring_peek(uint8_t *dst, size_t capacity);
static void ring_drop(size_t length);
static void debug_log_flush(void);
static bool debug_log_open_file(void);
static bool debug_log_ensure_directory(void);
static bool debug_log_rotate_if_needed(FSIZE_t incoming_length);

bool debug_log_init(void) {
    if (g_ready) {
        return true;
    }

    g_ring_mutex = osMutexNew(NULL);
    g_flush_semaphore = osSemaphoreNew(10u, 0u, NULL);

    if ((g_ring_mutex == NULL) || (g_flush_semaphore == NULL)) {
        return false;
    }

    const osThreadAttr_t attr = {
        .name = "debug_log_task",
        .priority = (osPriority_t) DEBUG_LOG_TASK_PRIORITY,
        .stack_size = (uint32_t) (DEBUG_LOG_TASK_STACK_WORDS * sizeof(StackType_t)),
    };

    g_task_handle = osThreadNew(debug_log_task_entry, NULL, &attr);
    g_ready = (g_task_handle != NULL);

    return g_ready;
}

void debug_log_capture(const char *data, size_t length) {
    if ((!g_ready) || (data == NULL) || (length == 0u)) {
        return;
    }

    if (osMutexAcquire(g_ring_mutex, osWaitForever) == osOK) {
        ring_push_bytes(data, length);
        (void) osMutexRelease(g_ring_mutex);
    }

    if ((osKernelGetState() == osKernelRunning) && (g_flush_semaphore != NULL)) {
        (void) osSemaphoreRelease(g_flush_semaphore);
    }
}

static void debug_log_task_entry(void *argument) {
    (void) argument;

    for (;;) {
        (void) osSemaphoreAcquire(g_flush_semaphore, pdMS_TO_TICKS(DEBUG_LOG_FLUSH_PERIOD_MS));
        debug_log_flush();
    }
}

static void debug_log_flush(void) {
    if (osKernelGetState() != osKernelRunning) {
        return;
    }

    uint8_t chunk[DEBUG_LOG_WRITE_CHUNK_BYTES];
    bool wrote_any = false;

    while (1) {
        size_t to_copy = 0u;

        if (osMutexAcquire(g_ring_mutex, osWaitForever) == osOK) {
            to_copy = ring_peek(chunk, sizeof chunk);
            (void) osMutexRelease(g_ring_mutex);
        }

        if (to_copy == 0u) {
            break;
        }

        FS_LOCK();
        bool ok = debug_log_open_file();
        if (ok) {
            ok = debug_log_rotate_if_needed((FSIZE_t) to_copy);
        }

        if (ok) {
            UINT written = 0u;
            FRESULT fr = f_write(&g_log_file, chunk, (UINT) to_copy, &written);
            if ((fr == FR_OK) && (written == to_copy)) {
                g_log_file_size += written;
                wrote_any = true;
            } else {
                ok = false;
            }
        }
        FS_UNLOCK();

        if (!ok) {
            break; /* Preserve buffered data for a future retry. */
        }

        if (osMutexAcquire(g_ring_mutex, osWaitForever) == osOK) {
            ring_drop(to_copy);
            (void) osMutexRelease(g_ring_mutex);
        }
    }

    if (wrote_any) {
        FS_LOCK();
        (void) f_sync(&g_log_file);
        FS_UNLOCK();
    }
}

static bool debug_log_open_file(void) {
    if (g_log_file_open) {
        return true;
    }

    if (!debug_log_ensure_directory()) {
        return false;
    }

    FRESULT fr = f_open(&g_log_file, DEBUG_LOG_PATH, FA_OPEN_ALWAYS | FA_WRITE);
    if (fr != FR_OK) {
        return false;
    }

    fr = f_lseek(&g_log_file, f_size(&g_log_file));
    if (fr != FR_OK) {
        (void) f_close(&g_log_file);
        return false;
    }

    g_log_file_size = f_size(&g_log_file);
    g_log_file_open = true;

    return true;
}

static bool debug_log_ensure_directory(void) {
    FILINFO info;
    FRESULT fr = f_stat(DEBUG_LOG_DIR, &info);

    if (fr == FR_OK) {
        return ((info.fattrib & AM_DIR) != 0u);
    }

    if (fr == FR_NO_FILE) {
        fr = f_mkdir(DEBUG_LOG_DIR);
        return (fr == FR_OK);
    }

    return false;
}

static bool debug_log_rotate_if_needed(FSIZE_t incoming_length) {
    if (!g_log_file_open) {
        return false;
    }

    if ((g_log_file_size + incoming_length) <= DEBUG_LOG_MAX_BYTES) {
        return true;
    }

    (void) f_close(&g_log_file);
    g_log_file_open = false;

    (void) f_unlink(DEBUG_LOG_PREVIOUS_PATH);
    (void) f_rename(DEBUG_LOG_PATH, DEBUG_LOG_PREVIOUS_PATH);

    FRESULT fr = f_open(&g_log_file, DEBUG_LOG_PATH, FA_CREATE_ALWAYS | FA_WRITE);
    if (fr != FR_OK) {
        return false;
    }

    g_log_file_size = 0u;
    g_log_file_open = true;
    return true;
}

static void ring_push_bytes(const char *data, size_t length) {
    for (size_t i = 0u; i < length; ++i) {
        g_ring.buffer[g_ring.head] = (uint8_t) data[i];
        g_ring.head = (g_ring.head + 1u) % DEBUG_LOG_BUFFER_BYTES;

        if (g_ring.count < DEBUG_LOG_BUFFER_BYTES) {
            ++g_ring.count;
        } else {
            g_ring.tail = (g_ring.tail + 1u) % DEBUG_LOG_BUFFER_BYTES;
        }
    }
}

static size_t ring_peek(uint8_t *dst, size_t capacity) {
    if ((dst == NULL) || (capacity == 0u)) {
        return 0u;
    }

    size_t available = g_ring.count;
    if (available > capacity) {
        available = capacity;
    }

    for (size_t i = 0u; i < available; ++i) {
        size_t idx = (g_ring.tail + i) % DEBUG_LOG_BUFFER_BYTES;
        dst[i] = g_ring.buffer[idx];
    }

    return available;
}

static void ring_drop(size_t length) {
    if (length > g_ring.count) {
        length = g_ring.count;
    }

    g_ring.count -= length;
    g_ring.tail = (g_ring.tail + length) % DEBUG_LOG_BUFFER_BYTES;
}
