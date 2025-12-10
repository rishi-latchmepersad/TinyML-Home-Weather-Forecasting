#include "inference_logger_task.h" // Pull in the interface definition so this implementation stays in sync with the header.
// -----------------------------------------------------------------------------
#include <stdio.h> // Provide snprintf for formatting strings before writing them to disk.
#include <string.h> // Provide memcpy/strncpy for manipulating filenames and timestamps.
// -----------------------------------------------------------------------------
#include "FreeRTOS.h" // Include core FreeRTOS types needed for task management and delays.
#include "cmsis_os.h" // Access CMSIS-RTOS abstraction used for mutexes and thread creation.
#include "ds3231.h" // Allow reading timestamps from the DS3231 real-time clock.
#include "diskio.h" // Provide low-level disk synchronization primitives for the SD card.
#include "ff.h" // Include FatFS file system APIs for interacting with the SD card.
#include "forecast_temp_task.h" // Retrieve the latest predicted temperature measurements.
#include "led_service.h" // Control the LED to indicate successful logging activity.
#include "sd_card.h" // Use helper functions for mounting the SD card before file access.
#include "task.h" // Access FreeRTOS task utilities such as vTaskDelayUntil.
// -----------------------------------------------------------------------------
#define LOG_PREFIX "[INFERENCE_LOGGER] "
// -----------------------------------------------------------------------------
#ifndef INFERENCE_LOGGER_TASK_STACK_WORDS // Allow overriding the stack allocation for the logger task at build time.
#define INFERENCE_LOGGER_TASK_STACK_WORDS   (2048u)
#endif // End of default stack size guard.
#ifndef INFERENCE_LOGGER_TASK_PRIORITY // Allow the application to override the logger task priority.
#define INFERENCE_LOGGER_TASK_PRIORITY      (osPriorityLow) // Run the logger at low priority so it does not block critical work.
#endif // End of default priority guard.
#ifndef INFERENCE_LOGGER_BUFFER_BYTES // Permit build-time customization of the write buffer size.
#define INFERENCE_LOGGER_BUFFER_BYTES       (2048u) // Buffer writes to limit SD card access frequency while staying memory conscious.
#endif // End of default buffer size guard.
#ifndef INFERENCE_LOGGER_BASE_DIR // Let integrators change the log directory without modifying code.
#define INFERENCE_LOGGER_BASE_DIR           "0:/logs" // Store logs on the SD card in the /logs directory for organization.
#endif // End of default base directory guard.
#ifndef INFERENCE_LOGGER_BACKOFF_MS // Allow configuring how long to wait after a failure before retrying.
#define INFERENCE_LOGGER_BACKOFF_MS         (500u) // Wait half a second to avoid hammering the SD card when errors occur.
#endif // End of default error backoff guard.
#ifndef INFERENCE_LOGGER_PERIOD_MS // Allow customizing how often we poll for new inference data.
#define INFERENCE_LOGGER_PERIOD_MS          (300000u) // Default to five-minute logging intervals to balance detail and storage.
#endif // End of default logging period guard.
// -----------------------------------------------------------------------------
extern osMutexId_t g_fs_mutex; // Reference the shared filesystem mutex declared elsewhere so we can synchronize access.
#ifndef FS_LOCK // Allow overriding the lock macros if the platform provides specialized locking.
#define FS_LOCK()   osMutexAcquire(g_fs_mutex, osWaitForever) // Acquire the filesystem mutex, blocking until it is available to ensure exclusive access.
#define FS_UNLOCK() osMutexRelease(g_fs_mutex) // Release the filesystem mutex so other tasks can interact with the SD card.
#endif // End of lock macro override guard.
// -----------------------------------------------------------------------------
static void inference_logger_task_entry(void *argument); // Forward declaration of the task entry function used when creating the thread.
static bool inference_logger_ensure_directory_exists(const char *dir_path); // Forward declaration for ensuring the log directory exists before writing.
static bool inference_logger_open_today_file(const char *date_yyyy_mm_dd); // Forward declaration for opening the daily log file based on the current date.
static bool inference_logger_flush_buffer_to_file(void); // Forward declaration for writing the buffered log data out to the SD card.
static bool inference_logger_write_header(FIL *file); // Forward declaration for emitting the CSV header with lead-time columns.
static size_t inference_logger_format_csv_line(const char *timestamp_iso8601, // Forward declaration for formatting an individual CSV log line.
                                               const float *predicted_temperatures_c, // Parameter describing the predicted temperature vector we want to store.
                                               size_t prediction_count, // Number of forecast slots carried in the vector.
                                               uint32_t inference_time_ms, // Duration of the inference in milliseconds.
                                               char *out_line, // Pointer to the destination buffer where the formatted line will be written.
                                               size_t out_capacity); // Capacity of the destination buffer to prevent overruns.
// -----------------------------------------------------------------------------
static TaskHandle_t g_inference_logger_task_handle = NULL; // Store the created task handle so we can detect if the task already exists.
static FIL g_active_file; // FatFS file object representing the currently open log file.
static bool g_file_open = false; // Track whether the active file is currently open to avoid redundant opens.
static char g_last_date_yyyy_mm_dd[11] = {0}; // Keep the last date string so we know when to rotate log files.
static char g_active_path[128] = {0}; // Store the path of the active log file for reopening and flushing operations.
#if defined(__GNUC__) // Apply additional alignment attributes when compiling with GCC to meet DMA/cache requirements.
__attribute__((aligned(32))) // Align the buffer to 32 bytes to keep cache lines consistent for SD card DMA.
#endif // Close GCC-specific alignment guard.
__attribute__((section(".sram1"), aligned(32))) // Place the buffer in SRAM1 with alignment suitable for peripheral access reliability.
static char g_write_buffer[INFERENCE_LOGGER_BUFFER_BYTES]; // Allocate the write buffer that batches log lines before flushing.
static size_t g_write_used = 0u; // Track how much of the write buffer is currently filled so we know when to flush.
// -----------------------------------------------------------------------------
typedef enum { // Enumerate the possible states of the logger state machine for clarity and maintainability.
    INFERENCE_LOGGER_STATE_STARTUP = 0, // Initial state where we attempt to mount the SD card.
    INFERENCE_LOGGER_STATE_ENSURE_DIR, // State that verifies the log directory exists before writing.
    INFERENCE_LOGGER_STATE_OPEN_TODAY, // State responsible for opening or creating today's log file.
    INFERENCE_LOGGER_STATE_RUNNING, // Normal operating state where we poll for new data and queue writes.
    INFERENCE_LOGGER_STATE_FLUSH, // State that flushes buffered data to persistent storage.
    INFERENCE_LOGGER_STATE_ROTATE, // State that closes the current file and opens a new one when the date changes.
    INFERENCE_LOGGER_STATE_ERROR_BACKOFF // State that delays after encountering an error before retrying.
} inference_logger_state_t; // Typedef for convenient use of the logger state enumeration.
// -----------------------------------------------------------------------------
bool inference_logger_task_create(void) { // Public API used to start the inference logger task.
    if (g_inference_logger_task_handle != NULL) { // Check if the task is already running to avoid creating duplicates.
        return true; // Task already exists, so treat creation as successful.
    } // End check for existing task.
// -----------------------------------------------------------------------------
    const osThreadAttr_t attr = { // Define the thread attributes used when creating the logger task.
        .name = "inference_logger_task", // Give the task a descriptive name for debugging and tracing.
        .priority = (osPriority_t) INFERENCE_LOGGER_TASK_PRIORITY, // Assign the configured priority to control scheduling.
        .stack_size = (uint32_t)(INFERENCE_LOGGER_TASK_STACK_WORDS * sizeof(StackType_t)), // Allocate the configured stack size in bytes to ensure sufficient room.
    }; // End of thread attribute initializer.
    g_inference_logger_task_handle = // Store the resulting task handle for future reference and duplicate detection.
        osThreadNew(inference_logger_task_entry, NULL, &attr); // Create the task with no arguments using the defined attributes.
    return (g_inference_logger_task_handle != NULL); // Return whether task creation succeeded so callers can react accordingly.
} // End of inference_logger_task_create.
// -----------------------------------------------------------------------------
static void inference_logger_task_entry(void *argument) { // Main FreeRTOS task function that manages the logging state machine.
    (void)argument; // Explicitly ignore the unused argument to avoid compiler warnings.
    inference_logger_state_t state = INFERENCE_LOGGER_STATE_STARTUP; // Begin in the startup state so we mount the SD card first.
    static TickType_t s_next_poll_tick = 0; // Track the next tick count when we should poll for new data to maintain consistent intervals.
    static bool s_poll_initialized = false; // Remember whether the periodic delay scheduler has been initialized to avoid invalid delays.
// -----------------------------------------------------------------------------
    while(1) { // Run indefinitely so logging continues throughout device operation.
        switch (state) { // Execute behavior based on the current state of the logger state machine.
        case INFERENCE_LOGGER_STATE_STARTUP: // Handle the initial attempt to mount storage.
            if (SD_Mount() == FR_OK) { // Try to mount the SD card and proceed if successful.
                state = INFERENCE_LOGGER_STATE_ENSURE_DIR; // Once mounted, ensure the logging directory is available before writing.
            } else { // Mount failed.
                state = INFERENCE_LOGGER_STATE_ERROR_BACKOFF; // Transition to error handling to delay and retry later.
            } // End mount check.
            break; // Exit the switch case to process the new state.
// -----------------------------------------------------------------------------
        case INFERENCE_LOGGER_STATE_ENSURE_DIR: // Ensure the log directory exists before we attempt to open files.
            if (inference_logger_ensure_directory_exists(INFERENCE_LOGGER_BASE_DIR)) { // Attempt to create or validate the base log directory.
                state = INFERENCE_LOGGER_STATE_OPEN_TODAY; // Proceed to open today's log once the directory is ready.
            } else { // Directory check failed.
                state = INFERENCE_LOGGER_STATE_ERROR_BACKOFF; // Retry later after a backoff to handle transient filesystem issues.
            } // End directory check.
            break; // Exit the switch case after handling the directory state.
// -----------------------------------------------------------------------------
        case INFERENCE_LOGGER_STATE_OPEN_TODAY: { // Open or create the log file corresponding to the current date.
            char ts[24]; // Buffer to hold the current timestamp in ISO-8601 format.
            if (ds3231_read_time_iso8601_utc_i2c1(ts, sizeof ts) != HAL_OK) { // Attempt to read the current time from the RTC.
                (void)snprintf(ts, sizeof ts, "2000-01-01T00:00:00Z"); // Fall back to a safe default timestamp if the RTC read fails.
            } // End RTC read handling.
            char date[11]; // Buffer to hold just the date component extracted from the timestamp.
            memcpy(date, ts, 10); // Copy the YYYY-MM-DD characters from the timestamp into the date buffer.
            date[10] = '\0'; // Null-terminate the date string so standard string functions can use it.
// -----------------------------------------------------------------------------
            if (inference_logger_open_today_file(date)) { // Try to open or create the day's log file using the formatted date.
                s_poll_initialized = false; // Reset the polling scheduler so we start a fresh interval after opening the file.
                state = INFERENCE_LOGGER_STATE_RUNNING; // Move into the running state to begin periodic logging.
            } else { // File open failed.
                state = INFERENCE_LOGGER_STATE_ERROR_BACKOFF; // Enter backoff to retry file operations later.
            } // End file open check.
            break; // Exit the block scope for the OPEN_TODAY state.
        } // Close scoped variables for OPEN_TODAY.
// -----------------------------------------------------------------------------
        case INFERENCE_LOGGER_STATE_RUNNING: { // Normal operation where we periodically record predictions.
            if (!s_poll_initialized) { // If we have not yet set the initial poll time since entering running state.
                s_next_poll_tick = xTaskGetTickCount(); // Initialize the next poll tick to the current tick count.
                s_poll_initialized = true; // Mark initialization complete so subsequent iterations delay relative to this timestamp.
            } else { // Polling has already been initialized.
                vTaskDelayUntil(&s_next_poll_tick, // Delay until the next scheduled poll tick to maintain consistent intervals.
                                pdMS_TO_TICKS(INFERENCE_LOGGER_PERIOD_MS)); // Convert the configured period to ticks for the delay call.
            } // End polling initialization check.
// -----------------------------------------------------------------------------
            char ts_now[24]; // Buffer for the current timestamp string to accompany the logged data.
            if (ds3231_read_time_iso8601_utc_i2c1(ts_now, sizeof ts_now) != HAL_OK) { // Attempt to read the current time for the log entry.
                (void)snprintf(ts_now, sizeof ts_now, "2000-01-01T00:00:00Z"); // Use a fallback timestamp if we cannot read the RTC.
            } // End current time retrieval.
            char date_now[11]; // Buffer to store the current date extracted from the timestamp.
            memcpy(date_now, ts_now, 10); // Copy the date portion from the timestamp for comparison with the active file date.
            date_now[10] = '\0'; // Null-terminate the date string for safe string operations.
// -----------------------------------------------------------------------------
            if (strncmp(date_now, g_last_date_yyyy_mm_dd, 10) != 0) { // Check if the date has changed since we opened the file.
                state = INFERENCE_LOGGER_STATE_ROTATE; // If the date changed, rotate to a new file for the new day.
                break; // Leave the RUNNING case so the state machine processes the rotation.
            } // End date change check.
// -----------------------------------------------------------------------------
            float predicted_temperatures_c[FORECAST_TEMP_FORECAST_HORIZON_SLOTS] = {0.0f}; // Buffer to hold the latest predicted temperature vector.
            uint32_t inference_time_ms = 0u; // Track the runtime of the most recent inference.
            if (forecast_temp_get_latest_prediction(predicted_temperatures_c, FORECAST_TEMP_FORECAST_HORIZON_SLOTS, &inference_time_ms)) { // Attempt to retrieve a new prediction from the forecasting module.
                const double first_slot = (FORECAST_TEMP_FORECAST_HORIZON_SLOTS > 0u) ? (double)predicted_temperatures_c[0] : 0.0;
                printf(LOG_PREFIX "We successfully got a %lu-slot prediction vector. First slot: %.2f C\n",
                       (unsigned long)FORECAST_TEMP_FORECAST_HORIZON_SLOTS,
                       first_slot);
                char line[512]; // Buffer to store the formatted CSV line before writing to the main buffer.
                size_t len = // Variable capturing the number of bytes in the formatted log entry.
                    inference_logger_format_csv_line(ts_now, predicted_temperatures_c,
                                                      FORECAST_TEMP_FORECAST_HORIZON_SLOTS,
                                                      inference_time_ms,
                                                      line, sizeof line); // Provide the destination buffer and its capacity to prevent overflow.
                if (len > 0u) { // Ensure the formatted line is valid before attempting to buffer it.
                    if (len > sizeof g_write_buffer) { // Refuse to log entries that can never fit in the buffer.
                        printf(LOG_PREFIX "Formatted inference line (%lu bytes) exceeds buffer (%lu bytes); dropping entry\r\n",
                               (unsigned long)len,
                               (unsigned long)sizeof g_write_buffer);
                        state = INFERENCE_LOGGER_STATE_ERROR_BACKOFF; // Retry later to avoid spinning forever.
                        break; // Abort this iteration because the data cannot be buffered safely.
                    }
                    if ((g_write_used + len) > sizeof g_write_buffer) { // Check if adding the line would exceed the buffer capacity.
                        if (!inference_logger_flush_buffer_to_file()) { // Flush current data before appending to prevent loss.
                            state = INFERENCE_LOGGER_STATE_ERROR_BACKOFF; // Retry after a backoff when the flush fails.
                            break; // Leave the RUNNING state to honor the backoff delay.
                        }
                    }
                    memcpy(&g_write_buffer[g_write_used], line, len); // Copy the formatted line into the write buffer at the current end.
                    g_write_used += len; // Update the number of bytes used to include the new line.
                    state = INFERENCE_LOGGER_STATE_FLUSH; // Trigger a flush so the newly added data is persisted promptly.
                    break; // Exit the RUNNING state to allow the flush to occur immediately.
                } // End formatted line validity check.
            } // End prediction availability check.
            else{
            	printf(LOG_PREFIX "We couldn't get a prediction. \n");
            }
            break; // Continue looping in RUNNING state when no flush is triggered.
        } // Close scoped variables for RUNNING state.
// -----------------------------------------------------------------------------
        case INFERENCE_LOGGER_STATE_FLUSH: { // Handle writing buffered data to the SD card.
            const size_t before = g_write_used; // Snapshot how much data is currently queued to decide whether we need to write.
            if (before == 0u) { // If there is nothing buffered, no flush is required.
                state = INFERENCE_LOGGER_STATE_RUNNING; // Return to the running state to continue polling for data.
                break; // Exit the FLUSH state early because there is nothing to do.
            } // End check for empty buffer.
// -----------------------------------------------------------------------------
            if (inference_logger_flush_buffer_to_file()) { // Attempt to write the buffered data out to the SD card.
                state = INFERENCE_LOGGER_STATE_RUNNING; // On success, return to running so we can gather more data.
            } else { // Flush failed.
                state = INFERENCE_LOGGER_STATE_ERROR_BACKOFF; // Enter backoff to handle filesystem errors gracefully.
            } // End flush result check.
            break; // Finish handling the FLUSH state.
        } // Close scoped variables for FLUSH state.
// -----------------------------------------------------------------------------
        case INFERENCE_LOGGER_STATE_ROTATE: // Handle closing the current log and preparing for a new day's file.
            if (g_file_open) { // Only attempt to close the file if it is currently open.
                (void)inference_logger_flush_buffer_to_file(); // Flush any remaining data before closing to avoid data loss.
                FS_LOCK(); // Acquire the filesystem lock to safely close the file.
                (void)f_close(&g_active_file); // Close the current log file using FatFS API.
                FS_UNLOCK(); // Release the filesystem lock so other operations can proceed.
                g_file_open = false; // Mark the file as closed so future operations know to reopen it.
            } // End file open check during rotation.
            state = INFERENCE_LOGGER_STATE_OPEN_TODAY; // Transition back to opening today's file (which will now be the new date).
            break; // Exit the ROTATE state after scheduling the reopen.
// -----------------------------------------------------------------------------
        case INFERENCE_LOGGER_STATE_ERROR_BACKOFF: // Handle any errors by tearing down state and retrying later.
            if (g_file_open) { // If a file is open when an error occurs, close it to avoid corruption.
                FS_LOCK(); // Acquire the filesystem mutex before closing the file.
                (void)f_close(&g_active_file); // Close the active file safely.
                FS_UNLOCK(); // Release the mutex once the close is complete.
                g_file_open = false; // Mark that no file is open so future retries know to reopen it.
            } // End file closure in error state.
            vTaskDelay(pdMS_TO_TICKS(INFERENCE_LOGGER_BACKOFF_MS)); // Wait for the configured backoff period to avoid rapid retries.
            state = INFERENCE_LOGGER_STATE_STARTUP; // Reset the state machine to attempt mounting and setup again.
            break; // Finish handling the error backoff state.
        } // End switch cases for the state machine.
    } // End infinite loop.
} // End of inference_logger_task_entry implementation.
// -----------------------------------------------------------------------------
static bool inference_logger_flush_buffer_to_file(void) { // Helper that writes the buffered log data to disk and resets the buffer.
    const size_t bytes_to_write = g_write_used; // Capture how many bytes are pending so we write exactly that amount.
// -----------------------------------------------------------------------------
    if (SD_Mount() != FR_OK) { // Ensure the SD card is mounted before touching FatFS.
        printf(LOG_PREFIX "SD_Mount failed while flushing inference log\r\n");
        return false; // Signal failure so the caller retries later.
    }
// -----------------------------------------------------------------------------
    if (g_active_path[0] == '\0') { // Refuse to flush when we do not know the active file path.
        printf(LOG_PREFIX "No active inference log path set; skipping flush\r\n");
        return false;
    }
// -----------------------------------------------------------------------------
    if (bytes_to_write == 0u) { // If no data is queued, we consider the flush successful without performing I/O.
        return true; // Nothing to do, so report success.
    } // End empty buffer check.
// -----------------------------------------------------------------------------
    if (!g_file_open) { // If the file is not currently open, open it before writing.
        FS_LOCK(); // Acquire the filesystem mutex to protect the FatFS calls.
        FRESULT fr = f_open(&g_active_file, g_active_path, FA_OPEN_ALWAYS | FA_WRITE); // Try opening the active file, creating it if necessary.
        bool header_ok = true; // Track whether header emission succeeded when needed.
        if (fr == FR_OK) { // Only proceed if the file opened successfully.
            if (f_size(&g_active_file) == 0) { // If the file is new and empty, we need to add a header row.
                header_ok = inference_logger_write_header(&g_active_file); // Emit the header that labels each forecast horizon.
                if (header_ok) { // When successful, push the header to the storage media.
                    (void)disk_ioctl(0, CTRL_SYNC, NULL); // Force the physical disk to commit data to reduce risk of loss on power failure.
                }
            } // End check for empty file requiring header.
            if (header_ok) { // Only seek when the header operation succeeded.
                fr = f_lseek(&g_active_file, f_size(&g_active_file)); // Seek to the end of the file so new entries append correctly.
                if (fr == FR_OK) {
                    g_file_open = true; // Record that the file is now open for future flushes.
                }
            } else {
                (void)f_close(&g_active_file); // Close the file so the caller can retry later.
                fr = FR_INT_ERR; // Force an error so the caller enters the backoff path.
            }
        } // End open success check.
        FS_UNLOCK(); // Release the filesystem lock regardless of success so others are not blocked.
// -----------------------------------------------------------------------------
        if (!g_file_open) { // If the open operation failed, we cannot flush data.
            return false; // Signal failure so the caller can enter backoff.
        } // End open failure handling.
    } // End check for unopened file.
// -----------------------------------------------------------------------------
    UINT bw = 0u; // Prepare a variable to capture the number of bytes written by FatFS.
    FS_LOCK(); // Acquire the filesystem mutex before writing to the file.
    FRESULT wr = f_write(&g_active_file, g_write_buffer, (UINT)bytes_to_write, &bw); // Write the buffered data to the file.
    if (wr == FR_OK && bw == bytes_to_write) { // If the write succeeded and wrote the expected number of bytes.
        wr = f_sync(&g_active_file); // Flush the file to ensure the data reaches non-volatile storage.
    } // End successful write check.
    FS_UNLOCK(); // Release the filesystem lock so other tasks may access the SD card.
// -----------------------------------------------------------------------------
    if (wr != FR_OK || bw != bytes_to_write) { // Verify the write and sync both succeeded.
        printf(LOG_PREFIX "Failed to write %u/%u bytes to inference log %s wr=%d\r\n",
               (unsigned)bw,
               (unsigned)bytes_to_write,
               g_active_path,
               (int)wr);
        FS_LOCK(); // Ensure the file handle is closed on failure so we do not leak descriptors.
        (void)f_close(&g_active_file);
        FS_UNLOCK();
        g_file_open = false; // Mark the file as closed so a future attempt will reopen cleanly.
        return false; // On failure, report back so the caller can handle the error.
    } // End write verification.
// -----------------------------------------------------------------------------
    FS_LOCK(); // Release the handle after a successful flush to keep descriptor usage low.
    (void)f_close(&g_active_file);
    FS_UNLOCK();
    g_file_open = false; // Mark the file as closed; the next flush will reopen as needed.
    led_service_activity_bump(1000); // Trigger LED activity feedback for one second to indicate a successful log flush.
    printf(LOG_PREFIX "Flushed %u bytes to inference log %s\r\n", (unsigned)bw, g_active_path);
    g_write_used = 0u; // Reset the buffered byte count so we can accumulate new data.
    return true; // Indicate that flushing succeeded.
} // End of inference_logger_flush_buffer_to_file helper.
// -----------------------------------------------------------------------------
static bool inference_logger_write_header(FIL *file) { // Helper that writes the CSV header with all forecast lead times.
    if (file == NULL) { // Guard against a NULL file pointer.
        return false;
    }
    char header[1024]; // Buffer to assemble the header line with all horizon columns.
    size_t used = 0u; // Track how many bytes of the buffer have been populated so far.
    int written = snprintf(header, sizeof header, "timestamp_iso8601,inference_time_ms"); // Seed the header with the timestamp and inference duration columns.
    if (written <= 0 || (size_t)written >= sizeof header) { // Validate the snprintf result for errors or truncation.
        return false;
    }
    used = (size_t)written; // Update the number of bytes consumed.
    for (size_t i = 0u; i < FORECAST_TEMP_FORECAST_HORIZON_SLOTS; ++i) { // Append one column per forecast slot.
        const uint32_t lead_minutes_total = (uint32_t)((i + 1u) * FORECAST_TEMP_MINUTES_PER_SLOT); // Convert slot index to minutes ahead.
        const uint32_t lead_hours = lead_minutes_total / 60u; // Derive the hour portion of the lead time.
        const uint32_t lead_minutes = lead_minutes_total % 60u; // Derive the remaining minutes beyond whole hours.
        written = snprintf(&header[used], sizeof header - used,
                           ",predicted_temperature_c_plus_%02luh%02lum",
                           (unsigned long)lead_hours,
                           (unsigned long)lead_minutes); // Append the column label with the corresponding lead time.
        if (written <= 0 || (size_t)written >= (sizeof header - used)) { // Abort on formatting errors or truncation.
            return false;
        }
        used += (size_t)written; // Consume the newly written bytes.
    }
    if ((used + 2u) >= sizeof header) { // Ensure there is room for CRLF terminator.
        return false;
    }
    header[used++] = '\r'; // Append carriage return to terminate the CSV header line.
    header[used++] = '\n'; // Append newline to complete the header line.
    header[used] = '\0'; // Null-terminate for safety when debugging.
    UINT bw = 0u; // Capture the number of bytes written by FatFS.
    FRESULT fr = f_write(file, header, (UINT)used, &bw); // Emit the header to the file.
    if ((fr != FR_OK) || (bw != (UINT)used)) { // Validate the write operation.
        return false;
    }
    fr = f_sync(file); // Flush the header so it is not lost on power failure.
    return (fr == FR_OK); // Report whether flushing succeeded.
} // End of inference_logger_write_header helper.
// -----------------------------------------------------------------------------
static bool inference_logger_ensure_directory_exists(const char *dir_path) { // Ensure the specified logging directory is present on the filesystem.
    FILINFO fi; // Structure used by FatFS to receive file/directory information.
    FRESULT fr; // Variable to capture FatFS return codes for error handling.
// -----------------------------------------------------------------------------
    FS_LOCK(); // Acquire the filesystem mutex so the directory check is thread safe.
    fr = f_stat(dir_path, &fi); // Query the filesystem to see if the directory already exists.
    FS_UNLOCK(); // Release the mutex once the check is complete.
// -----------------------------------------------------------------------------
    if (fr == FR_OK) { // If f_stat succeeded, the path exists.
        return ((fi.fattrib & AM_DIR) != 0); // Return true only if the existing path is a directory rather than a file.
    } // End existing path handling.
// -----------------------------------------------------------------------------
    if (fr == FR_NO_FILE) { // If the path does not exist, we need to create it.
        FS_LOCK(); // Acquire the filesystem mutex before creating the directory.
        fr = f_mkdir(dir_path); // Attempt to create the directory on the SD card.
        FS_UNLOCK(); // Release the mutex after the mkdir attempt.
        return (fr == FR_OK || fr == FR_EXIST); // Treat both successful creation and "already exists" as success to handle race conditions.
    } // End missing directory handling.
// -----------------------------------------------------------------------------
    printf(LOG_PREFIX "Failed to stat log directory %s fr=%d\r\n", dir_path, (int)fr);
    return false; // For any other error, report failure so the caller can back off and retry.
} // End of inference_logger_ensure_directory_exists helper.
// -----------------------------------------------------------------------------
static bool inference_logger_open_today_file(const char *date_yyyy_mm_dd) { // Open (or create) the log file for the provided date string.
    char path[128]; // Buffer to build the fully qualified file path.
    (void)snprintf(path, sizeof path, // Format the path string safely using snprintf to avoid buffer overflows.
                   INFERENCE_LOGGER_BASE_DIR "/inference_%s.csv", date_yyyy_mm_dd); // Combine the base directory and date into the file name.
// -----------------------------------------------------------------------------
    FILINFO fi; // Structure to hold file information when checking if the file already exists.
    FRESULT fr; // Variable to store FatFS return codes for control flow decisions.
// -----------------------------------------------------------------------------
    FS_LOCK(); // Acquire the filesystem mutex before interacting with FatFS.
    fr = f_stat(path, &fi); // Check whether the file already exists so we can decide whether to append or create.
    FS_UNLOCK(); // Release the filesystem mutex after the status check.
// -----------------------------------------------------------------------------
    if (fr == FR_OK) { // File already exists, so open it for appending.
        FS_LOCK(); // Acquire the filesystem mutex before opening the file.
        fr = f_open(&g_active_file, path, FA_OPEN_EXISTING | FA_WRITE); // Open the existing file with write access to append data.
        if (fr == FR_OK) { // If the file opened successfully, move the pointer to the end for appending.
            fr = f_lseek(&g_active_file, f_size(&g_active_file)); // Seek to the current end-of-file so new entries append correctly.
        } // End open success handling.
        FS_UNLOCK(); // Release the mutex after attempting to open and seek.
        if (fr != FR_OK) { // If either opening or seeking failed, signal failure.
            printf(LOG_PREFIX "Failed to reopen inference log %s fr=%d\r\n", path, (int)fr);
            return false; // Return false so the state machine can back off.
        } // End failure handling for existing file.
    } else if (fr == FR_NO_FILE) { // File does not exist, so create a new one with a header.
        FS_LOCK(); // Acquire the filesystem mutex before creating the file.
        fr = f_open(&g_active_file, path, FA_CREATE_NEW | FA_WRITE); // Create a new file ready for writing.
        if (fr == FR_OK) { // If creation succeeded, write the CSV header.
            const bool header_ok = inference_logger_write_header(&g_active_file); // Emit the header row with lead-time columns.
            (void)f_close(&g_active_file); // Close the file to finalize creation before reopening for appending.
            if (!header_ok) { // Abort when header generation fails.
                FS_UNLOCK(); // Release the mutex before returning so callers can retry later.
                return false;
            }
            (void)disk_ioctl(0, CTRL_SYNC, NULL); // Issue a disk sync command to push the header to the SD card.
        }
        FS_UNLOCK(); // Release the filesystem mutex after file creation.
        if (fr != FR_OK) { // If creation or header write failed, indicate failure.
            printf(LOG_PREFIX "Failed to create inference log %s fr=%d\r\n", path, (int)fr);
            return false; // Abort opening so the caller can handle the error.
        } // End failure handling after creation.
// -----------------------------------------------------------------------------
        FS_LOCK(); // Reacquire the filesystem mutex to reopen the newly created file for appending.
        fr = f_open(&g_active_file, path, FA_OPEN_EXISTING | FA_WRITE); // Open the file we just created so we can append new entries.
        if (fr == FR_OK) { // If opening succeeded, seek to the end to append data.
            fr = f_lseek(&g_active_file, f_size(&g_active_file)); // Position the file pointer at the end for consistent append behavior.
        } // End reopen success handling.
        FS_UNLOCK(); // Release the filesystem mutex after reopening and seeking.
        if (fr != FR_OK) { // If reopening failed, report failure.
            printf(LOG_PREFIX "Failed to reopen freshly created inference log %s fr=%d\r\n", path, (int)fr);
            return false; // Inform the caller to handle the error.
        } // End failure handling for reopened file.
    } else { // Some other error occurred while checking the file status.
        printf(LOG_PREFIX "Unable to stat inference log %s fr=%d\r\n", path, (int)fr);
        return false; // Propagate failure so the caller can enter the error state.
    } // End file existence handling.
// -----------------------------------------------------------------------------
    g_file_open = true; // Record that the file is currently open for subsequent operations.
    strncpy(g_last_date_yyyy_mm_dd, date_yyyy_mm_dd, // Update the cached date string so we know when to rotate files.
            sizeof g_last_date_yyyy_mm_dd); // Limit the copy to the buffer size to avoid overflow.
    g_last_date_yyyy_mm_dd[sizeof g_last_date_yyyy_mm_dd - 1] = '\0'; // Ensure the cached date string is null-terminated for safe comparisons.
    strncpy(g_active_path, path, sizeof g_active_path); // Remember the path so we can reopen or reference it later as needed.
    g_active_path[sizeof g_active_path - 1] = '\0'; // Guarantee the active path string is null-terminated.
    g_write_used = 0u; // Reset the buffered data count since we start fresh after opening a file.
// -----------------------------------------------------------------------------
    FS_LOCK(); // Close the file immediately so the logger does not hold an extra descriptor between flushes.
    (void)f_close(&g_active_file);
    FS_UNLOCK();
    g_file_open = false; // Mark the file as closed; the flush routine will reopen it when needed.
    return true; // Indicate that the file is ready for logging.
} // End of inference_logger_open_today_file helper.
// -----------------------------------------------------------------------------
static size_t inference_logger_format_csv_line(const char *timestamp_iso8601, // Format a log line from the provided timestamp and prediction vector.
                                               const float *predicted_temperatures_c, // Temperature predictions to include in the log line.
                                               size_t prediction_count, // Number of predictions contained in the vector.
                                               uint32_t inference_time_ms, // Duration of the inference in milliseconds.
                                               char *out_line, // Buffer where the formatted CSV line should be placed.
                                               size_t out_capacity) { // Size of the destination buffer to prevent overruns.
    if ((out_line == NULL) || (out_capacity == 0u) || (predicted_temperatures_c == NULL)) { // Validate inputs before formatting.
        return 0u;
    }
    const char *timestamp = (timestamp_iso8601 != NULL) ? timestamp_iso8601 : "TIME?"; // Provide a fallback label for missing timestamps.
    int n = snprintf(out_line, out_capacity, "%s,%lu", timestamp, (unsigned long)inference_time_ms); // Start the line with the timestamp and inference duration columns.
    if (n <= 0 || (size_t) n >= out_capacity) { // Abort when snprintf fails or truncates the output.
        return 0u;
    }
    size_t used = (size_t) n; // Track how many bytes were consumed so far.
    for (size_t i = 0u; i < prediction_count; ++i) { // Append each prediction to the CSV row.
        if (used >= out_capacity) { // Abort if the buffer is already full.
            return 0u;
        }
        n = snprintf(&out_line[used], out_capacity - used, ",%.6f", (double) predicted_temperatures_c[i]); // Append the prediction with six decimal places.
        if (n <= 0 || (size_t) n >= (out_capacity - used)) { // Abort on formatting errors or truncation.
            return 0u;
        }
        used += (size_t) n; // Consume the newly written bytes before appending the next value.
    }
    if ((used + 2u) >= out_capacity) { // Ensure there is room for CRLF and a terminating null character.
        return 0u;
    }
    out_line[used++] = '\r'; // Append carriage return to terminate the CSV row.
    out_line[used++] = '\n'; // Append newline to complete the row.
    out_line[used] = '\0'; // Null-terminate for safety even though the caller uses the returned length.
    return used; // Return the number of bytes written so the caller can update buffer usage.
} // End of inference_logger_format_csv_line helper.
// -----------------------------------------------------------------------------
