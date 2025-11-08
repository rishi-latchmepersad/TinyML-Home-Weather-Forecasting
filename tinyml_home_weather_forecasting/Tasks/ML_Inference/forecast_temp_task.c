// Pull in our own header so function prototypes and this implementation stay aligned.
#include "forecast_temp_task.h"
// Pull in the FreeRTOS kernel API for timing and task primitives.
#include "FreeRTOS.h"
// Pull in the FreeRTOS task API for vTaskDelayUntil().
#include "task.h"
// Pull in the math library so we can calculate sine waves for the hour feature.
#include <math.h>
// Pull in string helpers for memset and friends.
#include <string.h>
// Pull in stdio so we can log simple diagnostic messages.
#include <stdio.h>
// Pull in ctype so we can validate digits when parsing the RTC timestamp.
#include <ctype.h>
// Pull in stdint so we can handle quantization zero-points with explicit widths.
#include <stdint.h>
// Pull in our BME280 task interface to read the latest environmental sample.
#include "../Sensors/bme280_task.h"
// Pull in our VEML7700 interface to grab illuminance readings.
#include "../Sensors/veml7700_task.h"
// Pull in the DS3231 driver so we can ask the RTC for the current time.
#include "../../Drivers/DS3231/Inc/ds3231.h"
// Pull in the FatFs declarations so we can read persisted CSV logs.
#include "ff.h"
// Pull in the auto-generated network API so we can run inference.
#include "forecast_temp_ml_model.h"
// Pull in the data helpers so we can allocate the activation buffers correctly.
#include "forecast_temp_ml_model_data.h"
// Pull in the data parameter helpers so we can set up the activation table.
#include "forecast_temp_ml_model_data_params.h"
// Pull in stdlib so we can convert CSV strings into floats.
#include <stdlib.h>

// Provide a fallback for M_PI when the math library does not define it.
#ifndef M_PI
// Define M_PI as a float literal so our sine calculation has a sensible value.
#define M_PI 3.14159265358979323846f
#endif

// Declare the filesystem mutex so we can coordinate with the logger task.
extern osMutexId_t g_fs_mutex;
// Provide a helper macro that acquires the filesystem mutex before touching FatFs.
#define FS_LOCK()   osMutexAcquire(g_fs_mutex, osWaitForever)
// Provide a helper macro that releases the filesystem mutex after a FatFs call.
#define FS_UNLOCK() osMutexRelease(g_fs_mutex)

// Declare how many 32-bit words the forecasting task stack should use.
#define FORECAST_TEMP_TASK_STACK_WORDS      (1024u)
// Declare the CMSIS-RTOS priority the forecasting task should run at.
#define FORECAST_TEMP_TASK_PRIORITY         (osPriorityLow)
// Declare how often (in milliseconds) we poll the sensors for a new reading.
#define FORECAST_TEMP_TASK_PERIOD_MS        (60000u)
// Declare how many hourly samples we keep inside the sliding input window.
#define FORECAST_TEMP_WINDOW_LENGTH         (1u)
// Declare how many minute samples we fold into a single hourly aggregate.
#define FORECAST_TEMP_MINUTES_PER_HOUR      (60u)
// Declare how many hours back we look when computing the temperature delta.
#define FORECAST_TEMP_DELTA_T_LAG_HOURS     (1u)
// Declare how many hours back we look when computing the pressure delta.
#define FORECAST_TEMP_DELTA_P_LAG_HOURS     (6u)
// Declare how many engineered plus raw features the model expects per time step.
#define FORECAST_TEMP_FEATURE_COUNT         (7u)
// Declare how many chronological slots the compiled model encodes per inference.
#define FORECAST_TEMP_NETWORK_WINDOW_SLOTS  (AI_FORECAST_TEMP_ML_MODEL_IN_1_SIZE / FORECAST_TEMP_FEATURE_COUNT)
// Declare the capacity of our raw history ring so pressure deltas have room.
#define FORECAST_TEMP_HISTORY_CAPACITY      (FORECAST_TEMP_WINDOW_LENGTH + FORECAST_TEMP_DELTA_P_LAG_HOURS)
// Declare how many daily log files we will scan when replaying persisted data.
#define FORECAST_TEMP_BOOTSTRAP_MAX_FILES   (32u)
// Declare the FatFs directory where the measurement logger writes CSV files.
#define FORECAST_TEMP_LOG_DIRECTORY         "0:/logs"

// Sanity-check that our feature count divides cleanly into the network input
// tensor so we can stride through the flattened buffer without overruns.
#if ((AI_FORECAST_TEMP_ML_MODEL_IN_1_SIZE % FORECAST_TEMP_FEATURE_COUNT) != 0u)
#error "AI_FORECAST_TEMP_ML_MODEL_IN_1_SIZE must be divisible by FORECAST_TEMP_FEATURE_COUNT"
#endif

#if (FORECAST_TEMP_NETWORK_WINDOW_SLOTS < FORECAST_TEMP_WINDOW_LENGTH)
#error "FORECAST_TEMP_WINDOW_LENGTH cannot exceed the number of time slots encoded in the model input tensor"
#endif

// Define a small struct that records a measurement log filename and its date key.
typedef struct {
        // Store the basename of the CSV file discovered on the SD card.
        char filename[48];
        // Store the sortable integer representation of the YYYY-MM-DD date.
        uint32_t date_key;
} forecast_temp_log_file_t;

// Define a struct that groups individual sensor readings by timestamp.
typedef struct {
        // Remember the ISO-8601 timestamp shared by the grouped measurements.
        char timestamp_iso8601[32];
        // Track whether this bundle currently holds any readings.
        bool in_use;
        // Track whether we captured a temperature value for this timestamp.
        bool has_temperature;
        // Track whether we captured a humidity value for this timestamp.
        bool has_humidity;
        // Track whether we captured a pressure value for this timestamp.
        bool has_pressure;
        // Track whether we captured an illuminance value for this timestamp.
        bool has_illuminance;
        // Hold the temperature reading associated with this timestamp.
        float temperature_c;
        // Hold the humidity reading associated with this timestamp.
        float humidity_pct;
        // Hold the pressure reading associated with this timestamp.
        float pressure_pa;
        // Hold the illuminance reading associated with this timestamp.
        float illuminance_lux;
} forecast_temp_measurement_bundle_t;

// Define a struct that aggregates completed bundles into hourly averages.
typedef struct {
        // Remember which hour the current accumulation represents.
        char hour_key[16];
        // Track the running sum of temperature readings inside this hour.
        float temperature_sum;
        // Track the running sum of humidity readings inside this hour.
        float humidity_sum;
        // Track the running sum of pressure readings inside this hour.
        float pressure_sum;
        // Track the running sum of illuminance readings inside this hour.
        float illuminance_sum;
        // Track how many samples contribute to the running sums.
        uint32_t sample_count;
} forecast_temp_hour_bucket_t;

// Store the feature means calculated during the data-preparation phase.
static const float g_feature_means[FORECAST_TEMP_FEATURE_COUNT] = {
        28.862367f,
        86.239013f,
        101063.322303f,
        4508.542050f,
        0.001307f,
        0.046776f,
        0.005473f
};
// Store the feature standard deviations so we can normalize each component.
static const float g_feature_stds[FORECAST_TEMP_FEATURE_COUNT] = {
        4.067106f,
        12.165888f,
        167.785437f,
        6669.496605f,
        1.689075f,
        198.491012f,
        0.706338f
};

// Hold the FreeRTOS thread handle so we do not start the task twice.
static osThreadId_t g_forecast_thread_id = NULL;
// Hold a mutex so multiple callers can safely read the latest prediction.
static osMutexId_t g_prediction_mutex = NULL;
// Remember the most recent prediction so other modules can use it.
static float g_latest_prediction_c = 0.0f;
// Track whether we have produced a prediction yet.
static bool g_latest_prediction_valid = false;

// Keep the neural network handle returned by X-CUBE-AI.
static ai_handle g_forecast_network = AI_HANDLE_NULL;
// Provide storage for the activations buffer that the runtime will fill.
AI_ALIGNED(4)
static ai_u8 g_network_activations[AI_FORECAST_TEMP_ML_MODEL_DATA_ACTIVATIONS_SIZE];
// Provide storage for the quantized input tensor.
static int8_t g_network_input_buffer[AI_FORECAST_TEMP_ML_MODEL_IN_1_SIZE];
// Provide storage for the quantized output tensor.
static int8_t g_network_output_buffer[AI_FORECAST_TEMP_ML_MODEL_OUT_1_SIZE];
// Cache the input buffer descriptor returned by the runtime.
static ai_buffer *g_network_inputs = NULL;
// Cache the output buffer descriptor returned by the runtime.
static ai_buffer *g_network_outputs = NULL;
// Cache the scale that maps normalized floats into int8 for the model input.
static float g_input_scale = 1.0f;
// Cache the zero-point that maps normalized floats into int8 for the model input.
static int32_t g_input_zero_point = 0;
// Cache the scale that maps the int8 output back into floating-point units.
static float g_output_scale = 1.0f;
// Cache the zero-point that maps the int8 output back into floating-point units.
static int32_t g_output_zero_point = 0;

// Define a helper struct that accumulates minute-level readings into an hour.
typedef struct {
        // Track the running sum of temperature readings.
        float temperature_sum;
        // Track the running sum of humidity readings.
        float humidity_sum;
        // Track the running sum of pressure readings.
        float pressure_sum;
        // Track the running sum of illuminance readings.
        float illuminance_sum;
        // Track how many minute samples we have folded into this bucket.
        uint32_t sample_count;
} forecast_temp_hour_accumulator_t;
// Instantiate a single accumulator that the task updates each minute.
static forecast_temp_hour_accumulator_t g_hour_accumulator = { 0 };

// Enumerate the states that drive the forecasting task state machine.
typedef enum {
        // Perform one-time neural-network initialization work.
        FORECAST_TEMP_STATE_INIT_NETWORK = 0,
        // Replay persisted CSV logs from the SD card into the feature window.
        FORECAST_TEMP_STATE_BOOTSTRAP,
        // Sleep until the next minute tick fires.
        FORECAST_TEMP_STATE_WAIT_MINUTE,
        // Fetch the latest sensor samples from the producer tasks.
        FORECAST_TEMP_STATE_FETCH_SENSORS,
        // Fold the freshly fetched minute sample into the hourly accumulator.
        FORECAST_TEMP_STATE_ACCUMULATE_MINUTE,
        // Determine whether a full hour has elapsed and averages can be computed.
        FORECAST_TEMP_STATE_CHECK_HOUR,
        // Generate normalized features for the completed hour.
        FORECAST_TEMP_STATE_PREPARE_FEATURES,
        // Append the normalized feature vector into the sliding window.
        FORECAST_TEMP_STATE_APPEND_FEATURES,
        // Export the sliding window into the quantized tensor expected by the model.
        FORECAST_TEMP_STATE_EXPORT_WINDOW,
        // Run the neural network and capture the resulting prediction.
        FORECAST_TEMP_STATE_RUN_INFERENCE,
        // Publish the prediction so other modules can consume it.
        FORECAST_TEMP_STATE_PUBLISH_PREDICTION
} forecast_temp_task_state_t;

// Track working data that flows across state-machine transitions.
typedef struct {
        // Remember the scheduler tick used when sleeping for a minute.
        TickType_t last_wake_tick;
        // Cache the most recent minute-level BME280 temperature reading.
        float temperature_c;
        // Cache the most recent minute-level BME280 humidity reading.
        float humidity_pct;
        // Cache the most recent minute-level BME280 pressure reading.
        float pressure_pa;
        // Cache the most recent minute-level VEML7700 illuminance reading.
        float illuminance_lux;
        // Cache the averaged hourly temperature during the CHECK_HOUR step.
        float hourly_temperature_c;
        // Cache the averaged hourly humidity during the CHECK_HOUR step.
        float hourly_humidity_pct;
        // Cache the averaged hourly pressure during the CHECK_HOUR step.
        float hourly_pressure_pa;
        // Cache the averaged hourly illuminance during the CHECK_HOUR step.
        float hourly_illuminance_lux;
        // Hold the normalized features generated for the current hour.
        float normalized_features[FORECAST_TEMP_FEATURE_COUNT];
        // Hold the most recent neural-network prediction prior to publication.
        float predicted_temperature_c;
} forecast_temp_task_context_t;

// Store the raw hourly temperature history so delta computations are easy.
static float g_hourly_temperature_history[FORECAST_TEMP_HISTORY_CAPACITY] = { 0.0f };
// Store the raw hourly pressure history so delta computations are easy.
static float g_hourly_pressure_history[FORECAST_TEMP_HISTORY_CAPACITY] = { 0.0f };
// Track the index of the oldest element in the raw history ring buffer.
static size_t g_hourly_history_head = 0u;
// Track how many elements are currently stored in the raw history ring buffer.
static size_t g_hourly_history_count = 0u;

// Store the normalized feature vectors for each hourly step in a ring buffer.
static float g_feature_window[FORECAST_TEMP_WINDOW_LENGTH][FORECAST_TEMP_FEATURE_COUNT] = { 0.0f };
// Track the index of the oldest feature vector in the sliding window.
static size_t g_feature_window_head = 0u;
// Track how many feature vectors are currently valid in the sliding window.
static size_t g_feature_window_count = 0u;

// Forward declare the task entry point so the CMSIS wrapper can launch it.
static void forecast_temp_task_entry(void *argument);
// Forward declare a helper that initializes the neural network instance.
static bool forecast_temp_initialize_network(void);
// Forward declare a helper that extracts quantization info for a buffer.
static void forecast_temp_extract_quantization(const ai_buffer *buffer, float *scale_out, int32_t *zero_point_out);
// Forward declare a helper that reads the RTC and converts to local hour-of-day.
static bool forecast_temp_read_local_hour(uint8_t *hour_out);
// Forward declare a helper that drops minute samples into the hourly accumulator.
static void forecast_temp_accumulate_minute_sample(float temperature_c, float humidity_pct, float pressure_pa, float illuminance_lux);
// Forward declare a helper that reports when the accumulator has a full hour ready.
static bool forecast_temp_finalize_hour_sample(float *temperature_c_out, float *humidity_pct_out, float *pressure_pa_out, float *illuminance_lux_out);
// Forward declare a helper that updates the raw history ring buffer.
static void forecast_temp_store_hourly_history(float temperature_c, float pressure_pa);
// Forward declare a helper that builds the normalized feature vector for the new hour.
static void forecast_temp_prepare_features(float temperature_c, float humidity_pct, float pressure_pa, float illuminance_lux, float out_features[FORECAST_TEMP_FEATURE_COUNT]);
// Forward declare a helper that pushes a normalized feature vector into the sliding window.
static void forecast_temp_append_feature_vector(const float normalized_features[FORECAST_TEMP_FEATURE_COUNT]);
// Forward declare a helper that exports the sliding window into the int8 input tensor.
static bool forecast_temp_export_window_to_network(int8_t *destination_buffer);
// Forward declare a helper that runs the model once the window is full.
static bool forecast_temp_run_inference(float *prediction_out);
// Forward declare a helper that replays persisted CSV data into the sliding window.
static void forecast_temp_bootstrap_from_sd_card(void);
// Forward declare a helper that resets a measurement bundle back to an empty state.
static void forecast_temp_bootstrap_reset_bundle(forecast_temp_measurement_bundle_t *bundle);
// Forward declare a helper that feeds a completed bundle into the hourly accumulator.
static void forecast_temp_bootstrap_submit_bundle(forecast_temp_measurement_bundle_t *bundle, forecast_temp_hour_bucket_t *hour_bucket, uint32_t *minute_counter, uint32_t *hour_counter);
// Forward declare a helper that accumulates a bundle into the current hour bucket.
static void forecast_temp_bootstrap_accumulate_bundle(const forecast_temp_measurement_bundle_t *bundle, forecast_temp_hour_bucket_t *hour_bucket, uint32_t *minute_counter, uint32_t *hour_counter);
// Forward declare a helper that finalizes the current hour bucket into features.
static void forecast_temp_bootstrap_finalize_hour(forecast_temp_hour_bucket_t *bucket, uint32_t *hour_counter);
// Forward declare a helper that streams one CSV file into the bootstrap machinery.
static void forecast_temp_bootstrap_process_file(const char *file_path, forecast_temp_hour_bucket_t *hour_bucket, uint32_t *minute_counter, uint32_t *hour_counter);

// Spin up the forecasting task if it is not already running.
bool forecast_temp_task_start(void) {
        // Bail out early if the task already exists.
        if (g_forecast_thread_id != NULL) {
                return true;
        }
        // Describe the thread attributes so CMSIS can create it.
        const osThreadAttr_t thread_attributes = {
                .name = "ForecastTemp",
                .priority = FORECAST_TEMP_TASK_PRIORITY,
                .stack_size = (uint32_t) (FORECAST_TEMP_TASK_STACK_WORDS * sizeof(StackType_t))
        };
        // Create the mutex that protects the shared prediction output.
        if (g_prediction_mutex == NULL) {
                const osMutexAttr_t prediction_mutex_attributes = {
                        .name = "forecast_temp_prediction_mutex",
                        .attr_bits = osMutexRecursive | osMutexPrioInherit,
                        .cb_mem = NULL,
                        .cb_size = 0u
                };
                g_prediction_mutex = osMutexNew(&prediction_mutex_attributes);
        }
        // Start the FreeRTOS task using the CMSIS wrapper.
        g_forecast_thread_id = osThreadNew(forecast_temp_task_entry, NULL, &thread_attributes);
        // Report success only if the thread pointer is valid.
        return (g_forecast_thread_id != NULL);
}

// Allow other modules to fetch the latest predicted temperature.
bool forecast_temp_get_latest_prediction(float *temperature_c_out) {
        // Guard against a NULL pointer from the caller.
        if (temperature_c_out == NULL) {
                return false;
        }
        // Assume the prediction is unavailable until proven otherwise.
        bool has_prediction = false;
        // Acquire the mutex before touching the shared state.
        if (g_prediction_mutex != NULL) {
                (void) osMutexAcquire(g_prediction_mutex, osWaitForever);
        }
        // Copy the prediction when we have emitted one already.
        if (g_latest_prediction_valid) {
                *temperature_c_out = g_latest_prediction_c;
                has_prediction = true;
        }
        // Release the mutex now that we are done.
        if (g_prediction_mutex != NULL) {
                (void) osMutexRelease(g_prediction_mutex);
        }
        // Report back to the caller whether the copy succeeded.
        return has_prediction;
}

// Reset a measurement bundle so future lines can reuse it safely.
static void forecast_temp_bootstrap_reset_bundle(forecast_temp_measurement_bundle_t *bundle) {
        // Guard against a NULL pointer so we do not dereference it.
        if (bundle == NULL) {
                return;
        }
        // Clear the stored timestamp string.
        memset(bundle->timestamp_iso8601, 0, sizeof(bundle->timestamp_iso8601));
        // Mark the bundle as currently unused.
        bundle->in_use = false;
        // Mark that we have not yet captured a temperature reading.
        bundle->has_temperature = false;
        // Mark that we have not yet captured a humidity reading.
        bundle->has_humidity = false;
        // Mark that we have not yet captured a pressure reading.
        bundle->has_pressure = false;
        // Mark that we have not yet captured an illuminance reading.
        bundle->has_illuminance = false;
        // Reset the cached temperature value.
        bundle->temperature_c = 0.0f;
        // Reset the cached humidity value.
        bundle->humidity_pct = 0.0f;
        // Reset the cached pressure value.
        bundle->pressure_pa = 0.0f;
        // Reset the cached illuminance value.
        bundle->illuminance_lux = 0.0f;
}

// Accumulate a completed measurement bundle into the active hour bucket.
static void forecast_temp_bootstrap_accumulate_bundle(const forecast_temp_measurement_bundle_t *bundle, forecast_temp_hour_bucket_t *hour_bucket, uint32_t *minute_counter, uint32_t *hour_counter) {
        // Guard against NULL arguments before we touch any fields.
        if ((bundle == NULL) || (hour_bucket == NULL)) {
                return;
        }
        // Bail out early when the bundle is missing any of the required readings.
        if (!(bundle->has_temperature && bundle->has_humidity && bundle->has_pressure && bundle->has_illuminance)) {
                return;
        }
        // Ignore bundles that do not carry a timestamp yet.
        if (!bundle->in_use) {
                return;
        }
        // Skip the bundle if the timestamp string is unexpectedly short.
        if (strlen(bundle->timestamp_iso8601) < 13u) {
                return;
        }
        // Allocate a small buffer where we can store the YYYY-MM-DDTHH hour key.
        char bundle_hour_key[16] = { 0 };
        // Copy only the first 13 characters so we keep the date and hour portion.
        (void) strncpy(bundle_hour_key, bundle->timestamp_iso8601, 13u);
        // Ensure the copied hour key is null-terminated.
        bundle_hour_key[13] = '\0';
        // Detect when the active bucket belongs to a different hour than this bundle.
        if ((hour_bucket->sample_count > 0u) && (strncmp(hour_bucket->hour_key, bundle_hour_key, sizeof(hour_bucket->hour_key)) != 0)) {
                // Finalize the existing bucket so the new hour can start fresh.
                forecast_temp_bootstrap_finalize_hour(hour_bucket, hour_counter);
        }
        // Initialize the hour key whenever this is the first sample for the bucket.
        if (hour_bucket->sample_count == 0u) {
                // Copy the bundle hour key into the bucket so future comparisons work.
                (void) strncpy(hour_bucket->hour_key, bundle_hour_key, sizeof(hour_bucket->hour_key) - 1u);
                // Guarantee the stored hour key is null-terminated.
                hour_bucket->hour_key[sizeof(hour_bucket->hour_key) - 1u] = '\0';
        }
        // Add the bundle temperature into the running hourly sum.
        hour_bucket->temperature_sum += bundle->temperature_c;
        // Add the bundle humidity into the running hourly sum.
        hour_bucket->humidity_sum += bundle->humidity_pct;
        // Add the bundle pressure into the running hourly sum.
        hour_bucket->pressure_sum += bundle->pressure_pa;
        // Add the bundle illuminance into the running hourly sum.
        hour_bucket->illuminance_sum += bundle->illuminance_lux;
        // Count how many bundles contribute to the current hour.
        hour_bucket->sample_count += 1u;
        // Increment the optional minute counter when the caller provided one.
        if (minute_counter != NULL) {
                *minute_counter += 1u;
        }
}

// Finalize an hour bucket by generating features and running inference if possible.
static void forecast_temp_bootstrap_finalize_hour(forecast_temp_hour_bucket_t *bucket, uint32_t *hour_counter) {
        // Guard against NULL pointers and empty buckets.
        if ((bucket == NULL) || (bucket->sample_count == 0u)) {
                return;
        }
        // Compute the reciprocal of the sample count so we can form averages.
        const float reciprocal = 1.0f / (float) bucket->sample_count;
        // Derive the average temperature across the hour.
        const float hourly_temperature = bucket->temperature_sum * reciprocal;
        // Derive the average humidity across the hour.
        const float hourly_humidity = bucket->humidity_sum * reciprocal;
        // Derive the average pressure across the hour.
        const float hourly_pressure = bucket->pressure_sum * reciprocal;
        // Derive the average illuminance across the hour.
        const float hourly_illuminance = bucket->illuminance_sum * reciprocal;
        // Allocate storage for the normalized feature vector.
        float normalized_features[FORECAST_TEMP_FEATURE_COUNT] = { 0.0f };
        // Populate the normalized features using the same pipeline as the live task.
        forecast_temp_prepare_features(hourly_temperature, hourly_humidity, hourly_pressure, hourly_illuminance, normalized_features);
        // Push the normalized features into the sliding window ring buffer.
        forecast_temp_append_feature_vector(normalized_features);
        // Check whether the window has enough history to satisfy the model.
        if (g_feature_window_count >= FORECAST_TEMP_WINDOW_LENGTH) {
                // Attempt to export the normalized window into the quantized network buffer.
                if (forecast_temp_export_window_to_network(g_network_input_buffer)) {
                        // Prepare a variable to hold the predicted temperature.
                        float predicted = 0.0f;
                        // Invoke the neural network to obtain a prediction.
                        if (forecast_temp_run_inference(&predicted)) {
                                // Acquire the mutex before updating the shared prediction state.
                                if (g_prediction_mutex != NULL) {
                                        (void) osMutexAcquire(g_prediction_mutex, osWaitForever);
                                }
                                // Store the freshly predicted temperature.
                                g_latest_prediction_c = predicted;
                                // Flag that a prediction is now available.
                                g_latest_prediction_valid = true;
                                // Release the mutex once the shared state is updated.
                                if (g_prediction_mutex != NULL) {
                                        (void) osMutexRelease(g_prediction_mutex);
                                }
                        }
                }
        }
        // Increment the hour counter when the caller supplied storage for it.
        if (hour_counter != NULL) {
                *hour_counter += 1u;
        }
        // Clear the hour key so the next accumulation knows it is starting fresh.
        bucket->hour_key[0] = '\0';
        // Reset the temperature sum back to zero.
        bucket->temperature_sum = 0.0f;
        // Reset the humidity sum back to zero.
        bucket->humidity_sum = 0.0f;
        // Reset the pressure sum back to zero.
        bucket->pressure_sum = 0.0f;
        // Reset the illuminance sum back to zero.
        bucket->illuminance_sum = 0.0f;
        // Reset the sample counter back to zero.
        bucket->sample_count = 0u;
}

// Forward a completed bundle into the hourly accumulator and reset it for reuse.
static void forecast_temp_bootstrap_submit_bundle(forecast_temp_measurement_bundle_t *bundle, forecast_temp_hour_bucket_t *hour_bucket, uint32_t *minute_counter, uint32_t *hour_counter) {
        // Guard against NULL pointers before doing any work.
        if ((bundle == NULL) || (hour_bucket == NULL)) {
                return;
        }
        // Skip bundles that were never populated with a timestamp.
        if (!bundle->in_use) {
                return;
        }
        // Only accumulate bundles that contain every required reading.
        if (bundle->has_temperature && bundle->has_humidity && bundle->has_pressure && bundle->has_illuminance) {
                // Hand the bundle to the hour accumulator so the sums stay current.
                forecast_temp_bootstrap_accumulate_bundle(bundle, hour_bucket, minute_counter, hour_counter);
        }
        // Reset the bundle so the next timestamp starts with clean state.
        forecast_temp_bootstrap_reset_bundle(bundle);
}

// Parse one CSV file and feed its samples into the bootstrap machinery.
static void forecast_temp_bootstrap_process_file(const char *file_path, forecast_temp_hour_bucket_t *hour_bucket, uint32_t *minute_counter, uint32_t *hour_counter) {
        // Guard against NULL inputs before attempting to open the file.
        if ((file_path == NULL) || (hour_bucket == NULL)) {
                return;
        }
        // Declare a FatFs file object that will represent the open CSV.
        FIL file;
        // Take the filesystem mutex so the logger task does not race us.
        FS_LOCK();
        // Attempt to open the CSV file for reading.
        FRESULT fr = f_open(&file, file_path, FA_READ);
        // If the open call failed we can release the mutex and bail out.
        if (fr != FR_OK) {
                // Release the filesystem mutex now that we are done with FatFs.
                FS_UNLOCK();
                // Emit a diagnostic so we know which file could not be opened.
                printf("[forecast] bootstrap failed to open %s fr=%d\r\n", file_path, (int) fr);
                // Nothing more we can do for this file.
                return;
        }
        // Prepare a reusable bundle so we can group lines by timestamp.
        forecast_temp_measurement_bundle_t bundle;
        // Reset the bundle to its empty state before parsing lines.
        forecast_temp_bootstrap_reset_bundle(&bundle);
        // Track whether we have skipped the CSV header yet.
        bool header_skipped = false;
        // Loop until we reach the end of the file.
        while(1) {
                // Allocate a buffer to hold one CSV line including the newline terminator.
                char line[192] = { 0 };
                // Read the next line from the CSV file.
                char *result = f_gets(line, (int) sizeof(line), &file);
                // Break the loop once FatFs signals end-of-file.
                if (result == NULL) {
                        break;
                }
                // Skip the header row on the first iteration.
                if (!header_skipped) {
                        // Mark that future iterations should parse data rows.
                        header_skipped = true;
                        // Continue to the next line without processing the header.
                        continue;
                }
                // Remove trailing CR/LF characters so string operations behave predictably.
                line[strcspn(line, "\r\n")] = '\0';
                // Tokenize the line using commas as separators.
                char *timestamp = strtok(line, ",");
                // Grab the sensor column from the CSV row.
                char *sensor = strtok(NULL, ",");
                // Grab the quantity column from the CSV row.
                char *quantity = strtok(NULL, ",");
                // Grab the numeric value column from the CSV row.
                char *value_str = strtok(NULL, ",");
                // Grab the unit column from the CSV row (may be unused).
                char *unit = strtok(NULL, ",");
                // Ignore lines that do not have the expected number of columns.
                if ((timestamp == NULL) || (sensor == NULL) || (quantity == NULL) || (value_str == NULL)) {
                        continue;
                }
                // Trim any lingering whitespace or newline characters from the unit string.
                if (unit != NULL) {
                        unit[strcspn(unit, "\r\n")] = '\0';
                }
                // Convert the numeric value from ASCII into a float.
                const float value = strtof(value_str, NULL);
                // Detect when a new timestamp begins so we can flush the previous bundle.
                if (!bundle.in_use || (strncmp(bundle.timestamp_iso8601, timestamp, sizeof(bundle.timestamp_iso8601)) != 0)) {
                        // Submit the previous bundle before starting the new one.
                        forecast_temp_bootstrap_submit_bundle(&bundle, hour_bucket, minute_counter, hour_counter);
                        // Reset the bundle so it can hold readings for the new timestamp.
                        forecast_temp_bootstrap_reset_bundle(&bundle);
                        // Copy the new timestamp into the bundle for future comparisons.
                        (void) strncpy(bundle.timestamp_iso8601, timestamp, sizeof(bundle.timestamp_iso8601) - 1u);
                        // Guarantee the stored timestamp is null-terminated.
                        bundle.timestamp_iso8601[sizeof(bundle.timestamp_iso8601) - 1u] = '\0';
                        // Mark that the bundle now contains an active timestamp.
                        bundle.in_use = true;
                }
                // Match the sensor and quantity so we can file the value into the correct slot.
                if ((strcmp(sensor, "bme280") == 0) && (strcmp(quantity, "temperature_c") == 0)) {
                        // Cache the BME280 temperature reading.
                        bundle.temperature_c = value;
                        // Record that the bundle now has a temperature measurement.
                        bundle.has_temperature = true;
                } else if ((strcmp(sensor, "bme280") == 0) && (strcmp(quantity, "humidity_pct") == 0)) {
                        // Cache the BME280 humidity reading.
                        bundle.humidity_pct = value;
                        // Record that the bundle now has a humidity measurement.
                        bundle.has_humidity = true;
                } else if ((strcmp(sensor, "bme280") == 0) && (strcmp(quantity, "pressure_pa") == 0)) {
                        // Cache the BME280 pressure reading.
                        bundle.pressure_pa = value;
                        // Record that the bundle now has a pressure measurement.
                        bundle.has_pressure = true;
                } else if ((strcmp(sensor, "veml7700") == 0) && (strcmp(quantity, "lux_lx") == 0)) {
                        // Cache the VEML7700 illuminance reading.
                        bundle.illuminance_lux = value;
                        // Record that the bundle now has an illuminance measurement.
                        bundle.has_illuminance = true;
                } else {
                        // Ignore other sensor quantities that the forecasting model does not consume.
                        continue;
                }
        }
        // Submit the final bundle so the last timestamp contributes to the aggregates.
        forecast_temp_bootstrap_submit_bundle(&bundle, hour_bucket, minute_counter, hour_counter);
        // Close the CSV file now that parsing is complete.
        (void) f_close(&file);
        // Release the filesystem mutex so other tasks may use FatFs.
        FS_UNLOCK();
}

// Discover recent CSV logs on the SD card and replay them into the feature window.
static void forecast_temp_bootstrap_from_sd_card(void) {
        // Allocate storage for the set of measurement files we care about.
        forecast_temp_log_file_t files[FORECAST_TEMP_BOOTSTRAP_MAX_FILES];
        // Track how many files we actually discovered in the directory.
        size_t file_count = 0u;
        // Declare a FatFs directory object so we can iterate through 0:/logs.
        DIR directory;
        // Take the filesystem mutex before issuing FatFs calls.
        FS_LOCK();
        // Attempt to open the logs directory on the SD card.
        FRESULT fr = f_opendir(&directory, FORECAST_TEMP_LOG_DIRECTORY);
        // Abort the bootstrap when the directory cannot be opened.
        if (fr != FR_OK) {
                // Release the filesystem mutex now that we are done with FatFs.
                FS_UNLOCK();
                // Emit a diagnostic so we know why the replay step was skipped.
                printf("[forecast] bootstrap failed to open %s fr=%d\r\n", FORECAST_TEMP_LOG_DIRECTORY, (int) fr);
                // Without directory access we cannot preload anything.
                return;
        }
        // Loop through every entry that FatFs reports inside the directory.
        while(1) {
                // Clear the file-info structure before each read.
                FILINFO info;
                memset(&info, 0, sizeof(info));
                // Fetch the next directory entry from FatFs.
                fr = f_readdir(&directory, &info);
                // Break out of the loop when an error occurs.
                if (fr != FR_OK) {
                        break;
                }
                // Stop the scan once FatFs reports the end of the directory.
                if (info.fname[0] == '\0') {
                        break;
                }
                // Skip subdirectories because we only care about CSV files.
                if ((info.fattrib & AM_DIR) != 0) {
                        continue;
                }
                // Prefer the long filename when available, else fall back to the 8.3 alias.
                const char *name_ptr = (info.fname[0] != '\0') ? info.fname : info.altname;
                // Skip entries without a usable filename string.
                if ((name_ptr == NULL) || (name_ptr[0] == '\0')) {
                        continue;
                }
                // Parse the date components embedded within the filename.
                unsigned int year = 0u;
                unsigned int month = 0u;
                unsigned int day = 0u;
                // Extract YYYY, MM, and DD from names like measurements_2025-10-21.csv.
                if (sscanf(name_ptr, "measurements_%4u-%2u-%2u.csv", &year, &month, &day) != 3) {
                        continue;
                }
                // Combine the date components into a sortable integer key.
                const uint32_t date_key = (year * 10000u) + (month * 100u) + day;
                // When we still have capacity we append the file to the list.
                if (file_count < FORECAST_TEMP_BOOTSTRAP_MAX_FILES) {
                        // Copy the filename so we can build full paths later on.
                        (void) strncpy(files[file_count].filename, name_ptr, sizeof(files[file_count].filename) - 1u);
                        // Guarantee the stored filename is null-terminated.
                        files[file_count].filename[sizeof(files[file_count].filename) - 1u] = '\0';
                        // Store the sortable date key alongside the filename.
                        files[file_count].date_key = date_key;
                        // Increase the count now that we added a file.
                        file_count += 1u;
                } else {
                        // Track which entry currently represents the oldest date.
                        size_t oldest_index = 0u;
                        // Remember the smallest date key seen so far.
                        uint32_t oldest_key = files[0].date_key;
                        // Walk the array to locate the oldest file we have stored.
                        for (size_t i = 1u; i < file_count; ++i) {
                                if (files[i].date_key < oldest_key) {
                                        oldest_key = files[i].date_key;
                                        oldest_index = i;
                                }
                        }
                        // Replace the oldest entry only when the new file is more recent.
                        if (date_key > oldest_key) {
                                (void) strncpy(files[oldest_index].filename, name_ptr, sizeof(files[oldest_index].filename) - 1u);
                                files[oldest_index].filename[sizeof(files[oldest_index].filename) - 1u] = '\0';
                                files[oldest_index].date_key = date_key;
                        }
                }
        }
        // Close the directory handle now that we are done scanning it.
        (void) f_closedir(&directory);
        // Release the filesystem mutex after the directory walk completes.
        FS_UNLOCK();
        // Abort when no measurement files were discovered.
        if (file_count == 0u) {
                printf("[forecast] bootstrap found no CSV logs to replay\r\n");
                return;
        }
        // Sort the discovered files by date so we replay them chronologically.
        for (size_t i = 0u; i < file_count; ++i) {
                // Walk every subsequent entry so we can bubble the earliest date forward.
                for (size_t j = i + 1u; j < file_count; ++j) {
                        // Swap the entries whenever we find an earlier date out of order.
                        if (files[j].date_key < files[i].date_key) {
                                // Temporarily stash the ith entry while we swap it with the jth.
                                forecast_temp_log_file_t tmp = files[i];
                                // Move the newer file into the current position.
                                files[i] = files[j];
                                // Move the previously stored file into the later slot.
                                files[j] = tmp;
                        }
                }
        }
        // Prepare the hour bucket that will accumulate bundles into hourly averages.
        forecast_temp_hour_bucket_t hour_bucket;
        // Zero-initialize the hour bucket before it sees any data.
        memset(&hour_bucket, 0, sizeof(hour_bucket));
        // Track how many minute-level bundles we replay from storage.
        uint32_t minute_counter = 0u;
        // Track how many hourly aggregates we reconstruct during replay.
        uint32_t hour_counter = 0u;
        // Iterate over the sorted files so older data feeds the window first.
        for (size_t i = 0u; i < file_count; ++i) {
                // Build the absolute path to the CSV file on the SD card.
                char path[96];
                (void) snprintf(path, sizeof(path), "%s/%s", FORECAST_TEMP_LOG_DIRECTORY, files[i].filename);
                // Process the CSV so its samples contribute to the bootstrap replay.
                forecast_temp_bootstrap_process_file(path, &hour_bucket, &minute_counter, &hour_counter);
        }
        // Flush any partial hour that may still be sitting in the accumulator.
        forecast_temp_bootstrap_finalize_hour(&hour_bucket, &hour_counter);
        // Emit a summary so we know how much persisted data was consumed.
        if (hour_counter > 0u) {
                printf("[forecast] bootstrap replayed %lu samples across %lu hours\r\n", (unsigned long) minute_counter, (unsigned long) hour_counter);
                printf("[forecast] bootstrap window_count=%lu history_count=%lu\r\n", (unsigned long) g_feature_window_count, (unsigned long) g_hourly_history_count);
        } else {
                printf("[forecast] bootstrap did not assemble any complete hours\r\n");
        }
}

// Main loop implemented as a state machine that polls sensors, updates features, and runs the model.
static void forecast_temp_task_entry(void *argument) {
        // Silence the unused-parameter warning for CMSIS.
        (void) argument;
        // Initialize the state-machine context so every field starts from a known value.
        forecast_temp_task_context_t context;
        memset(&context, 0, sizeof(context));
        // Begin execution in the initialization state.
        forecast_temp_task_state_t state = FORECAST_TEMP_STATE_INIT_NETWORK;
        // Stay in the state machine forever so the prediction remains fresh.
        while(1) {
                switch (state) {
                case FORECAST_TEMP_STATE_INIT_NETWORK:
                        // Initialize the neural network before the task begins processing data.
                        if (!forecast_temp_initialize_network()) {
                                printf("[forecast] failed to init neural network\r\n");
                        }
                        // Proceed immediately to the bootstrap phase regardless of success so the loop continues.
                        state = FORECAST_TEMP_STATE_BOOTSTRAP;
                        break;
                case FORECAST_TEMP_STATE_BOOTSTRAP:
                        // Replay persisted SD-card measurements so the window fills immediately.
                        forecast_temp_bootstrap_from_sd_card();
                        // Initialize the tick count used by vTaskDelayUntil().
                        context.last_wake_tick = xTaskGetTickCount();
                        // Transition into the periodic wait state.
                        state = FORECAST_TEMP_STATE_WAIT_MINUTE;
                        break;
                case FORECAST_TEMP_STATE_WAIT_MINUTE:
                        // Lazily initialize the wake tick when the state machine ever re-enters this state.
                        if (context.last_wake_tick == 0) {
                                context.last_wake_tick = xTaskGetTickCount();
                        }
                        // Sleep until the next one-minute boundary.
                        vTaskDelayUntil(&context.last_wake_tick, pdMS_TO_TICKS(FORECAST_TEMP_TASK_PERIOD_MS));
                        // After sleeping, attempt to fetch new sensor data.
                        state = FORECAST_TEMP_STATE_FETCH_SENSORS;
                        break;
                case FORECAST_TEMP_STATE_FETCH_SENSORS: {
                        // Pull the latest BME280 sample so we have temperature, humidity, and pressure.
                        const bool have_bme = bme280_get_latest(&context.temperature_c, &context.humidity_pct, &context.pressure_pa);
                        // Pull the latest VEML7700 sample so we have illuminance.
                        const bool have_veml = veml7700_get_latest_lux(&context.illuminance_lux);
                        // Skip this minute if either sensor has not produced a valid sample yet.
                        if (!(have_bme && have_veml)) {
                                state = FORECAST_TEMP_STATE_WAIT_MINUTE;
                                break;
                        }
                        // Fold the valid minute into the running hourly sums.
                        state = FORECAST_TEMP_STATE_ACCUMULATE_MINUTE;
                        break;
                }
                case FORECAST_TEMP_STATE_ACCUMULATE_MINUTE:
                        // Fold this minute into the running hourly sums.
                        forecast_temp_accumulate_minute_sample(context.temperature_c, context.humidity_pct, context.pressure_pa, context.illuminance_lux);
                        // Check whether a full hour boundary has been reached yet.
                        state = FORECAST_TEMP_STATE_CHECK_HOUR;
                        break;
                case FORECAST_TEMP_STATE_CHECK_HOUR: {
                        // Determine if the accumulator has enough data to emit an hourly average.
                        const bool have_full_hour = forecast_temp_finalize_hour_sample(&context.hourly_temperature_c, &context.hourly_humidity_pct, &context.hourly_pressure_pa, &context.hourly_illuminance_lux);
                        if (!have_full_hour) {
                                // Wait for additional minute samples when an hour has not elapsed.
                                printf("[forecast] waiting for full hour (%lu/%lu minute samples)\r\n",
                                       (unsigned long) g_hour_accumulator.sample_count,
                                       (unsigned long) FORECAST_TEMP_MINUTES_PER_HOUR);
                                state = FORECAST_TEMP_STATE_WAIT_MINUTE;
                                break;
                        }
                        // A full hour rolled over, so continue with feature preparation.
                        state = FORECAST_TEMP_STATE_PREPARE_FEATURES;
                        break;
                }
                case FORECAST_TEMP_STATE_PREPARE_FEATURES:
                        // Build the normalized feature vector for this completed hour.
                        forecast_temp_prepare_features(context.hourly_temperature_c, context.hourly_humidity_pct, context.hourly_pressure_pa, context.hourly_illuminance_lux, context.normalized_features);
                        // Queue the normalized vector for insertion into the sliding window.
                        state = FORECAST_TEMP_STATE_APPEND_FEATURES;
                        break;
                case FORECAST_TEMP_STATE_APPEND_FEATURES:
                        // Push the new feature vector into the sliding window history.
                        forecast_temp_append_feature_vector(context.normalized_features);
                        // Run inference only when the window is completely full.
                        if (g_feature_window_count < FORECAST_TEMP_WINDOW_LENGTH) {
                                state = FORECAST_TEMP_STATE_WAIT_MINUTE;
                        } else {
                                state = FORECAST_TEMP_STATE_EXPORT_WINDOW;
                        }
                        break;
                case FORECAST_TEMP_STATE_EXPORT_WINDOW:
                        // Export the sliding window into the int8 input tensor and bail out on failure.
                        if (!forecast_temp_export_window_to_network(g_network_input_buffer)) {
                                state = FORECAST_TEMP_STATE_WAIT_MINUTE;
                        } else {
                                state = FORECAST_TEMP_STATE_RUN_INFERENCE;
                        }
                        break;
                case FORECAST_TEMP_STATE_RUN_INFERENCE:
                        // Run the neural network and capture the predicted value.
                        if (!forecast_temp_run_inference(&context.predicted_temperature_c)) {
                                state = FORECAST_TEMP_STATE_WAIT_MINUTE;
                        } else {
                                state = FORECAST_TEMP_STATE_PUBLISH_PREDICTION;
                        }
                        break;
                case FORECAST_TEMP_STATE_PUBLISH_PREDICTION:
                        // Publish the prediction under the mutex so other modules can read it.
                        if (g_prediction_mutex != NULL) {
                                (void) osMutexAcquire(g_prediction_mutex, osWaitForever);
                        }
                        g_latest_prediction_c = context.predicted_temperature_c;
                        g_latest_prediction_valid = true;
                        if (g_prediction_mutex != NULL) {
                                (void) osMutexRelease(g_prediction_mutex);
                        }
                        // Return to the wait state so the cycle repeats on the next minute.
                        state = FORECAST_TEMP_STATE_WAIT_MINUTE;
                        break;
                default:
                        // Recover gracefully by restarting the periodic loop.
                        state = FORECAST_TEMP_STATE_WAIT_MINUTE;
                        break;
                }
        }
}

// Instantiate and configure the neural network runtime.
static bool forecast_temp_initialize_network(void) {
        // Point the activation table at our static activation buffer.
        ai_handle *activations_table = AI_FORECAST_TEMP_ML_MODEL_DATA_ACTIVATIONS_TABLE_GET();
        activations_table[0] = AI_HANDLE_PTR(g_network_activations);
        // Fetch the weights table that already points at the constant arrays.
        ai_handle *weights_table = AI_FORECAST_TEMP_ML_MODEL_DATA_WEIGHTS_TABLE_GET();
        // Create and initialize the network in one call.
        ai_error err = ai_forecast_temp_ml_model_create_and_init(&g_forecast_network, activations_table, weights_table);
        // Abort if the runtime failed to create the network instance.
        if (err.type != AI_ERROR_NONE) {
                printf("[forecast] create_and_init failed type=%d code=%d\r\n", err.type, err.code);
                return false;
        }
        // Cache the input buffer descriptors from the runtime.
        g_network_inputs = ai_forecast_temp_ml_model_inputs_get(g_forecast_network, NULL);
        // Cache the output buffer descriptors from the runtime.
        g_network_outputs = ai_forecast_temp_ml_model_outputs_get(g_forecast_network, NULL);
        // Abort if either descriptor pointer is missing.
        if ((g_network_inputs == NULL) || (g_network_outputs == NULL)) {
                printf("[forecast] failed to query network buffers\r\n");
                return false;
        }
        // Extract quantization info for the input tensor so we can pack data correctly.
        forecast_temp_extract_quantization(&g_network_inputs[0], &g_input_scale, &g_input_zero_point);
        // Extract quantization info for the output tensor so we can decode results correctly.
        forecast_temp_extract_quantization(&g_network_outputs[0], &g_output_scale, &g_output_zero_point);
        // Return success now that the runtime is ready.
        return true;
}

// Pull the quantization scale and zero-point from a buffer descriptor.
static void forecast_temp_extract_quantization(const ai_buffer *buffer, float *scale_out, int32_t *zero_point_out) {
        // Guard against missing metadata so we do not dereference NULL pointers.
        if ((buffer == NULL) || (scale_out == NULL) || (zero_point_out == NULL)) {
                return;
        }
        // Default to identity quantization when metadata is absent.
        *scale_out = 1.0f;
        *zero_point_out = 0;
        // Skip extraction when the metadata pointer is missing.
        if (buffer->meta_info == NULL) {
                return;
        }
        // Skip extraction when the integer quantization info is missing.
        if (buffer->meta_info->intq_info == NULL) {
                return;
        }
        // Skip extraction when there are no quantization entries.
        if (buffer->meta_info->intq_info->size == 0u) {
                return;
        }
        // Grab the first quantization entry because our tensors use per-tensor scaling.
        const ai_intq_info *info = &buffer->meta_info->intq_info->info[0];
        // Copy the scale when the runtime reports one.
        if (info->scale != NULL) {
                *scale_out = info->scale[0];
        }
        // Copy the zero-point when the runtime reports one. The metadata stores the value
        // in an int8 array for signed tensors, so read the first element with the correct width
        // and promote it to an int32_t to match the caller's expectations.
        if (info->zeropoint != NULL) {
                const int8_t *zero_point_array = (const int8_t *) info->zeropoint;
                *zero_point_out = (int32_t) zero_point_array[0];
        }
}

// Read the RTC and return the local hour-of-day (0-23).
static bool forecast_temp_read_local_hour(uint8_t *hour_out) {
        // Guard against NULL output pointers.
        if (hour_out == NULL) {
                return false;
        }
        // Allocate a small buffer for the ISO-8601 timestamp returned by the driver.
        char iso_buffer[24] = { 0 };
        // Query the DS3231 over I2C to get the current UTC timestamp.
        const HAL_StatusTypeDef status = ds3231_read_time_iso8601_utc_i2c1(iso_buffer, sizeof(iso_buffer));
        // Abort when the RTC read fails.
        if (status != HAL_OK) {
                return false;
        }
        // Validate that the hour characters are digits before converting them.
        if (!(isdigit((unsigned char) iso_buffer[11]) && isdigit((unsigned char) iso_buffer[12]))) {
                return false;
        }
        // Convert the UTC hour from ASCII into an integer.
        int utc_hour = ((iso_buffer[11] - '0') * 10) + (iso_buffer[12] - '0');
        // Adjust the UTC hour to local time (America/Port_of_Spain is UTC-4).
        int local_hour = utc_hour - 4;
        // Wrap the hour into the 0-23 range when subtracting crosses midnight.
        if (local_hour < 0) {
                local_hour += 24;
        }
        // Clamp the hour to a valid range just in case.
        if (local_hour < 0) {
                local_hour = 0;
        } else if (local_hour > 23) {
                local_hour = 23;
        }
        // Return the computed hour to the caller.
        *hour_out = (uint8_t) local_hour;
        // Signal success to the caller.
        return true;
}

// Add a minute-level reading into the hourly accumulator.
static void forecast_temp_accumulate_minute_sample(float temperature_c, float humidity_pct, float pressure_pa, float illuminance_lux) {
        // Add the new temperature sample into the running sum.
        g_hour_accumulator.temperature_sum += temperature_c;
        // Add the new humidity sample into the running sum.
        g_hour_accumulator.humidity_sum += humidity_pct;
        // Add the new pressure sample into the running sum.
        g_hour_accumulator.pressure_sum += pressure_pa;
        // Add the new illuminance sample into the running sum.
        g_hour_accumulator.illuminance_sum += illuminance_lux;
        // Increment the sample counter so we know how many minutes have accumulated.
        g_hour_accumulator.sample_count += 1u;
}

// Convert the minute accumulator into an hourly average when enough data exists.
static bool forecast_temp_finalize_hour_sample(float *temperature_c_out, float *humidity_pct_out, float *pressure_pa_out, float *illuminance_lux_out) {
        // Ensure all output pointers are valid.
        if ((temperature_c_out == NULL) || (humidity_pct_out == NULL) || (pressure_pa_out == NULL) || (illuminance_lux_out == NULL)) {
                return false;
        }
        // Only compute an average once enough minute samples are available.
        if (g_hour_accumulator.sample_count < FORECAST_TEMP_MINUTES_PER_HOUR) {
                return false;
        }
        // Compute the reciprocal once so we do not divide repeatedly.
        const float reciprocal = 1.0f / (float) g_hour_accumulator.sample_count;
        // Calculate the hourly average temperature.
        *temperature_c_out = g_hour_accumulator.temperature_sum * reciprocal;
        // Calculate the hourly average humidity.
        *humidity_pct_out = g_hour_accumulator.humidity_sum * reciprocal;
        // Calculate the hourly average pressure.
        *pressure_pa_out = g_hour_accumulator.pressure_sum * reciprocal;
        // Calculate the hourly average illuminance.
        *illuminance_lux_out = g_hour_accumulator.illuminance_sum * reciprocal;
        // Reset the accumulator so the next hour starts fresh.
        memset(&g_hour_accumulator, 0, sizeof(g_hour_accumulator));
        // Signal that the caller received a complete hour.
        return true;
}

// Store the latest hourly temperature and pressure so deltas stay accurate.
static void forecast_temp_store_hourly_history(float temperature_c, float pressure_pa) {
        // Drop the oldest sample when the ring buffer is already full.
        if (g_hourly_history_count >= FORECAST_TEMP_HISTORY_CAPACITY) {
                g_hourly_history_head = (g_hourly_history_head + 1u) % FORECAST_TEMP_HISTORY_CAPACITY;
                g_hourly_history_count -= 1u;
        }
        // Compute the index where the new sample should be written.
        const size_t tail_index = (g_hourly_history_head + g_hourly_history_count) % FORECAST_TEMP_HISTORY_CAPACITY;
        // Store the new hourly temperature sample.
        g_hourly_temperature_history[tail_index] = temperature_c;
        // Store the new hourly pressure sample.
        g_hourly_pressure_history[tail_index] = pressure_pa;
        // Increment the count now that a new sample has been inserted.
        g_hourly_history_count += 1u;
}

// Build the normalized feature vector for the latest hour of data.
static void forecast_temp_prepare_features(float temperature_c, float humidity_pct, float pressure_pa, float illuminance_lux, float out_features[FORECAST_TEMP_FEATURE_COUNT]) {
        // Compute the temperature delta relative to the previous hour when possible.
        float delta_temperature = 0.0f;
        // Compute the pressure delta relative to six hours prior when possible.
        float delta_pressure = 0.0f;
        // Only compute the temperature delta when at least one previous hour exists.
        if (g_hourly_history_count >= FORECAST_TEMP_DELTA_T_LAG_HOURS) {
                const size_t previous_index = (g_hourly_history_head + g_hourly_history_count - 1u) % FORECAST_TEMP_HISTORY_CAPACITY;
                const float previous_temperature = g_hourly_temperature_history[previous_index];
                delta_temperature = temperature_c - previous_temperature;
        }
        // Only compute the pressure delta when the history is deep enough.
        if (g_hourly_history_count >= FORECAST_TEMP_DELTA_P_LAG_HOURS) {
                const size_t lag_index = (g_hourly_history_head + g_hourly_history_count - FORECAST_TEMP_DELTA_P_LAG_HOURS) % FORECAST_TEMP_HISTORY_CAPACITY;
                const float lag_pressure = g_hourly_pressure_history[lag_index];
                delta_pressure = pressure_pa - lag_pressure;
        }
        // Read the local hour so we can generate the cyclical sine feature.
        uint8_t local_hour = 0u;
        float sin_hour = 0.0f;
        if (forecast_temp_read_local_hour(&local_hour)) {
                sin_hour = sinf((2.0f * (float) M_PI * (float) local_hour) / 24.0f);
        }
        // Populate the raw feature vector in the same order used during training.
        const float raw_features[FORECAST_TEMP_FEATURE_COUNT] = {
                temperature_c,
                humidity_pct,
                pressure_pa,
                illuminance_lux,
                delta_temperature,
                delta_pressure,
                sin_hour
        };
        // Normalize each feature using the stored StandardScaler parameters.
        for (size_t i = 0u; i < FORECAST_TEMP_FEATURE_COUNT; ++i) {
                out_features[i] = (raw_features[i] - g_feature_means[i]) / g_feature_stds[i];
        }
        // Store the raw values for delta computations in future iterations.
        forecast_temp_store_hourly_history(temperature_c, pressure_pa);
}

// Push a normalized feature vector into the sliding window ring buffer.
static void forecast_temp_append_feature_vector(const float normalized_features[FORECAST_TEMP_FEATURE_COUNT]) {
        // Drop the oldest feature vector when the window is already full.
        if (g_feature_window_count >= FORECAST_TEMP_WINDOW_LENGTH) {
                g_feature_window_head = (g_feature_window_head + 1u) % FORECAST_TEMP_WINDOW_LENGTH;
                g_feature_window_count -= 1u;
        }
        // Compute where the new feature vector should be written.
        const size_t tail_index = (g_feature_window_head + g_feature_window_count) % FORECAST_TEMP_WINDOW_LENGTH;
        // Copy the feature vector into the ring buffer.
        memcpy(g_feature_window[tail_index], normalized_features, sizeof(float) * FORECAST_TEMP_FEATURE_COUNT);
        // Increment the count now that a new vector has been stored.
        g_feature_window_count += 1u;
}

// Convert the sliding window into the quantized tensor expected by the network.
static bool forecast_temp_export_window_to_network(int8_t *destination_buffer) {
        // Guard against NULL destination pointers.
        if (destination_buffer == NULL) {
                return false;
        }
        // Ensure the window is full before exporting.
        if (g_feature_window_count < FORECAST_TEMP_WINDOW_LENGTH) {
                return false;
        }
        // Walk the window in chronological order and quantize each feature.
        size_t out_index = 0u;
        for (size_t time_index = 0u; time_index < FORECAST_TEMP_NETWORK_WINDOW_SLOTS; ++time_index) {
                size_t source_index;
                if (time_index < FORECAST_TEMP_WINDOW_LENGTH) {
                        source_index = (g_feature_window_head + time_index) % FORECAST_TEMP_WINDOW_LENGTH;
                } else {
                        // When the trained model encodes more time slots than the live window tracks,
                        // reuse the most recent feature vector so inference can still proceed.
                        const size_t last_valid_index = (g_feature_window_head + FORECAST_TEMP_WINDOW_LENGTH - 1u) % FORECAST_TEMP_WINDOW_LENGTH;
                        source_index = last_valid_index;
                }
                for (size_t feature_index = 0u; feature_index < FORECAST_TEMP_FEATURE_COUNT; ++feature_index) {
                        const float normalized_value = g_feature_window[source_index][feature_index];
                        float scaled_value = normalized_value / g_input_scale;
                        scaled_value += (float) g_input_zero_point;
                        int32_t quantized = (int32_t) lroundf(scaled_value);
                        if (quantized < -128) {
                                quantized = -128;
                        } else if (quantized > 127) {
                                quantized = 127;
                        }
                        destination_buffer[out_index] = (int8_t) quantized;
                        out_index += 1u;
                }
        }
        // Confirm that we filled the buffer completely.
        return (out_index == AI_FORECAST_TEMP_ML_MODEL_IN_1_SIZE);
}

// Run the neural network and decode the resulting prediction.
static bool forecast_temp_run_inference(float *prediction_out) {
        // Guard against NULL output pointers.
        if (prediction_out == NULL) {
                return false;
        }
        // Abort when the network handle is not valid.
        if (g_forecast_network == AI_HANDLE_NULL) {
                return false;
        }
        // Attach the input tensor buffer to the runtime descriptor.
        g_network_inputs[0].data = AI_HANDLE_PTR(g_network_input_buffer);
        // Attach the output tensor buffer to the runtime descriptor.
        g_network_outputs[0].data = AI_HANDLE_PTR(g_network_output_buffer);
        // Run a single forward pass through the model.
        const ai_i32 batch_count = ai_forecast_temp_ml_model_run(g_forecast_network, g_network_inputs, g_network_outputs);
        // Verify that the runtime processed exactly one batch.
        if (batch_count != 1) {
                printf("[forecast] inference failed batch_count=%ld\r\n", (long) batch_count);
                return false;
        }
        // Decode the int8 output back into floating-point units.
        const float dequantized = ((float) g_network_output_buffer[0] - (float) g_output_zero_point) * g_output_scale;
        // Return the decoded prediction to the caller.
        *prediction_out = dequantized;
        printf("[forecast] inference predicted %.2f C\r\n", (double) dequantized);
        // Signal success to the caller.
        return true;
}
