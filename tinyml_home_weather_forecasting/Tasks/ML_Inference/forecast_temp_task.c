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
// Pull in our BME280 task interface to read the latest environmental sample.
#include "Tasks/Sensors/bme280_task.h"
// Pull in our VEML7700 interface to grab illuminance readings.
#include "Tasks/Sensors/veml7700_task.h"
// Pull in the DS3231 driver so we can ask the RTC for the current time.
#include "Drivers/DS3231/Inc/ds3231.h"
// Pull in the auto-generated network API so we can run inference.
#include "X-CUBE-AI/App/forecast_temp_ml_model.h"
// Pull in the data helpers so we can allocate the activation buffers correctly.
#include "X-CUBE-AI/App/forecast_temp_ml_model_data.h"
// Pull in the data parameter helpers so we can set up the activation table.
#include "X-CUBE-AI/App/forecast_temp_ml_model_data_params.h"

// Provide a fallback for M_PI when the math library does not define it.
#ifndef M_PI
// Define M_PI as a float literal so our sine calculation has a sensible value.
#define M_PI 3.14159265358979323846f
#endif

// Declare how many 32-bit words the forecasting task stack should use.
#define FORECAST_TEMP_TASK_STACK_WORDS      (1024u)
// Declare the CMSIS-RTOS priority the forecasting task should run at.
#define FORECAST_TEMP_TASK_PRIORITY         (osPriorityLow)
// Declare how often (in milliseconds) we poll the sensors for a new reading.
#define FORECAST_TEMP_TASK_PERIOD_MS        (60000u)
// Declare how many hourly samples we keep inside the sliding input window.
#define FORECAST_TEMP_WINDOW_LENGTH         (168u)
// Declare how many minute samples we fold into a single hourly aggregate.
#define FORECAST_TEMP_MINUTES_PER_HOUR      (60u)
// Declare how many hours back we look when computing the temperature delta.
#define FORECAST_TEMP_DELTA_T_LAG_HOURS     (1u)
// Declare how many hours back we look when computing the pressure delta.
#define FORECAST_TEMP_DELTA_P_LAG_HOURS     (6u)
// Declare how many engineered plus raw features the model expects per time step.
#define FORECAST_TEMP_FEATURE_COUNT         (7u)
// Declare the capacity of our raw history ring so pressure deltas have room.
#define FORECAST_TEMP_HISTORY_CAPACITY      (FORECAST_TEMP_WINDOW_LENGTH + FORECAST_TEMP_DELTA_P_LAG_HOURS)

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

// Main loop that polls sensors, updates features, and runs the model.
static void forecast_temp_task_entry(void *argument) {
        // Silence the unused-parameter warning for CMSIS.
        (void) argument;
        // Initialize the neural network before we start looping.
        if (!forecast_temp_initialize_network()) {
                printf("[forecast] failed to init neural network\r\n");
        }
        // Remember the scheduler tick count so we can sleep accurately.
        TickType_t last_wake_tick = xTaskGetTickCount();
        // Loop forever so the prediction stays up to date.
        for (;;) {
                // Sleep until the next one-minute boundary.
                vTaskDelayUntil(&last_wake_tick, pdMS_TO_TICKS(FORECAST_TEMP_TASK_PERIOD_MS));
                // Pull the latest BME280 sample so we have temperature, humidity, and pressure.
                float temperature_c = 0.0f;
                float humidity_pct = 0.0f;
                float pressure_pa = 0.0f;
                const bool have_bme = bme280_get_latest(&temperature_c, &humidity_pct, &pressure_pa);
                // Pull the latest VEML7700 sample so we have illuminance.
                float illuminance_lux = 0.0f;
                const bool have_veml = veml7700_get_latest(&illuminance_lux, NULL, NULL);
                // Skip this minute if either sensor has not produced a valid sample yet.
                if (!(have_bme && have_veml)) {
                        continue;
                }
                // Fold this minute into the running hourly sums.
                forecast_temp_accumulate_minute_sample(temperature_c, humidity_pct, pressure_pa, illuminance_lux);
                // See whether we have collected a full hour of data yet.
                float hourly_temperature_c = 0.0f;
                float hourly_humidity_pct = 0.0f;
                float hourly_pressure_pa = 0.0f;
                float hourly_illuminance_lux = 0.0f;
                const bool have_full_hour = forecast_temp_finalize_hour_sample(&hourly_temperature_c, &hourly_humidity_pct, &hourly_pressure_pa, &hourly_illuminance_lux);
                // Only proceed to feature generation when a full hour rolls over.
                if (!have_full_hour) {
                        continue;
                }
                // Build the normalized feature vector for this completed hour.
                float normalized_features[FORECAST_TEMP_FEATURE_COUNT] = { 0.0f };
                forecast_temp_prepare_features(hourly_temperature_c, hourly_humidity_pct, hourly_pressure_pa, hourly_illuminance_lux, normalized_features);
                // Push the new feature vector into the sliding window history.
                forecast_temp_append_feature_vector(normalized_features);
                // Run inference only when the window is completely full.
                if (g_feature_window_count < FORECAST_TEMP_WINDOW_LENGTH) {
                        continue;
                }
                // Export the sliding window into the int8 input tensor.
                if (!forecast_temp_export_window_to_network(g_network_input_buffer)) {
                        continue;
                }
                // Run the neural network and capture the predicted value.
                float predicted_temperature_c = 0.0f;
                const bool inference_ok = forecast_temp_run_inference(&predicted_temperature_c);
                // Skip publication when the runtime reports an error.
                if (!inference_ok) {
                        continue;
                }
                // Publish the prediction under the mutex so other modules can read it.
                if (g_prediction_mutex != NULL) {
                        (void) osMutexAcquire(g_prediction_mutex, osWaitForever);
                }
                g_latest_prediction_c = predicted_temperature_c;
                g_latest_prediction_valid = true;
                if (g_prediction_mutex != NULL) {
                        (void) osMutexRelease(g_prediction_mutex);
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
        // Copy the zero-point when the runtime reports one.
        if (info->zeropoint != NULL) {
                const ai_i32 *zero_point_array = (const ai_i32 *) info->zeropoint;
                *zero_point_out = zero_point_array[0];
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
        for (size_t time_index = 0u; time_index < FORECAST_TEMP_WINDOW_LENGTH; ++time_index) {
                const size_t source_index = (g_feature_window_head + time_index) % FORECAST_TEMP_WINDOW_LENGTH;
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
        // Signal success to the caller.
        return true;
}
