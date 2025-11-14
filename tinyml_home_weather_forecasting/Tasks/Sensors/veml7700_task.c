/*
 * File:    veml7700_task.c
 * Brief:   VEML7700 ambient light sensor task implemented as a deterministic
 *          state machine using CMSIS-RTOS2 (osThreadNew/osDelay).
 */

#include "veml7700_task.h"      /* Public API and defaults */
#include <stdio.h>
#include <string.h>
#include "measurement_logger_task.h"
#include "led_service.h"

/* ======================== Device Constants (Datasheet) ======================== */
#define VEML7700_I2C_ADDR7                      (0x10u)  /* 7-bit address */

#define VEML7700_REG_ALS_CONF_0                 (0x00u)
#define VEML7700_REG_ALS_WH                     (0x01u)
#define VEML7700_REG_ALS_WL                     (0x02u)
#define VEML7700_REG_PWR_SAV                    (0x03u)
#define VEML7700_REG_ALS_DATA                   (0x04u)
#define VEML7700_REG_WHITE_DATA                 (0x05u)
#define VEML7700_REG_ALS_INT                    (0x06u)

/* ======================== Tunable Module Constants ======================== */
#define VEML7700_I2C_TIMEOUT_MS                 (20u)
#define VEML7700_DEVICE_READY_TRIALS            (1u)
#define VEML7700_SAMPLE_PERIOD_MS               (10000u)
#define VEML7700_MAX_CONSECUTIVE_ERRORS         (10u)
#define VEML7700_FATAL_ERROR_HOLD_MS            (1000u)

/* ======================== State Machine Types ======================== */
typedef enum {
	veml7700_fsm_state_uninitialized = 0,
	veml7700_fsm_state_configure_device,
	veml7700_fsm_state_wait_first_integration,
	veml7700_fsm_state_sample_sensors,
	veml7700_fsm_state_handle_read_error,
	veml7700_fsm_state_handle_fatal_error
} veml7700_fsm_state_t;

/* ======================== Module Context ======================== */
typedef struct {
	I2C_HandleTypeDef *i2c_handle_ptr;
	uint8_t gain_code; /* ALS_CONF_0[12:11] */
	uint8_t integration_time_code; /* ALS_CONF_0[9:6]   */
	uint16_t last_als_counts;
	uint16_t last_white_counts;
	float last_lux;
	bool has_latest_sample;
	uint32_t integration_time_delay_ms;
	uint32_t consecutive_error_count;
} veml7700_module_context_t;

/* ======================== File-Private Data ======================== */
static veml7700_module_context_t g_veml7700_context;
static osThreadId_t g_veml7700_thread_id = NULL;
static osMutexId_t g_veml7700_sample_mutex_id = NULL;

/* ======================== Forward Declarations ======================== */
static bool i2c_write_register_u16_(uint8_t register_address,
		uint16_t register_value);
static bool i2c_read_register_u16_(uint8_t register_address,
		uint16_t *register_value_out);
static uint16_t convert_it_code_to_ms_(uint8_t integration_time_code);
static float convert_gain_code_to_gain_scalar_(uint8_t gain_code);
static float convert_counts_to_lux_(uint16_t als_counts, uint8_t gain_code,
		uint8_t integration_time_code);
static bool veml7700_configure_device_(void);
static void veml7700_thread_entry_function_(void *thread_argument_ptr);

/* =======================================================================
 * Function: i2c_write_register_u16_
 * Purpose:  Write a 16-bit little-endian value to a VEML7700 register.
 * Params:   register_address  - 8-bit register address.
 *           register_value    - 16-bit value to write (LSB first on wire).
 * Returns:  true if write succeeded; false otherwise.
 * Side Effects:   Performs blocking I2C transaction via HAL.
 * Preconditions:  g_veml7700_context.i2c_handle_ptr != NULL.
 * Postconditions: Register updated if true is returned.
 * Concurrency:    Called only by the VEML thread; no external locking required.
 * Timing:         Blocks up to VEML7700_I2C_TIMEOUT_MS.
 * Errors:         Returns false on HAL error.
 * Notes:          Uses 8-bit mem addr; 7-bit device addr shifted left per HAL.
 * =======================================================================
 */
static bool i2c_write_register_u16_(uint8_t register_address,
		uint16_t register_value) {
	uint8_t register_buffer[2u];
	register_buffer[0u] = (uint8_t) (register_value & 0xFFu); /* LSB first */
	register_buffer[1u] = (uint8_t) ((register_value >> 8) & 0xFFu); /* MSB next  */

	return (HAL_I2C_Mem_Write(g_veml7700_context.i2c_handle_ptr,
			(VEML7700_I2C_ADDR7 << 1), register_address,
			I2C_MEMADD_SIZE_8BIT, register_buffer,
			(uint16_t) sizeof(register_buffer),
			(uint32_t) VEML7700_I2C_TIMEOUT_MS) == HAL_OK);
}

/* =======================================================================
 * Function: i2c_read_register_u16_
 * Purpose:  Read a 16-bit little-endian value from a VEML7700 register.
 * Params:   register_address   - 8-bit register address.
 *           register_value_out - pointer to receive 16-bit value.
 * Returns:  true if read succeeded; false otherwise.
 * Side Effects:   Performs blocking I2C transaction via HAL.
 * Preconditions:  g_veml7700_context.i2c_handle_ptr != NULL; pointer valid.
 * Postconditions: *register_value_out updated on success.
 * Concurrency:    Called only by the VEML thread; no external locking required.
 * Timing:         Blocks up to VEML7700_I2C_TIMEOUT_MS.
 * Errors:         Returns false on HAL error.
 * Notes:          Combines two bytes: LSB then MSB.
 * =======================================================================
 */
static bool i2c_read_register_u16_(uint8_t register_address,
		uint16_t *register_value_out) {
	uint8_t register_buffer[2u] = { 0u, 0u };

	if (register_value_out == NULL) {
		return false;
	}

	if (HAL_I2C_Mem_Read(g_veml7700_context.i2c_handle_ptr,
			(VEML7700_I2C_ADDR7 << 1), register_address,
			I2C_MEMADD_SIZE_8BIT, register_buffer,
			(uint16_t) sizeof(register_buffer),
			(uint32_t) VEML7700_I2C_TIMEOUT_MS) != HAL_OK) {
		return false;
	}

	*register_value_out = (uint16_t) ((uint16_t) register_buffer[0u]
			| ((uint16_t) register_buffer[1u] << 8));
	return true;
}

/* =======================================================================
 * Function: convert_it_code_to_ms_
 * Purpose:  Convert VEML7700 integration-time code to milliseconds.
 * Params:   integration_time_code - encoded per datasheet.
 * Returns:  Integration time in milliseconds (approx).
 * Side Effects:   None.
 * Preconditions:  None.
 * Postconditions: None.
 * Concurrency:    Thread-safe (pure function).
 * Timing:         Constant time.
 * Errors:         Defaults to 100 ms if code is unknown.
 * Notes:          Matches device table for IT codes.
 * =======================================================================
 */
static uint16_t convert_it_code_to_ms_(uint8_t integration_time_code) {
	switch (integration_time_code) {
	case VEML7700_IT_CODE_100MS:
		return 100u;
	case VEML7700_IT_CODE_200MS:
		return 200u;
	case VEML7700_IT_CODE_400MS:
		return 400u;
	case VEML7700_IT_CODE_800MS:
		return 800u;
	case VEML7700_IT_CODE_50MS:
		return 50u;
	case VEML7700_IT_CODE_25MS:
		return 25u;
	default:
		return 100u;
	}
}

/* =======================================================================
 * Function: convert_gain_code_to_gain_scalar_
 * Purpose:  Convert VEML7700 gain code to a numeric scalar.
 * Params:   gain_code - encoded per datasheet.
 * Returns:  Gain scalar (unitless).
 * Side Effects:   None.
 * Preconditions:  None.
 * Postconditions: None.
 * Concurrency:    Thread-safe (pure function).
 * Timing:         Constant time.
 * Errors:         Defaults to 1.0f if code is unknown.
 * Notes:          Used in lux conversion calculation.
 * =======================================================================
 */
static float convert_gain_code_to_gain_scalar_(uint8_t gain_code) {
	switch (gain_code) {
	case VEML7700_GAIN_CODE_1X:
		return 1.0f; /* 1x   */
	case VEML7700_GAIN_CODE_2X:
		return 2.0f; /* 2x   */
	case VEML7700_GAIN_CODE_1_8X:
		return (1.0f / 8.0f);/* 1/8x */
	case VEML7700_GAIN_CODE_1_4X:
		return (1.0f / 4.0f);/* 1/4x */
	default:
		return 1.0f;
	}
}

/* =======================================================================
 * Function: convert_counts_to_lux_
 * Purpose:  Convert raw ALS counts to approximate lux.
 * Params:   als_counts            - ALS register counts.
 *           gain_code             - VEML gain code.
 *           integration_time_code - VEML integration time code.
 * Returns:  Approximate lux as a float.
 * Side Effects:   None.
 * Preconditions:  Codes must match device configuration.
 * Postconditions: None.
 * Concurrency:    Thread-safe (pure function).
 * Timing:         Constant time.
 * Errors:         N/A.
 * Notes:          Base resolution 0.0042 lx/ct at GAIN=2, IT=800ms;
 *                 scaled by (800/IT_ms) * (2/gain).
 * =======================================================================
 */
static float convert_counts_to_lux_(uint16_t als_counts, uint8_t gain_code,
		uint8_t integration_time_code) {
	const float base_resolution_lux_per_count = 0.0042f;
	const float it_ms = (float) convert_it_code_to_ms_(integration_time_code);
	const float gain = convert_gain_code_to_gain_scalar_(gain_code);
	const float resolution = base_resolution_lux_per_count * (800.0f / it_ms)
			* (2.0f / gain);

	return ((float) als_counts) * resolution;
}

/* =======================================================================
 * Function: veml7700_configure_device_
 * Purpose:  Program ALS configuration and perform a presence check.
 * Params:   None.
 * Returns:  true on success; false on any I2C/programming failure.
 * Side Effects:   Writes to multiple device registers; prints logs on failure.
 * Preconditions:  g_veml7700_context.i2c_handle_ptr initialized; gain/IT set.
 * Postconditions: Device configured for continuous ALS conversion if success.
 * Concurrency:    Called from sensor thread only.
 * Timing:         Blocks on I2C; delays one integration time on success.
 * Errors:         false if device NACKs or a write fails.
 * Notes:          Disables power-save; clears thresholds; SD=0 (ALS enabled).
 * =======================================================================
 */
static bool veml7700_configure_device_(void) {
	if (HAL_I2C_IsDeviceReady(g_veml7700_context.i2c_handle_ptr,
			(VEML7700_I2C_ADDR7 << 1),
			VEML7700_DEVICE_READY_TRIALS, (uint32_t) 10u) != HAL_OK) {
		printf("[VEML7700] No ACK at 0x%02X on this I2C bus\r\n",
		VEML7700_I2C_ADDR7);
		return false;
	}

	/* ALS_CONF_0: bits[12:11]=gain, bits[9:6]=IT, bit0=SD(0=on) */
	const uint16_t configuration_value =
			(uint16_t) ((((uint16_t) g_veml7700_context.gain_code & 0x3u) << 11)
					| (((uint16_t) g_veml7700_context.integration_time_code
							& 0xFu) << 6) | 0u /* SD=0 power on */);

	if (!i2c_write_register_u16_(VEML7700_REG_ALS_CONF_0,
			configuration_value)) {
		printf("[VEML7700] Configuration write failed\r\n");
		return false;
	}

	/* Optional housekeeping (do not fail the init if these NACK) */
	(void) i2c_write_register_u16_(VEML7700_REG_PWR_SAV, 0x0000u);
	(void) i2c_write_register_u16_(VEML7700_REG_ALS_WH, 0x0000u);
	(void) i2c_write_register_u16_(VEML7700_REG_ALS_WL, 0x0000u);

	/* Cache the integration delay for the state machine. */
	g_veml7700_context.integration_time_delay_ms =
			(uint32_t) convert_it_code_to_ms_(
					g_veml7700_context.integration_time_code);

	/* Wait one full integration before first read */
	osDelay(g_veml7700_context.integration_time_delay_ms);
	return true;
}

/* =======================================================================
 * Function: veml7700_thread_entry_function_
 * Purpose:  CMSIS-RTOS2 thread entry function implementing a VEML7700 sampling
 *           state machine with error handling and controlled timing.
 * Params:   thread_argument_ptr - Unused (must be NULL).
 * Returns:  None (does not return).
 * Side Effects:   Updates module-global latest sample variables under mutex.
 * Preconditions:  Context fields initialized; kernel running; mutex created.
 * Postconditions: Thread runs until fatal error; then terminates itself.
 * Concurrency:    Only this thread accesses I2C/helper functions; data sharing
 *                 with other threads via veml7700_get_latest().
 * Timing:         Periodic sampling per VEML7700_SAMPLE_PERIOD_MS.
 * Errors:         On repeated failures transitions to fatal state.
 * =======================================================================
 */
static void veml7700_thread_entry_function_(void *thread_argument_ptr) {
	(void) thread_argument_ptr;

	veml7700_fsm_state_t current_state = veml7700_fsm_state_configure_device;

	while (1) {
		switch (current_state) {
		case veml7700_fsm_state_configure_device: {
			const bool configured_ok = veml7700_configure_device_();
			g_veml7700_context.consecutive_error_count = 0u;

			current_state =
					(configured_ok) ?
							veml7700_fsm_state_sample_sensors :
							veml7700_fsm_state_handle_fatal_error;
		}
			break;

		case veml7700_fsm_state_wait_first_integration: {
			osDelay(g_veml7700_context.integration_time_delay_ms);
			current_state = veml7700_fsm_state_sample_sensors;
		}
			break;

		case veml7700_fsm_state_sample_sensors: {
			uint16_t als_counts = 0u;
			uint16_t white_counts = 0u;

			const bool als_ok = i2c_read_register_u16_(VEML7700_REG_ALS_DATA,
					&als_counts);
			const bool white_ok = i2c_read_register_u16_(
			VEML7700_REG_WHITE_DATA, &white_counts);

			if (als_ok && white_ok) {
				const float lux_estimate = convert_counts_to_lux_(als_counts,
						g_veml7700_context.gain_code,
						g_veml7700_context.integration_time_code);

				if (g_veml7700_sample_mutex_id != NULL) {
					(void) osMutexAcquire(g_veml7700_sample_mutex_id,
					osWaitForever);
				}
				g_veml7700_context.last_als_counts = als_counts;
				g_veml7700_context.last_white_counts = white_counts;
				g_veml7700_context.last_lux = lux_estimate;
				g_veml7700_context.has_latest_sample = true;
				if (g_veml7700_sample_mutex_id != NULL) {
					(void) osMutexRelease(g_veml7700_sample_mutex_id);
				}
				//put the sensor data on the queue for the SD card
				static const char *SENSOR = "veml7700";
				(void) measurement_logger_enqueue(
						&(measurement_logger_message_t ) { SENSOR,
										"lux_lx", lux_estimate, "lx" }, 10);
				//drive external RGB LED teal for this sensor
				led_service_pulse_activity_rgb(0, 128, 128, 1000);
				/*
				 //Optional debug printout
				 printf("[VEML7700] ALS=%5u WHITE=%5u Lux≈%.2f\r\n",
				 (unsigned) als_counts, (unsigned) white_counts,
				 (double) lux_estimate);
				 */
				g_veml7700_context.consecutive_error_count = 0u;
				osDelay(VEML7700_SAMPLE_PERIOD_MS);
			} else {
				printf("[VEML7700] Read failed (ALS ok=%u, WHITE ok=%u)\r\n",
						(unsigned) als_ok, (unsigned) white_ok);
				current_state = veml7700_fsm_state_handle_read_error;
			}
		}
			break;

		case veml7700_fsm_state_handle_read_error: {
			g_veml7700_context.consecutive_error_count++;

			if (g_veml7700_context.consecutive_error_count
					>= VEML7700_MAX_CONSECUTIVE_ERRORS) {
				current_state = veml7700_fsm_state_handle_fatal_error;
			} else {
				/* Small backoff before next attempt; keep overall cadence reasonable */
				osDelay(50u);
				current_state = veml7700_fsm_state_sample_sensors;
			}
		}
			break;

		case veml7700_fsm_state_handle_fatal_error:
		default: {
			// show the red led
			led_command_t err = { .led_identifier = led_identifier_ld3,
					.pattern_identifier = led_pattern_identifier_error_code,
					.error_code_count = 1, /* one-blink code = sensor */
					.duration_ms = 0, /* persist until cleared */
					.priority_level = 10 };
			(void) led_service_set_pattern(&err);
			// wait 5s and restart task loop
			printf(
					"We ran into an error with the VEML7700 sensor. Restarting task loop in 5s.");
			vTaskDelay(pdMS_TO_TICKS(5000));
			current_state = veml7700_fsm_state_configure_device;
		}
			break;
		}
	}
}

/* =======================================================================
 * Function: veml7700_start_task
 * Purpose:  Create and start the VEML7700 CMSIS-RTOS2 sampling thread.
 * Params:   hi2c            - pointer to initialized HAL I2C handle.
 *           prio            - CMSIS-RTOS2 thread priority (e.g., osPriorityLow).
 *           stack_words     - thread stack size in 32-bit words.
 * Returns:  osThreadId_t of the created thread, or NULL on failure.
 * Side Effects:   Initializes module context and creates a CMSIS-RTOS2 thread
 *                 and mutex for sample sharing.
 * Preconditions:  Kernel initialized; I2C configured and ready.
 * Postconditions: Thread running on success; context populated.
 * Concurrency:    Not thread-safe; call during system init.
 * Timing:         Bounded by osThreadNew() and non-blocking steps.
 * Errors:         Returns NULL if parameters invalid or thread creation fails.
 * Notes:          Matches header API exactly.
 * =======================================================================
 */
osThreadId_t veml7700_start_task(I2C_HandleTypeDef *hi2c, osPriority_t prio,
		uint32_t stack_words) {
	if ((hi2c == NULL)) {
		return NULL;
	}

	if (g_veml7700_thread_id != NULL) {
		return g_veml7700_thread_id; /* Already running */
	}

	/* Initialize module context */
	memset(&g_veml7700_context, 0, sizeof(g_veml7700_context));
	g_veml7700_context.i2c_handle_ptr = hi2c;
	g_veml7700_context.gain_code = VEML7700_DEFAULT_GAIN_CODE;
	g_veml7700_context.integration_time_code = VEML7700_DEFAULT_IT_CODE;
	g_veml7700_context.integration_time_delay_ms =
			(uint32_t) convert_it_code_to_ms_(
					g_veml7700_context.integration_time_code);

	/* Create a mutex for sharing the latest sample */
	const osMutexAttr_t mutex_attributes = { .name = "veml7700_sample_mutex",
			.attr_bits = osMutexRecursive | osMutexPrioInherit, .cb_mem = NULL,
			.cb_size = 0u };
	g_veml7700_sample_mutex_id = osMutexNew(&mutex_attributes);

	/* Create the CMSIS-RTOS2 thread */
	const osThreadAttr_t thread_attributes = { .name = "VEML7700", .priority =
			prio, .stack_size = (uint32_t) (stack_words * sizeof(uint32_t)) };

	g_veml7700_thread_id = osThreadNew(veml7700_thread_entry_function_, NULL,
			&thread_attributes);

	return g_veml7700_thread_id;
}

/* =======================================================================
 * Function: veml7700_get_latest
 * Purpose:  Retrieve the most recent sample produced by the VEML thread.
 * Params:   lux_out             - optional pointer to receive lux (may be NULL).
 *           als_counts_out      - optional pointer to receive ALS counts.
 *           white_counts_out    - optional pointer to receive WHITE counts.
 * Returns:  true if a valid sample is available; false otherwise.
 * Side Effects:   None beyond copying data.
 * Preconditions:  VEML thread has run at least once and produced a sample.
 * Postconditions: Output pointers updated only if non-NULL and sample available.
 * Concurrency:    Uses a mutex to read a coherent snapshot while the thread may
 *                 update the values.
 * Timing:         Short and bounded; no I2C operations.
 * Errors:         false if no sample available yet.
 * Notes:          Safe to call from any thread context.
 * =======================================================================
 */
bool veml7700_get_latest(float *lux_out, uint16_t *als_counts_out,
                uint16_t *white_counts_out) {
        bool is_valid = false;

	if (g_veml7700_sample_mutex_id != NULL) {
		(void) osMutexAcquire(g_veml7700_sample_mutex_id, osWaitForever);
	}

	if (g_veml7700_context.has_latest_sample) {
		if (lux_out != NULL) {
			*lux_out = g_veml7700_context.last_lux;
		}
		if (als_counts_out != NULL) {
			*als_counts_out = g_veml7700_context.last_als_counts;
		}
		if (white_counts_out != NULL) {
			*white_counts_out = g_veml7700_context.last_white_counts;
		}
		is_valid = true;
	}

	if (g_veml7700_sample_mutex_id != NULL) {
		(void) osMutexRelease(g_veml7700_sample_mutex_id);
	}

        return is_valid;
}

/* =======================================================================
 * Function: veml7700_get_latest_lux
 * Purpose:  Retrieve only the cached lux value from the latest sample.
 * Params:   lux_out            - pointer to receive lux estimate (must be non-NULL).
 * Returns:  true if a valid sample is available; false otherwise.
 * Side Effects:   None beyond copying data via veml7700_get_latest().
 * Preconditions:  VEML thread has run at least once and produced a sample.
 * Postconditions: *lux_out updated on success.
 * Concurrency:    Delegates locking to veml7700_get_latest().
 * Timing:         Short and bounded; no I2C operations.
 * Errors:         false if no sample available yet or lux_out is NULL.
 * Notes:          Convenience wrapper so callers do not need to supply NULLs.
 * =======================================================================
 */
bool veml7700_get_latest_lux(float *lux_out) {
        if (lux_out == NULL) {
                return false;
        }

        return veml7700_get_latest(lux_out, NULL, NULL);
}
