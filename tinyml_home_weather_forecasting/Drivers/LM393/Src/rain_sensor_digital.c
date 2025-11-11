/*=============================== Includes =================================*/
#include <stdio.h>
#include <string.h>
#include "stm32f7xx_hal.h"
#include "cmsis_os2.h"
#include "main.h"
#include "rain_sensor_digital.h"
#include "led_service.h"
#include "measurement_logger_task.h"

/*============================ Module State ================================*/

/* Finite-State Machine states for debouncing transitions. */
typedef enum {
	RAIN_SM_DRY_STABLE = 0,
	RAIN_SM_CONFIRM_WET,
	RAIN_SM_WET_STABLE,
	RAIN_SM_CONFIRM_DRY
} rain_sm_state_t;

/* Volatile because state is observed by other threads via getter. */
static volatile rain_digital_state_t g_public_state = RAIN_DIGITAL_STATE_DRY;
static const char *SENSOR = "lm393";
/* Internal FSM bookkeeping. */
static rain_sm_state_t g_sm_state = RAIN_SM_DRY_STABLE;
static uint32_t g_transition_start_ms = 0U;
static uint32_t g_last_heartbeat_ms = 0U;

/*============================== Prototypes ================================*/

/**
 * @brief   Return OS time in ms.
 *
 * Purpose: Encapsulate time source for testability.
 * Returns: milliseconds since kernel start.
 * Timing:  O(1).
 * Notes:   Wraparound semantics follow osKernelGetTickCount().
 */
static inline uint32_t RainDigital_Priv_Millis(void);

/**
 * @brief   Read raw DO level and map to logical wet/dry.
 *
 * Purpose: Convert pin logic to boolean “is_wet”.
 * Returns: true if wet, false if dry.
 * Notes:   Honors RAIN_DO_ACTIVE_LOW_WHEN_WET.
 */
static bool RainDigital_Priv_ReadIsWet(void);

/**
 * @brief   Advance the FSM one step based on current raw signal and time.
 *
 * Purpose: Implement debounced transitions DRY<->WET with confirm states.
 * Params:  raw_is_wet  Current raw reading (true=wet).
 * Returns: void.
 * Side Effects: Updates g_sm_state, g_public_state, and prints on transitions.
 * Timing:  O(1).
 * Errors:  None.
 */
static void RainDigital_Priv_FsmStep(bool raw_is_wet);

/**
 * @brief   Optionally print a periodic heartbeat line.
 *
 * Purpose: Give a slow cadence “still wet/dry” message for logs.
 * Returns: void.
 * Notes:   Controlled by RAIN_DO_ENABLE_HEARTBEAT_PRINTF.
 */
static void RainDigital_Priv_MaybePrintHeartbeat(void);

static void blink_led_for_sensor(void);

/*============================== Definitions ===============================*/

static inline uint32_t RainDigital_Priv_Millis(void) {
	/* Purpose: encapsulated millisecond tick. */
	return osKernelGetTickCount();
}

static bool RainDigital_Priv_ReadIsWet(void) {
	/* Purpose: map GPIO level to boolean wetness by polarity macro. */
	GPIO_PinState level = HAL_GPIO_ReadPin(RAIN_DO_GPIO_PORT, RAIN_DO_GPIO_PIN);
#if (RAIN_DO_ACTIVE_LOW_WHEN_WET == 1)
	return (level == GPIO_PIN_RESET);
#else
    return (level == GPIO_PIN_SET);
#endif
}

static void RainDigital_Priv_FsmStep(bool raw_is_wet) {
	const uint32_t now_ms = RainDigital_Priv_Millis();

	switch (g_sm_state) {
	case RAIN_SM_DRY_STABLE: {
		/* Transition attempt to WET? Start confirm window. */
		if (raw_is_wet) {
			g_sm_state = RAIN_SM_CONFIRM_WET;
			g_transition_start_ms = now_ms;
		}
		break;
	}

	case RAIN_SM_CONFIRM_WET: {
		/* Abort if raw reverted to dry before debounce timeout. */
		if (!raw_is_wet) {
			g_sm_state = RAIN_SM_DRY_STABLE;
			break;
		}

		/* Confirm WET after debounce window. */
		if ((now_ms - g_transition_start_ms) >= RAIN_DO_DEBOUNCE_MS) {
			g_sm_state = RAIN_SM_WET_STABLE;
			g_public_state = RAIN_DIGITAL_STATE_WET;
			printf("[RAIN] STARTED (t=%lu ms)\r\n", (unsigned long) now_ms);
			//put the sensor data on the queue for the SD card
			(void) measurement_logger_enqueue(&(measurement_logger_message_t ) {
							SENSOR, "is_raining", 1, "N/A" }, 10);
			blink_led_for_sensor();
		}
		break;
	}

	case RAIN_SM_WET_STABLE: {
		/* Transition attempt to DRY? Start confirm window. */
		if (!raw_is_wet) {
			g_sm_state = RAIN_SM_CONFIRM_DRY;
			g_transition_start_ms = now_ms;
		}
		break;
	}

	case RAIN_SM_CONFIRM_DRY: {
		/* Abort if raw went back to wet before debounce timeout. */
		if (raw_is_wet) {
			g_sm_state = RAIN_SM_WET_STABLE;
			break;
		}

		/* Confirm DRY after debounce window. */
		if ((now_ms - g_transition_start_ms) >= RAIN_DO_DEBOUNCE_MS) {
			g_sm_state = RAIN_SM_DRY_STABLE;
			g_public_state = RAIN_DIGITAL_STATE_DRY;
			printf("[RAIN] STOPPED (t=%lu ms)\r\n", (unsigned long) now_ms);
			//put the sensor data on the queue for the SD card
			(void) measurement_logger_enqueue(&(measurement_logger_message_t ) {
							SENSOR, "is_raining", 0, "N/A" }, 10);
			blink_led_for_sensor();
		}
		break;
	}

	default: {
		/* Defensive: reset to a sane state. */
		g_sm_state = RAIN_SM_DRY_STABLE;
		g_public_state = RAIN_DIGITAL_STATE_DRY;
		break;
	}
	}
}

static void blink_led_for_sensor(void) {
	//drive external RGB LED amber for this sensor
	led_service_pulse_activity_rgb(255,130,0, 1000);
}

static void RainDigital_Priv_MaybePrintHeartbeat(void) {
#if (RAIN_DO_ENABLE_HEARTBEAT_PRINTF == 1)
	const uint32_t now_ms = RainDigital_Priv_Millis();
	if ((now_ms - g_last_heartbeat_ms) >= RAIN_DO_HEARTBEAT_PERIOD_MS) {
		g_last_heartbeat_ms = now_ms;
		const int is_raining_state =
				(g_public_state == RAIN_DIGITAL_STATE_WET) ? 1 : 0;
		printf("[RAIN] HEARTBEAT: is_raining=%d (t=%lu ms)\r\n",
				is_raining_state, (unsigned long) now_ms);
		//put the sensor data on the queue for the SD card
		(void) measurement_logger_enqueue(&(measurement_logger_message_t ) {
						SENSOR, "is_raining", is_raining_state, "N/A" },
				10);
		blink_led_for_sensor();
	}
#else
	(void) g_last_heartbeat_ms; /* suppress unused warning if disabled */
#endif
}

/*=============================== API ======================================*/

bool RainDigital_Service_Initialize(void) {
	/*----------------------------------------------------------------------
	 * Purpose: Initialize module state based on a single DO reading.
	 * Params : none
	 * Returns: true on success.
	 * Side Effects: Sets initial FSM/public state and heartbeat timer.
	 * Preconditions: GPIO configured as input; board powered; RTOS running.
	 * Postconditions: Ready for polling in RainDigital_Service_Task().
	 * Concurrency: Should be called once from init context before task start.
	 * Timing: O(1).
	 * Errors: None expected; returns true unconditionally.
	 * Notes: If your DO polarity differs, update RAIN_DO_ACTIVE_LOW_WHEN_WET.
	 *---------------------------------------------------------------------*/
	const bool wet_now = RainDigital_Priv_ReadIsWet();
	g_sm_state = wet_now ? RAIN_SM_WET_STABLE : RAIN_SM_DRY_STABLE;
	g_public_state = wet_now ? RAIN_DIGITAL_STATE_WET : RAIN_DIGITAL_STATE_DRY;
	g_transition_start_ms = RainDigital_Priv_Millis();
	g_last_heartbeat_ms = g_transition_start_ms;

	printf("[RAIN] INIT: state=%s (t=%lu ms)\r\n", wet_now ? "WET" : "DRY",
			(unsigned long) g_transition_start_ms);

	return true;
}

rain_digital_state_t RainDigital_Service_GetState(void) {
	/*----------------------------------------------------------------------
	 * Purpose: Provide the latest debounced wet/dry state to callers.
	 * Params : none
	 * Returns: RAIN_DIGITAL_STATE_DRY or RAIN_DIGITAL_STATE_WET.
	 * Side Effects: None.
	 * Preconditions: RainDigital_Service_Initialize() was called.
	 * Postconditions: None.
	 * Concurrency: Safe to call from any thread; state is volatile.
	 * Timing: O(1).
	 * Errors: None.
	 * Notes: Designed for lightweight logging or feature computation.
	 *---------------------------------------------------------------------*/
	return g_public_state;
}

void RainDigital_Service_Task(void *argument) {
	/*----------------------------------------------------------------------
	 * Purpose: Poll DO at a fixed cadence, run FSM debounce, printf events.
	 * Params : argument (unused; pass NULL)
	 * Returns: never returns.
	 * Side Effects: Prints transition and (optional) heartbeat lines.
	 * Preconditions: RTOS running; module initialized; GPIO configured.
	 * Postconditions: N/A.
	 * Concurrency: Runs as a dedicated thread; interacts via globals only.
	 * Timing: Poll every RAIN_DO_TASK_PERIOD_MS; debounce uses time windows.
	 * Errors: None; defensive reset on unexpected states.
	 * Notes: Keep printf fast to avoid blocking other tasks.
	 *---------------------------------------------------------------------*/
	(void) argument;

	while (1) {
		const bool raw_is_wet = RainDigital_Priv_ReadIsWet();
		RainDigital_Priv_FsmStep(raw_is_wet);
		RainDigital_Priv_MaybePrintHeartbeat();
		const rain_digital_state_t s = RainDigital_Service_GetState();
		//printf("[RAIN] STATUS: %s\r\n", (s == RAIN_DIGITAL_STATE_WET) ? "WET" : "DRY");
		osDelay(RAIN_DO_TASK_PERIOD_MS);
	}
}

void RainDigital_Service_EXTI_Notify(uint16_t gpio_pin) {
	/*----------------------------------------------------------------------
	 * Purpose: Compatibility stub for projects that still wire EXTI callbacks.
	 * Params : gpio_pin (ignored)
	 * Returns: void
	 * Side Effects: None.
	 * Preconditions: None.
	 * Postconditions: None.
	 * Concurrency: ISR-safe (does nothing).
	 * Timing: O(1).
	 * Errors: None.
	 * Notes: Polling mode does not require EXTI; this is a no-op.
	 *---------------------------------------------------------------------*/
	(void) gpio_pin;
}
