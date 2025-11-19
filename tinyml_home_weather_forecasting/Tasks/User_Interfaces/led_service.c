#include "led_service.h"

/* ===== Internal configuration ===== */
#define LED_SERVICE_QUEUE_LENGTH                       (8u)
#define LED_SERVICE_TASK_TICK_MS                       (20u)   /* update timestep */
#define LED_SERVICE_HEARTBEAT_PERIOD_MS                (2000u) /* green */
#define LED_SERVICE_HEARTBEAT_PULSE_MS                 (500u)
#define LED_SERVICE_ERROR_ON_MS                        (1500u)
#define LED_SERVICE_ERROR_OFF_MS                       (1500u)
#define LED_SERVICE_ERROR_PAUSE_MS                     (1500u)
#define LED_SERVICE_ACTIVITY_CREDIT_DECAY_MS       (LED_SERVICE_TASK_TICK_MS) /* 20 ms per tick */
#define LED_SERVICE_ACTIVITY_CREDIT_CAP_MS         (2000u)                    /* clamp to 2 s */
#define LED_SERVICE_RGB_PULSE_QUEUE_LENGTH             (8u)

/* Command for an RGB pulse (queued) */
typedef struct {
    uint8_t  r_8bit;
    uint8_t  g_8bit;
    uint8_t  b_8bit;
    uint16_t duration_ms;
} led_rgb_pulse_cmd_t;

typedef struct {
	GPIO_TypeDef *gpio_port;
	uint16_t gpio_pin;
	led_pattern_identifier_t active_pattern_identifier;
	uint16_t active_period_ms;
	uint8_t active_duty_percent;
	uint16_t remaining_duration_ms;
	uint8_t active_error_code_count;
	uint8_t active_priority_level;
	/* runtime */
	uint16_t phase_ms;
	uint8_t error_cycle_index;
	bool is_output_on;
} led_runtime_t;

/* Module statics */
static QueueHandle_t g_led_command_queue_handle = NULL;
static TaskHandle_t g_led_task_handle = NULL;
static led_runtime_t g_leds[3];
/* Queue handle + currently active pulse */
static QueueHandle_t g_rgb_pulse_queue_handle = NULL;


static void prv_led_service_task(void *argument);
static void prv_led_apply_output(led_runtime_t *led_ptr, bool turn_on);
static uint16_t g_led_activity_credit_ms = 0u; /* shared LD2 rate-meter credit */
/* ===== GPIO helper ===== */
static inline void prv_gpio_write(GPIO_TypeDef *port, uint16_t pin,
		GPIO_PinState state) {
	HAL_GPIO_WritePin(port, pin, state);
}


/* ===== External RGB (PWM) support ===== */
static TIM_HandleTypeDef *g_rgb_timer_handle_ptr = NULL;
static uint32_t g_rgb_channel_r = 0, g_rgb_channel_g = 0, g_rgb_channel_b = 0;
static bool g_rgb_is_common_anode = false;

typedef struct {
	uint16_t remaining_ms;
	uint8_t r_8bit;
	uint8_t g_8bit;
	uint8_t b_8bit;
} led_rgb_flash_request_t;

static led_rgb_flash_request_t g_rgb_flash_req = { 0, 0, 0, 0 };

static inline void prv_rgb_apply_8bit(uint8_t r, uint8_t g, uint8_t b) {
	if (g_rgb_timer_handle_ptr == NULL) {
		return;
	}

	const uint32_t period = __HAL_TIM_GET_AUTORELOAD(g_rgb_timer_handle_ptr);
	/* Map 0..255 -> 0..period */
	uint32_t r_counts = ((uint32_t) r * period) / 255U;
	uint32_t g_counts = ((uint32_t) g * period) / 255U;
	uint32_t b_counts = ((uint32_t) b * period) / 255U;

	if (g_rgb_is_common_anode) {
		/* CA: brighter = lower compare (sink current) */
		__HAL_TIM_SET_COMPARE(g_rgb_timer_handle_ptr, g_rgb_channel_r,
				period - r_counts);
		__HAL_TIM_SET_COMPARE(g_rgb_timer_handle_ptr, g_rgb_channel_g,
				period - g_counts);
		__HAL_TIM_SET_COMPARE(g_rgb_timer_handle_ptr, g_rgb_channel_b,
				period - b_counts);
	} else {
		/* CC (HW-479): brighter = higher compare (source current) */
		__HAL_TIM_SET_COMPARE(g_rgb_timer_handle_ptr, g_rgb_channel_r,
				r_counts);
		__HAL_TIM_SET_COMPARE(g_rgb_timer_handle_ptr, g_rgb_channel_g,
				g_counts);
		__HAL_TIM_SET_COMPARE(g_rgb_timer_handle_ptr, g_rgb_channel_b,
				b_counts);
	}
}

/* ***************************************************************************
 * Function: led_service_rgb_board_init
 * Purpose : Start PWM on three channels for the external RGB LED module.
 * Params  : See header.
 * Returns : bool - true on success.
 * Side effects : Starts hardware PWM on given channels.
 * Preconditions : Timer base + channels configured by CubeMX.
 * Postconditions: RGB channels are ready for duty updates via CCR.
 * Concurrency : Not thread-safe vs. another init; call once at startup.
 * Timing  : O(1).
 * Errors  : Returns false if timer_handle_ptr is NULL.
 * Notes   : Recommended PWM freq ~1 kHz; keep GPIO current < MCU limits.
 * ***************************************************************************/
bool led_service_rgb_board_init(TIM_HandleTypeDef *timer_handle_ptr,
		uint32_t channel_red, uint32_t channel_green, uint32_t channel_blue,
		bool is_common_anode) {
	if (timer_handle_ptr == NULL) {
		return false;
	}

	g_rgb_timer_handle_ptr = timer_handle_ptr;
	g_rgb_channel_r = channel_red;
	g_rgb_channel_g = channel_green;
	g_rgb_channel_b = channel_blue;
	g_rgb_is_common_anode = is_common_anode;

	/* Start PWM on all three channels */
	if (HAL_TIM_PWM_Start(g_rgb_timer_handle_ptr, g_rgb_channel_r) != HAL_OK) {
		return false;
	}
	if (HAL_TIM_PWM_Start(g_rgb_timer_handle_ptr, g_rgb_channel_g) != HAL_OK) {
		return false;
	}
	if (HAL_TIM_PWM_Start(g_rgb_timer_handle_ptr, g_rgb_channel_b) != HAL_OK) {
		return false;
	}

	/* Ensure OFF initially */
	prv_rgb_apply_8bit(0, 0, 0);

	g_rgb_pulse_queue_handle = xQueueCreate(LED_SERVICE_RGB_PULSE_QUEUE_LENGTH,
	                                        sizeof(led_rgb_pulse_cmd_t));
	if (g_rgb_pulse_queue_handle == NULL) {
	    return false;
	}

	return true;
}

/* ***************************************************************************
 * Function: led_service_pulse_activity_rgb
 * Purpose  : Queue a timed RGB pulse (FIFO, non-preemptive).
 * Params   : red_8bit, green_8bit, blue_8bit - 0..255 intensities
 *            duration_ms - pulse length in milliseconds (0 => 1 ms)
 * Returns  : void
 * Side effects : Enqueues a pulse command; service task plays it later.
 * Preconditions : led_service_init() and led_service_rgb_board_init() done.
 * Postconditions: Each queued pulse will display fully, in order queued.
 * Concurrency : Task-safe via FreeRTOS queue (non-blocking send, 0 timeout).
 * Timing   : O(1).
 * Errors   : If the queue is full, the newest pulse is dropped silently.
 * Notes    : Add a *_from_isr() variant if you need to call from ISRs.
 * ***************************************************************************/
void led_service_pulse_activity_rgb(uint8_t red_8bit,
                                    uint8_t green_8bit,
                                    uint8_t blue_8bit,
                                    uint16_t duration_ms)
{
    if (g_rgb_pulse_queue_handle == NULL) {
        return;
    }

    led_rgb_pulse_cmd_t cmd;
    cmd.r_8bit     = red_8bit;
    cmd.g_8bit     = green_8bit;
    cmd.b_8bit     = blue_8bit;
    cmd.duration_ms = (duration_ms == 0u) ? 1u : duration_ms;

    (void) xQueueSend(g_rgb_pulse_queue_handle, &cmd, 0);
}

void led_service_pulse_activity_rgb_from_isr(uint8_t r, uint8_t g, uint8_t b, uint16_t duration_ms)
{
    if (g_rgb_pulse_queue_handle == NULL) { return; }
    led_rgb_pulse_cmd_t cmd = { r, g, b, (duration_ms == 0u) ? 1u : duration_ms };
    BaseType_t hpw = pdFALSE;
    (void) xQueueSendFromISR(g_rgb_pulse_queue_handle, &cmd, &hpw);
    portYIELD_FROM_ISR(hpw);
}


bool led_service_init(void) {
	/* Create queue */
	g_led_command_queue_handle = xQueueCreate(LED_SERVICE_QUEUE_LENGTH,
			sizeof(led_command_t));
	if (g_led_command_queue_handle == NULL) {
		return false;
	}

	/* Enable GPIO clocks if not already enabled (F7 example) */
	__HAL_RCC_GPIOA_CLK_ENABLE();
	__HAL_RCC_GPIOB_CLK_ENABLE();

	/* Configure LED pins: push-pull, no pull, low speed is fine */
	GPIO_InitTypeDef gpio_init = { 0 };

	gpio_init.Mode = GPIO_MODE_OUTPUT_PP;
	gpio_init.Pull = GPIO_NOPULL;
	gpio_init.Speed = GPIO_SPEED_FREQ_LOW;

	/* LD1 */
	gpio_init.Pin = LED1_Pin;
	HAL_GPIO_Init(LED1_GPIO_Port, &gpio_init);
	/* LD2 */
	gpio_init.Pin = LED2_Pin;
	HAL_GPIO_Init(LED2_GPIO_Port, &gpio_init);
	/* LD3 */
	gpio_init.Pin = LED3_Pin;
	HAL_GPIO_Init(LED3_GPIO_Port, &gpio_init);

	/* Drive OFF initially (active high on Nucleo LEDs) */
	prv_gpio_write(LED1_GPIO_Port, LED1_Pin, GPIO_PIN_RESET);
	prv_gpio_write(LED2_GPIO_Port, LED2_Pin, GPIO_PIN_RESET);
	prv_gpio_write(LED3_GPIO_Port, LED3_Pin, GPIO_PIN_RESET);

	/* Populate runtime metadata */
	g_leds[led_identifier_ld1] =
			(led_runtime_t ) { .gpio_port = LED1_GPIO_Port, .gpio_pin =
					LED1_Pin, .active_pattern_identifier =
							led_pattern_identifier_heartbeat,
							.active_period_ms = LED_SERVICE_HEARTBEAT_PERIOD_MS,
							.active_duty_percent = 0,
							.remaining_duration_ms = 0,
							.active_error_code_count = 0,
							.active_priority_level = 1, .phase_ms = 0,
							.error_cycle_index = 0, .is_output_on = false };
	g_leds[led_identifier_ld2] = (led_runtime_t ) { .gpio_port = LED2_GPIO_Port,
					.gpio_pin = LED2_Pin, .active_pattern_identifier =
							led_pattern_identifier_off, .active_priority_level =
							0 };
	g_leds[led_identifier_ld3] = (led_runtime_t ) { .gpio_port = LED3_GPIO_Port,
					.gpio_pin = LED3_Pin, .active_pattern_identifier =
							led_pattern_identifier_off, .active_priority_level =
							5 /* errors outrank others */
			};
	return true;
}

bool led_service_start(UBaseType_t task_priority_uxprio,
		uint16_t task_stack_words) {
	BaseType_t rc = xTaskCreate(prv_led_service_task, "led_service",
			task_stack_words, NULL, task_priority_uxprio, &g_led_task_handle);
	return (rc == pdPASS);
}

bool led_service_set_pattern(const led_command_t *command_ptr) {
	if (g_led_command_queue_handle == NULL || command_ptr == NULL) {
		return false;
	}
	return (xQueueSend(g_led_command_queue_handle, command_ptr, 0) == pdPASS);
}

bool led_service_set_pattern_from_isr(const led_command_t *command_ptr) {
	if (g_led_command_queue_handle == NULL || command_ptr == NULL) {
		return false;
	}
	BaseType_t xHigherPriorityTaskWoken = pdFALSE;
	BaseType_t rc = xQueueSendFromISR(g_led_command_queue_handle, command_ptr,
			&xHigherPriorityTaskWoken);
	portYIELD_FROM_ISR(xHigherPriorityTaskWoken);
	return (rc == pdPASS);
}

void led_service_pulse_activity(void) {
	/* ~200 ms of visibility per event; tweak as you wish (100..400) */
	led_service_activity_bump(200u);
}

/**
 * \brief   Add activity credit so LD2 (blue) stays ON while recent events are happening.
 *
 * \param   credit_ms   Milliseconds of credit to add (1..LED_SERVICE_ACTIVITY_CREDIT_CAP_MS).
 * \return  void
 *
 * \sideeffects  Updates a shared counter; LED task consumes it each tick.
 * \pre          led_service_start() was called.
 * \post         LD2 is forced ON until credit decays to zero.
 * \concurrency  Task-context only; uses a tiny critical section.
 * \timing       O(1).
 * \errors       None.
 * \notes        Call after each measurement and/or each successful log write.
 */
void led_service_activity_bump(uint16_t credit_ms) {
	if (credit_ms == 0u) {
		credit_ms = 1u;
	}
	taskENTER_CRITICAL();
	uint32_t new_credit = (uint32_t) g_led_activity_credit_ms
			+ (uint32_t) credit_ms;
	if (new_credit > LED_SERVICE_ACTIVITY_CREDIT_CAP_MS) {
		new_credit = LED_SERVICE_ACTIVITY_CREDIT_CAP_MS;
	}
	g_led_activity_credit_ms = (uint16_t) new_credit;
	taskEXIT_CRITICAL();
}

/* ===== Task and rendering ===== */

static void prv_led_render_tick(led_runtime_t *led_ptr, uint16_t tick_ms) {
	/* Update duration countdown */
	if (led_ptr->remaining_duration_ms > 0) {
		if (tick_ms >= led_ptr->remaining_duration_ms) {
			led_ptr->remaining_duration_ms = 0;
		} else {
			led_ptr->remaining_duration_ms -= tick_ms;
		}
	}

	/* Advance phase */
	led_ptr->phase_ms += tick_ms;

	switch (led_ptr->active_pattern_identifier) {
	case led_pattern_identifier_off:
		prv_led_apply_output(led_ptr, false);
		break;

	case led_pattern_identifier_on:
		prv_led_apply_output(led_ptr, true);
		break;

	case led_pattern_identifier_blink: {
		uint16_t period =
				(led_ptr->active_period_ms > 0u) ?
						led_ptr->active_period_ms : 1000u;
		uint16_t on_ms = (period * (uint16_t) led_ptr->active_duty_percent)
				/ 100u;
		uint16_t t = (uint16_t) (led_ptr->phase_ms % period);
		prv_led_apply_output(led_ptr, (t < on_ms));
	}
		break;

	case led_pattern_identifier_heartbeat: {
		uint16_t period = LED_SERVICE_HEARTBEAT_PERIOD_MS;
		uint16_t on_ms = LED_SERVICE_HEARTBEAT_PULSE_MS;
		uint16_t t = (uint16_t) (led_ptr->phase_ms % period);
		prv_led_apply_output(led_ptr, (t < on_ms));
	}
		break;

	case led_pattern_identifier_error_code: {
		/* Pattern: blink error_code_count times (on/off), then long pause */
		uint8_t n =
				(led_ptr->active_error_code_count == 0u) ?
						1u : led_ptr->active_error_code_count;
		uint32_t cycle_ms =
				(uint32_t) n
						* (LED_SERVICE_ERROR_ON_MS + LED_SERVICE_ERROR_OFF_MS)+ LED_SERVICE_ERROR_PAUSE_MS;
		uint32_t t = (uint32_t) (led_ptr->phase_ms % cycle_ms);

		/* Within the n blinks window? */
		uint32_t onoff_span_ms = (uint32_t) n
				* (LED_SERVICE_ERROR_ON_MS + LED_SERVICE_ERROR_OFF_MS);
		if (t < onoff_span_ms) {
			uint32_t segment = t
					% (LED_SERVICE_ERROR_ON_MS + LED_SERVICE_ERROR_OFF_MS);
			prv_led_apply_output(led_ptr, (segment < LED_SERVICE_ERROR_ON_MS));
		} else {
			prv_led_apply_output(led_ptr, false); /* pause */
		}
	}
		break;

	default:
		prv_led_apply_output(led_ptr, false);
		break;
	}

	/* Auto-revert on expiry */
	if (led_ptr->remaining_duration_ms == 0
			&& (led_ptr->active_pattern_identifier
					== led_pattern_identifier_blink
					|| led_ptr->active_pattern_identifier
							== led_pattern_identifier_error_code)) {
		/* Revert activity or transient errors to a safe default per LED */
		if (led_ptr->gpio_pin == LED1_Pin) {
			led_ptr->active_pattern_identifier =
					led_pattern_identifier_heartbeat;
		} else {
			led_ptr->active_pattern_identifier = led_pattern_identifier_off;
		}
		led_ptr->phase_ms = 0;
	}
}

static void prv_led_accept_command(const led_command_t *cmd) {
	led_runtime_t *led_ptr = &g_leds[cmd->led_identifier];

	/* Respect priority: only preempt if >= current priority */
	if (cmd->priority_level < led_ptr->active_priority_level) {
		return;
	}

	led_ptr->active_pattern_identifier = cmd->pattern_identifier;
	led_ptr->active_period_ms = cmd->period_ms;
	led_ptr->active_duty_percent = cmd->duty_percent;
	led_ptr->remaining_duration_ms = cmd->duration_ms;
	led_ptr->active_error_code_count = cmd->error_code_count;
	led_ptr->active_priority_level = cmd->priority_level;
	led_ptr->phase_ms = 0;
}

static void prv_led_apply_output(led_runtime_t *led_ptr, bool turn_on) {
	if (turn_on != led_ptr->is_output_on) {
		prv_gpio_write(led_ptr->gpio_port, led_ptr->gpio_pin,
				turn_on ? GPIO_PIN_SET : GPIO_PIN_RESET);
		led_ptr->is_output_on = turn_on;
	}
}



static void prv_led_service_task(void *argument) {
	(void) argument;
	const TickType_t tick_period = pdMS_TO_TICKS(LED_SERVICE_TASK_TICK_MS);
	TickType_t last_wake = xTaskGetTickCount();

	while (1) {
		/* Drain a few commands each cycle (bounded work) */
		for (uint8_t i = 0; i < 3; i++) {
			led_command_t cmd;
			if (xQueueReceive(g_led_command_queue_handle, &cmd, 0) == pdPASS) {
				prv_led_accept_command(&cmd);
			} else {
				break;
			}
		}

		/* Render onboard LEDs */
		/* LD1 (green): normal (heartbeat or whatever pattern is set) */
		prv_led_render_tick(&g_leds[led_identifier_ld1],
		LED_SERVICE_TASK_TICK_MS);

		/* LD2 (blue): rate-meter override — ON while credit remains */
		if (g_led_activity_credit_ms > 0u) {
			/* decay credit per tick */
			taskENTER_CRITICAL();
			if (g_led_activity_credit_ms > LED_SERVICE_ACTIVITY_CREDIT_DECAY_MS) {
				g_led_activity_credit_ms = (uint16_t) (g_led_activity_credit_ms
						- LED_SERVICE_ACTIVITY_CREDIT_DECAY_MS);
			} else {
				g_led_activity_credit_ms = 0u;
			}
			taskEXIT_CRITICAL();

			/* force ON (overrides any configured pattern while credit > 0) */
			prv_led_apply_output(&g_leds[led_identifier_ld2], true);
		} else {
			/* no recent activity -> render LD2’s configured pattern (usually OFF) */
			prv_led_render_tick(&g_leds[led_identifier_ld2],
			LED_SERVICE_TASK_TICK_MS);
		}

		/* LD3 (red): normal (off, coded error, etc.) */
		prv_led_render_tick(&g_leds[led_identifier_ld3],
		LED_SERVICE_TASK_TICK_MS);

		//end onboard LED renders

		/* External RGB pulse renderer: non-preemptive FIFO */
		if (g_rgb_flash_req.remaining_ms > 0u) {
		    /* keep showing the active color and decay its timer */
		    prv_rgb_apply_8bit(g_rgb_flash_req.r_8bit,
		                       g_rgb_flash_req.g_8bit,
		                       g_rgb_flash_req.b_8bit);

		    if (LED_SERVICE_TASK_TICK_MS >= g_rgb_flash_req.remaining_ms) {
		        g_rgb_flash_req.remaining_ms = 0u;
		    } else {
		        g_rgb_flash_req.remaining_ms =
		            (uint16_t)(g_rgb_flash_req.remaining_ms - LED_SERVICE_TASK_TICK_MS);
		    }
		} else {
		    /* no active pulse -> try to start the next queued one */
		    led_rgb_pulse_cmd_t next_cmd;
		    if (g_rgb_pulse_queue_handle != NULL &&
		        xQueueReceive(g_rgb_pulse_queue_handle, &next_cmd, 0) == pdPASS)
		    {
		        taskENTER_CRITICAL();
		        g_rgb_flash_req.r_8bit      = next_cmd.r_8bit;
		        g_rgb_flash_req.g_8bit      = next_cmd.g_8bit;
		        g_rgb_flash_req.b_8bit      = next_cmd.b_8bit;
		        g_rgb_flash_req.remaining_ms = (next_cmd.duration_ms == 0u) ? 1u : next_cmd.duration_ms;
		        taskEXIT_CRITICAL();

		        prv_rgb_apply_8bit(g_rgb_flash_req.r_8bit,
		                           g_rgb_flash_req.g_8bit,
		                           g_rgb_flash_req.b_8bit);
		    } else {
		        /* idle: ensure OFF */
		        prv_rgb_apply_8bit(0, 0, 0);
		    }
		}


		vTaskDelayUntil(&last_wake, tick_period);
	}
}


