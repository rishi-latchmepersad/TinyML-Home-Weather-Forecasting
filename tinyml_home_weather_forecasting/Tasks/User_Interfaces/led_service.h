/**
 * \file       led_service.h
 * \brief      User-LED UI service (Nucleo LD1/LD2/LD3) — health, activity, faults.
 *
 * \purpose    Centralized, thread-safe control of three user LEDs with readable,
 *             field-friendly patterns. Owns LED GPIO and renders patterns in a
 *             low-priority task without busy-waiting.
 *
 * \hardware   LD1 (green): PB0 or PA5 (Zio D13)  [Prefer PB0 if using SPI1]
 *             LD2 (blue) : PB7
 *             LD3 (red)  : PB14
 *             Solder bridges: SB120 ON / SB119 OFF → LD1 on PB0.
 *             We also use a HW-479 arduino RGB LED board
 *             B (blue): PE13
 *             G (green): PE11
 *             R (red): PE9
 *
 * \patterns   - Health (LD1): Heartbeat — 2 s period, ~500 ms pulse (default).
 *             - Activity (LD2): Single short blink per event via led_service_activity_bump(200);
 *                               for long transfers or other significant events use something like led_service_activity_bump(1000);
 *             - Faults (LD3): Coded N-blink with pause (1=sensor, 2=storage, 3=network).
 *                             Solid ON indicates fatal/latched safe mode until cleared.
 *             - Sensors (HW-479): Used to make the sensor activity more intuitive. We can add a color for each sensor.
 *             	 - BME280: Purple
 *             	 - VEML7700: Teal
 *
 * \usage      bool ok = led_service_init();
 *             ok &= led_service_start(osPriorityLow1, 256*4);
 *             // activity: led_service_pulse_activity();
 *             // errors:   led_service_set_pattern(&cmd_with_error_code);
 *
 * \timing     Service updates every 20 ms; no HAL_Delay() used in the task.
 * \concurrency
 *             - Single writer to GPIO (this service task).
 *             - Thread-safe non-blocking API; ISR-safe variant provided.
 *
 * \cautions   - Do not use HAL_Delay() inside RTOS tasks (use vTaskDelay/osDelay).
 *             - Ensure no peripheral steals PB7 (I2C1) or PA5 (SPI1_SCK) if selected.
 */

#pragma once
#include "stm32f7xx_hal.h"
#include "stm32f7xx_hal_tim.h"
#include "FreeRTOS.h"
#include "queue.h"
#include "task.h"
#include <stdbool.h>
#include <stdint.h>
/* ===== Board mapping: choose LD1 pin based on solder bridges SB119/SB120 ===== */
// Define exactly one of these based on your board solder bridge setting.
#define LED1_ON_PB0   1
/* #define LED1_ON_PA5   1 */

#if defined(LED1_ON_PB0)
#define LED1_GPIO_Port  GPIOB
#define LED1_Pin        GPIO_PIN_0
#elif defined(LED1_ON_PA5)
  #define LED1_GPIO_Port  GPIOA
  #define LED1_Pin        GPIO_PIN_5
#else
  #error "Select LD1 location: define LED1_ON_PB0 or LED1_ON_PA5"
#endif

#define LED2_GPIO_Port  GPIOB
#define LED2_Pin        GPIO_PIN_7     /* Blue */
#define LED3_GPIO_Port  GPIOB
#define LED3_Pin        GPIO_PIN_14    /* Red */

typedef enum {
	led_identifier_ld1 = 0, //green
	led_identifier_ld2 = 1, //blue
	led_identifier_ld3 = 2 //red
} led_identifier_t;

typedef enum {
	led_pattern_identifier_off = 0,
	led_pattern_identifier_on,
	led_pattern_identifier_blink, /* uses period_ms and duty_percent */
	led_pattern_identifier_heartbeat, /* fixed nice “pulse” */
	led_pattern_identifier_error_code /* blinks N times, long pause */
} led_pattern_identifier_t;

typedef struct {
	led_identifier_t led_identifier;
	led_pattern_identifier_t pattern_identifier;
	uint16_t period_ms; /* for blink */
	uint8_t duty_percent; /* 1..99     */
	uint16_t duration_ms; /* 0 = forever */
	uint8_t error_code_count; /* for error_code pattern */
	uint8_t priority_level; /* 0=low (default), larger preempts */
} led_command_t;

/* ===================== Public API ===================== */

/**
 * \brief   Initialize the LED service module GPIO and queue, but do not start the task.
 *
 * \param   none
 * \return  bool - true if initialization completed successfully; false otherwise.
 *
 * \sideeffects  Configures GPIO clocks and output modes for LD1/LD2/LD3.
 * \pre          HAL must be initialized. FreeRTOS not required yet.
 * \post         LED outputs are driven to OFF.
 * \concurrency  Not thread-safe; call once during system init.
 * \timing       O(1) except for HAL GPIO init latency.
 * \errors       Returns false if queue allocation fails.
 * \notes        Choose LED1 pin mapping via compile-time defines above.
 */
bool led_service_init(void);

/**
 * \brief   Create and start the LED service task.
 *
 * \param   task_priority_uxprio    [in] FreeRTOS task priority.
 * \param   task_stack_words        [in] Stack size in words (not bytes).
 * \return  bool - true if the task started successfully; false otherwise.
 *
 * \sideeffects  Spawns a FreeRTOS task that owns all LED GPIO updates.
 * \pre          led_service_init() completed successfully.
 * \post         Heartbeat on LD1 is enabled by default.
 * \concurrency  Thread-safe; call once from initialization context.
 * \timing       O(1) to create the task.
 * \errors       Returns false if task creation fails.
 * \notes        Recommended: low priority and ~10–20 ms tick period inside the task.
 */
bool led_service_start(UBaseType_t task_priority_uxprio,
		uint16_t task_stack_words);

/**
 * \brief   Request a pattern be rendered on the given LED.
 *
 * \param   command_ptr  [in] Pointer to fully-populated command structure.
 * \return  bool - true if the command was queued; false if the queue was full.
 *
 * \sideeffects  None (deferred to the LED task).
 * \pre          led_service_start() has been called.
 * \post         Pattern will take effect when dequeued; may preempt if priority is higher.
 * \concurrency  Thread-safe; ISR-safe when used via led_service_set_pattern_from_isr().
 * \timing       O(1) enqueue.
 * \errors       Returns false if queue is full.
 * \notes        For simple activity blinks, see led_service_pulse_activity().
 */
bool led_service_set_pattern(const led_command_t *command_ptr);

/**
 * \brief   ISR-safe variant to set a pattern from an interrupt handler.
 *
 * \param   command_ptr  [in] Pointer to command (resides in caller's stack/const).
 * \return  bool - true on success; false if the queue was full.
 *
 * \sideeffects  None.
 * \pre          ISR context only when you cannot block.
 * \post         Yields at ISR exit if a higher-priority task was woken.
 * \concurrency  ISR-safe.
 * \timing       O(1).
 * \errors       Queue full -> returns false.
 * \notes        Use for quick “activity” indications triggered by peripherals.
 */
bool led_service_set_pattern_from_isr(const led_command_t *command_ptr);

/**
 * \brief   Convenience: short activity blink on the blue LED (LD2).
 *
 * \param   none
 * \return  void
 *
 * \sideeffects  Enqueues a brief blink that auto-expires, priority low.
 * \pre          led_service_start() has been called.
 * \post         A 80ms ON / 200ms off pulse on LD2, does not preempt errors.
 * \concurrency  Thread-safe.
 * \timing       O(1).
 * \errors       Silently drops if queue is full.
 * \notes        Call whenever you log data or finish an IO packet.
 */
void led_service_pulse_activity(void);

/**
 * \brief   Add activity credit to the blue LED (LD2). LED stays ON while credit > 0.
 *
 * \param   credit_ms   Milliseconds of credit to add (e.g., 150..400).
 * \return  void
 *
 * \sideeffects  Extends a shared activity timer used by the LED task.
 * \pre          led_service_start() has been called.
 * \post         Blue LED is forced ON while any credit remains.
 * \concurrency  Task-context only (not ISR-safe).
 * \timing       O(1).
 * \errors       None.
 * \notes        Call at the end of each measurement and/or after each log write.
 */
void led_service_activity_bump(uint16_t credit_ms);

/* ***************************************************************************
 * Function: led_service_rgb_board_init
 * Purpose : Initialize PWM outputs for an external RGB LED module.
 * Params  : timer_handle_ptr  - pointer to the configured TIM instance (e.g., &htim3)
 *           channel_red       - timer channel used for RED   (e.g., TIM_CHANNEL_1)
 *           channel_green     - timer channel used for GREEN (e.g., TIM_CHANNEL_2)
 *           channel_blue      - timer channel used for BLUE  (e.g., TIM_CHANNEL_4)
 *           is_common_anode   - true for CA, false for CC (HW-479 is false)
 * Returns : bool - true on success.
 * Side effects : Starts PWM on the three channels (consumes those pins).
 * Notes   : Timer base + PWM channels must be CubeMX-configured. Target ~0.5–2 kHz.
 * ***************************************************************************/
bool led_service_rgb_board_init(TIM_HandleTypeDef *timer_handle_ptr,
		uint32_t channel_red, uint32_t channel_green, uint32_t channel_blue,
		bool is_common_anode);

/* ***************************************************************************
 * Function: led_service_pulse_activity_rgb
 * Purpose : Flash the RGB LED a requested color for a bounded duration, then turn it off.
 * Params  : red_8bit, green_8bit, blue_8bit - 0..255 brightness per channel
 *           duration_ms                     - flash length in milliseconds
 * Returns : void
 * Side effects : Updates a shared request consumed by the LED service task each tick.
 * Notes   : Non-blocking; safe to call from normal task context. For ISR use, create an _from_isr variant.
 * ***************************************************************************/
void led_service_pulse_activity_rgb(uint8_t red_8bit, uint8_t green_8bit,
		uint8_t blue_8bit, uint16_t duration_ms);

