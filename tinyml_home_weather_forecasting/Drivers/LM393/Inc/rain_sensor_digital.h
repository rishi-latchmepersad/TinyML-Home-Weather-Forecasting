#ifndef RAIN_SENSOR_DIGITAL_H
#define RAIN_SENSOR_DIGITAL_H

/*============================= Configuration =============================*/
/* Update these to match your wiring. Default: GPIOE pin 15. */
#define RAIN_DO_GPIO_PORT                     GPIOE
#define RAIN_DO_GPIO_PIN                      GPIO_PIN_15

/* Set to 1 if DO goes LOW when wet; 0 if HIGH when wet. */
#define RAIN_DO_ACTIVE_LOW_WHEN_WET           (1)

/* Debounce window for confirming wet/dry transitions (ms). */
#define RAIN_DO_DEBOUNCE_MS                   (150U)

/* Polling period for the task (ms). */
#define RAIN_DO_TASK_PERIOD_MS                (500U)

/* Enable periodic heartbeat printf (0 = off, 1 = on). */
#define RAIN_DO_ENABLE_HEARTBEAT_PRINTF       (0)

/* Heartbeat period (ms) – used only if heartbeat is enabled. */
#define RAIN_DO_HEARTBEAT_PERIOD_MS           (10000U)

/*=============================== Includes =================================*/
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*=============================== Types ====================================*/

/* Public wet/dry status returned by getters. */
typedef enum
{
    RAIN_DIGITAL_STATE_DRY = 0,
    RAIN_DIGITAL_STATE_WET
} rain_digital_state_t;

/*=============================== API ======================================*/

/**
 * @brief   Initialize the digital rain detector service (polling mode).
 * @return  true on success, false otherwise.
 *
 * Side Effects: Reads the input once to seed internal state.
 * Precondition: GPIO for DO configured as input (pull-up recommended).
 * Postcondition: Module internal state initialized; ready to start task().
 */
bool RainDigital_Service_Initialize(void);

/**
 * @brief   Return the latest debounced wet/dry state.
 * @return  RAIN_DIGITAL_STATE_DRY or RAIN_DIGITAL_STATE_WET.
 */
rain_digital_state_t RainDigital_Service_GetState(void);

/**
 * @brief   CMSIS-RTOS2 task: polls DO, debounces via FSM, prints events.
 * @param   argument (unused; pass NULL)
 *
 * Notes: Uses printf on transitions; optional heartbeat prints can be enabled
 *        via RAIN_DO_ENABLE_HEARTBEAT_PRINTF.
 */
void RainDigital_Service_Task(void *argument);

/**
 * @brief   EXTI edge notify hook (kept for link compatibility).
 * @param   gpio_pin  GPIO pin reported by HAL.
 *
 * Notes: In pure polling mode this is a stub; safe to call or ignore.
 */
void RainDigital_Service_EXTI_Notify(uint16_t gpio_pin);

#ifdef __cplusplus
}
#endif

#endif /* RAIN_SENSOR_DIGITAL_H */
