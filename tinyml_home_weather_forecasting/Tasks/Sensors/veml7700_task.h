#pragma once
#include "stm32f7xx_hal.h"
#include "cmsis_os2.h"
#include <stdint.h>
#include <stdbool.h>

/* ===== Public config defaults (good indoor range without saturating) ===== */
#define VEML7700_GAIN_CODE_1X     (0x0u)
#define VEML7700_GAIN_CODE_2X     (0x1u)
#define VEML7700_GAIN_CODE_1_8X   (0x2u)
#define VEML7700_GAIN_CODE_1_4X   (0x3u)

#define VEML7700_IT_CODE_100MS    (0x0u)
#define VEML7700_IT_CODE_200MS    (0x1u)
#define VEML7700_IT_CODE_400MS    (0x2u)
#define VEML7700_IT_CODE_800MS    (0x3u)
#define VEML7700_IT_CODE_50MS     (0x8u)
#define VEML7700_IT_CODE_25MS     (0xCu)

/* Change these if you want a different operating point */
#ifndef VEML7700_DEFAULT_GAIN_CODE
#define VEML7700_DEFAULT_GAIN_CODE   VEML7700_GAIN_CODE_1_8X
#endif
#ifndef VEML7700_DEFAULT_IT_CODE
#define VEML7700_DEFAULT_IT_CODE     VEML7700_IT_CODE_100MS
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief Start the VEML7700 reader as a CMSIS-RTOS2 thread.
 * \param hi2c             Pointer to I2C handle (use &hi2c4)
 * \param prio             Thread priority (e.g., osPriorityLow/Normal)
 * \param stack_words      Stack size in 32-bit words (e.g., 512)
 * \return osThreadId_t    Thread ID (NULL on failure)
 */
osThreadId_t veml7700_start_task(I2C_HandleTypeDef *hi2c,
                                 osPriority_t prio,
                                 uint32_t stack_words);

/**
 * \brief Get the most recent readings captured by the task.
 * \param lux_out          [out] approximate lux (may be 0 on first read)
 * \param als_counts_out   [out] raw ALS counts
 * \param white_counts_out [out] raw “white” counts
 * \return bool            true if values are valid
 */
bool veml7700_get_latest(float *lux_out, uint16_t *als_counts_out, uint16_t *white_counts_out);

#ifdef __cplusplus
}
#endif
