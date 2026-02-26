/*
 * retarget.c
 *
 *  Created on: Aug 20, 2025
 *      Author: rishi_latchmepersad
 */
#include "stm32f7xx_hal.h"
#include <stdio.h>
#include "cmsis_os.h"
#include "debug_log.h"

extern UART_HandleTypeDef huart3;
extern osMutexId_t printfMutexHandle;

/**
 * Redirect stdout and stderr to our UART3 debug.
 *
 * X-CUBE-AI's aiTestUtility.c provides its own _write() implementation when
 * UNITY_TEST builds are enabled. Guard this implementation to avoid duplicate
 * symbol errors at link time.
 */
#if !defined(UNITY_TEST)
int _write(int fd, char *ptr, int len) {
  if (!(fd == 1 || fd == 2)) return -1;

  // If RTOS not running yet, or mutex not created, transmit directly.
  if (osKernelGetState() != osKernelRunning || printfMutexHandle == NULL) {
    HAL_StatusTypeDef s = HAL_UART_Transmit(&huart3, (uint8_t*)ptr, len, HAL_MAX_DELAY);
    debug_log_capture(ptr, (size_t)len);
    return (s == HAL_OK) ? len : -1;
  }

  if (osMutexAcquire(printfMutexHandle, osWaitForever) == osOK) {
    HAL_StatusTypeDef s = HAL_UART_Transmit(&huart3, (uint8_t*)ptr, len, HAL_MAX_DELAY);
    debug_log_capture(ptr, (size_t)len);
    osMutexRelease(printfMutexHandle);
    return (s == HAL_OK) ? len : -1;
  }
  return -1;
}
#endif
