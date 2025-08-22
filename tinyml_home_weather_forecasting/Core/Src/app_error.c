/*
 * app_error.c
 *
 *  Created on: Aug 21, 2025
 *      Author: rishi_latchmepersad
 */


/**
 * @brief This error function prints an error to the console and resets the system
 * @retval None
 */

#include "stm32f7xx_hal.h"
#include <stdio.h>
#include "app_error.h"
#include "cmsis_os.h"


__attribute__((noreturn)) //help the compiler optimize the no-return path
void error_handler_with_message(const char *msg) {
	// if this function is called with a blank message, just show the default
	const char *text = (msg && msg[0] != '\0') ? msg : ERROR_DEFAULT_MSG;
	//print the error message
	printf("\r\n[ERROR] %s \r\n", text);
	printf("Rebooting in 5s.\r\n");
	fflush(stdout); //clear the buffer to ensure that the message is printed
	// if we aren't in an ISR, then wait 5s
	if (__get_IPSR() == 0U) {
		HAL_Delay(5000);
	}
	//call the data synchronization barrier so that all pending bus writes can be committed
	__DSB();
	//call the instruction synchronization barrier to flush the CPU pipeline and re-fetch fresh instructions
	__ISB();
	//reboot the system
	NVIC_SystemReset();
	//set up an empty infinite loop to ensure we don't return
	while (1) {

	}
}
/**
 * @brief This is the default error function
 * @retval None
 */
void Error_Handler(void) {
	/* USER CODE BEGIN Error_Handler_Debug */
	// Minimal wrapper so HAL calls land here too.
	printf("\r\n[ERROR] %s\r\n", ERROR_DEFAULT_MSG);
	fflush(stdout);

	if (__get_IPSR() == 0U) {
		HAL_Delay(5000);
	}

	__DSB();
	__ISB();
	NVIC_SystemReset();
	while (1) { /* no return */
	}
	/* USER CODE END Error_Handler_Debug */
}
