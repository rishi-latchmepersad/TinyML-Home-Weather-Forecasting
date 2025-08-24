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
	//light up the LED;
	error_indicator_red_led_solid_on();
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
 * \brief   Drive the red user LED (LD3 on PB14) to a solid ON state immediately.
 *
 * \param   none
 * \return  void
 *
 * \sideeffects  Enables GPIOB clock, configures PB14 as push-pull output, sets it HIGH.
 * \pre          None; safe to call from Error_Handler or before RTOS starts.
 * \post         Red LED is visibly ON.
 * \concurrency  Not thread-safe by design; intended for fault/critical paths.
 * \timing       O(1) except for HAL GPIO init latency.
 * \errors       None.
 * \notes        Does not rely on the LED service task or any RTOS primitives.
 */
void error_indicator_red_led_solid_on(void)
{
    __HAL_RCC_GPIOB_CLK_ENABLE();

    GPIO_InitTypeDef gpio_init_structure = {0};
    gpio_init_structure.Pin   = GPIO_PIN_14;                 /* LD3 */
    gpio_init_structure.Mode  = GPIO_MODE_OUTPUT_PP;
    gpio_init_structure.Pull  = GPIO_NOPULL;
    gpio_init_structure.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOB, &gpio_init_structure);

    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_14, GPIO_PIN_SET);     /* active high on Nucleo */
}
