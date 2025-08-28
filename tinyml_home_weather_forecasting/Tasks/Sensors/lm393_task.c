#include "rain_sensor_digital.h"
#include <stdint.h>
#include <stdbool.h>
#include "cmsis_os2.h"

static const osThreadAttr_t g_rain_do_task_attr = { .name = "rain_sensor_task",
		.priority = osPriorityNormal, .stack_size = 768 };

void lm393_init_and_start_task(void) {
	(void) RainDigital_Service_Initialize();
	(void) osThreadNew(RainDigital_Service_Task, NULL, &g_rain_do_task_attr);
}

