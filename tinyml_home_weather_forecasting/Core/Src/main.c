/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : main.c
 * @brief          : Main program body
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2025 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "cmsis_os.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include "bme280_task.h"
#include "bme280_defs.h"
#include "stdbool.h"
#include "string.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
int _write(int fd, char *ptr, int len);
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

I2C_HandleTypeDef hi2c2;

SD_HandleTypeDef hsd1;

UART_HandleTypeDef huart3;

/* Definitions for defaultTask */
osThreadId_t defaultTaskHandle;
const osThreadAttr_t defaultTask_attributes = { .name = "defaultTask",
		.stack_size = 512 * 4, .priority = (osPriority_t) osPriorityNormal, };
/* USER CODE BEGIN PV */
osThreadId_t bme280SensorTaskHandle;
const osThreadAttr_t bme280SensorTask_attributes = { .name = "bme280SensorTask",
		.stack_size = 256 * 4, .priority = (osPriority_t) osPriorityNormal, };
osMutexId_t printfMutexHandle;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MPU_Config(void);
static void MX_GPIO_Init(void);
static void MX_I2C2_Init(void);
static void MX_SDMMC1_SD_Init(void);
static void MX_USART3_UART_Init(void);
void StartDefaultTask(void *argument);

/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
/**
 * Redirect stdout and stderr to our UART3 debug
 */
int _write(int fd, char *ptr, int len) {
	HAL_StatusTypeDef hstatus;
	int ret = -1;

	if (fd == 1 || fd == 2) {
		// Acquire mutex
		if (osMutexAcquire(printfMutexHandle, osWaitForever) == osOK) {
			hstatus = HAL_UART_Transmit(&huart3, (uint8_t*) ptr, len,
			HAL_MAX_DELAY);
			if (hstatus == HAL_OK) {
				ret = len;
			}
			// Release mutex
			osMutexRelease(printfMutexHandle);
		}
	}
	return ret;
}
/* USER CODE END 0 */

/**
 * @brief  The application entry point.
 * @retval int
 */
int main(void) {

	/* USER CODE BEGIN 1 */

	/* USER CODE END 1 */

	/* MPU Configuration--------------------------------------------------------*/
	MPU_Config();

	/* MCU Configuration--------------------------------------------------------*/

	/* Reset of all peripherals, Initializes the Flash interface and the Systick. */
	HAL_Init();

	/* USER CODE BEGIN Init */

	/* USER CODE END Init */

	/* Configure the system clock */
	SystemClock_Config();

	/* USER CODE BEGIN SysInit */

	/* USER CODE END SysInit */

	/* Initialize all configured peripherals */
	MX_GPIO_Init();
	MX_I2C2_Init();
	MX_SDMMC1_SD_Init();
	MX_USART3_UART_Init();
	/* USER CODE BEGIN 2 */
	printf("Test message via printf\n");  // Check if printf works

	// Direct UART test (bypass printf)
	const char *test_msg = "Direct UART test (PD8/PD9)\r\n";
	HAL_UART_Transmit(&huart3, (uint8_t*) test_msg, strlen(test_msg),
			HAL_MAX_DELAY);

	/* USER CODE END 2 */

	/* Init scheduler */
	osKernelInitialize();

	/* USER CODE BEGIN RTOS_MUTEX */
	printfMutexHandle = osMutexNew(NULL);
	if (printfMutexHandle == NULL) {
		Error_Handler();
	}
	/* USER CODE END RTOS_MUTEX */

	/* USER CODE BEGIN RTOS_SEMAPHORES */
	/* add semaphores, ... */
	/* USER CODE END RTOS_SEMAPHORES */

	/* USER CODE BEGIN RTOS_TIMERS */
	/* start timers, add new ones, ... */
	/* USER CODE END RTOS_TIMERS */

	/* USER CODE BEGIN RTOS_QUEUES */
	/* add queues, ... */
	/* USER CODE END RTOS_QUEUES */

	/* Create the thread(s) */
	/* creation of defaultTask */
	defaultTaskHandle = osThreadNew(StartDefaultTask, NULL,
			&defaultTask_attributes);

	/* USER CODE BEGIN RTOS_THREADS */
	/* add threads, ... */
	//bme280SensorTaskHandle = osThreadNew(bme280SensorTask, NULL,
	//		&bme280SensorTask_attributes);
	/* USER CODE END RTOS_THREADS */

	/* USER CODE BEGIN RTOS_EVENTS */
	/* add events, ... */
	/* USER CODE END RTOS_EVENTS */

	/* Start scheduler */
	osKernelStart();

	/* We should never get here as control is now taken by the scheduler */

	/* Infinite loop */
	/* USER CODE BEGIN WHILE */
	while (1) {
		/* USER CODE END WHILE */

		/* USER CODE BEGIN 3 */
	}
	/* USER CODE END 3 */
}

/**
 * @brief System Clock Configuration
 * @retval None
 */
void SystemClock_Config(void) {
	RCC_OscInitTypeDef RCC_OscInitStruct = { 0 };
	RCC_ClkInitTypeDef RCC_ClkInitStruct = { 0 };

	/** Configure the main internal regulator output voltage
	 */
	__HAL_RCC_PWR_CLK_ENABLE();
	__HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);

	/** Initializes the RCC Oscillators according to the specified parameters
	 * in the RCC_OscInitTypeDef structure.
	 */
	RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
	RCC_OscInitStruct.HSIState = RCC_HSI_ON;
	RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
	RCC_OscInitStruct.PLL.PLLState = RCC_PLL_NONE;
	if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) {
		Error_Handler();
	}

	/** Initializes the CPU, AHB and APB buses clocks
	 */
	RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK
			| RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
	RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_HSI;
	RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
	RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
	RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

	if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_0) != HAL_OK) {
		Error_Handler();
	}
}

/**
 * @brief I2C2 Initialization Function
 * @param None
 * @retval None
 */
static void MX_I2C2_Init(void) {

	/* USER CODE BEGIN I2C2_Init 0 */

	/* USER CODE END I2C2_Init 0 */

	/* USER CODE BEGIN I2C2_Init 1 */

	/* USER CODE END I2C2_Init 1 */
	hi2c2.Instance = I2C2;
	hi2c2.Init.Timing = 0x00303D5B;
	hi2c2.Init.OwnAddress1 = 0;
	hi2c2.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
	hi2c2.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
	hi2c2.Init.OwnAddress2 = 0;
	hi2c2.Init.OwnAddress2Masks = I2C_OA2_NOMASK;
	hi2c2.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
	hi2c2.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
	if (HAL_I2C_Init(&hi2c2) != HAL_OK) {
		Error_Handler();
	}

	/** Configure Analogue filter
	 */
	if (HAL_I2CEx_ConfigAnalogFilter(&hi2c2, I2C_ANALOGFILTER_ENABLE)
			!= HAL_OK) {
		Error_Handler();
	}

	/** Configure Digital filter
	 */
	if (HAL_I2CEx_ConfigDigitalFilter(&hi2c2, 0) != HAL_OK) {
		Error_Handler();
	}
	/* USER CODE BEGIN I2C2_Init 2 */

	/* USER CODE END I2C2_Init 2 */

}

/**
 * @brief SDMMC1 Initialization Function
 * @param None
 * @retval None
 */
static void MX_SDMMC1_SD_Init(void) {

	/* USER CODE BEGIN SDMMC1_Init 0 */

	/* USER CODE END SDMMC1_Init 0 */

	/* USER CODE BEGIN SDMMC1_Init 1 */

	/* USER CODE END SDMMC1_Init 1 */
	hsd1.Instance = SDMMC1;
	hsd1.Init.ClockEdge = SDMMC_CLOCK_EDGE_RISING;
	hsd1.Init.ClockBypass = SDMMC_CLOCK_BYPASS_DISABLE;
	hsd1.Init.ClockPowerSave = SDMMC_CLOCK_POWER_SAVE_DISABLE;
	hsd1.Init.BusWide = SDMMC_BUS_WIDE_1B;
	hsd1.Init.HardwareFlowControl = SDMMC_HARDWARE_FLOW_CONTROL_DISABLE;
	hsd1.Init.ClockDiv = 0;
	if (HAL_SD_Init(&hsd1) != HAL_OK) {
		Error_Handler();
	}
	if (HAL_SD_ConfigWideBusOperation(&hsd1, SDMMC_BUS_WIDE_4B) != HAL_OK) {
		Error_Handler();
	}
	/* USER CODE BEGIN SDMMC1_Init 2 */

	/* USER CODE END SDMMC1_Init 2 */

}

/**
 * @brief USART3 Initialization Function
 * @param None
 * @retval None
 */
static void MX_USART3_UART_Init(void) {

	/* USER CODE BEGIN USART3_Init 0 */

	/* USER CODE END USART3_Init 0 */

	/* USER CODE BEGIN USART3_Init 1 */

	/* USER CODE END USART3_Init 1 */
	huart3.Instance = USART3;
	huart3.Init.BaudRate = 115200;
	huart3.Init.WordLength = UART_WORDLENGTH_8B;
	huart3.Init.StopBits = UART_STOPBITS_1;
	huart3.Init.Parity = UART_PARITY_NONE;
	huart3.Init.Mode = UART_MODE_TX_RX;
	huart3.Init.HwFlowCtl = UART_HWCONTROL_NONE;
	huart3.Init.OverSampling = UART_OVERSAMPLING_16;
	huart3.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
	huart3.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
	if (HAL_UART_Init(&huart3) != HAL_OK) {
		Error_Handler();
	}
	/* USER CODE BEGIN USART3_Init 2 */

	/* USER CODE END USART3_Init 2 */

}

/**
 * @brief GPIO Initialization Function
 * @param None
 * @retval None
 */
static void MX_GPIO_Init(void) {
	GPIO_InitTypeDef GPIO_InitStruct = { 0 };
	/* USER CODE BEGIN MX_GPIO_Init_1 */

	/* USER CODE END MX_GPIO_Init_1 */

	/* GPIO Ports Clock Enable */
	__HAL_RCC_GPIOB_CLK_ENABLE();
	__HAL_RCC_GPIOD_CLK_ENABLE();
	__HAL_RCC_GPIOC_CLK_ENABLE();
	__HAL_RCC_GPIOA_CLK_ENABLE();

	/*Configure GPIO pin : Card_Detect_Pin */
	GPIO_InitStruct.Pin = Card_Detect_Pin;
	GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
	GPIO_InitStruct.Pull = GPIO_PULLDOWN;
	HAL_GPIO_Init(Card_Detect_GPIO_Port, &GPIO_InitStruct);

	/* USER CODE BEGIN MX_GPIO_Init_2 */

	/* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */
void DebugSDWiring(void) {
	printf("=== SD Wiring Debug ===\n");

	// Check power supply to SD card
	printf("1. Check SD card power supply (should be 3.3V)\n");

	// Test each SDMMC pin individually
	printf("2. Testing SDMMC pin configurations...\n");

	// Get GPIO pin states
	GPIO_TypeDef *ports[] = { GPIOA, GPIOB, GPIOC, GPIOD };
	char *port_names[] = { "GPIOA", "GPIOB", "GPIOC", "GPIOD" };

	for (int i = 0; i < 4; i++) {
		uint16_t idr = ports[i]->IDR;  // Input data register
		uint16_t odr = ports[i]->ODR;  // Output data register
		printf("%s IDR: 0x%04X, ODR: 0x%04X\n", port_names[i], idr, odr);
	}

	printf("3. Testing your SDMMC pins:\n");
	printf("PC12 (CLK): %d\n", HAL_GPIO_ReadPin(GPIOC, GPIO_PIN_12));
	printf("PD2 (CMD): %d\n", HAL_GPIO_ReadPin(GPIOD, GPIO_PIN_2));
	printf("PC8 (D0): %d\n", HAL_GPIO_ReadPin(GPIOC, GPIO_PIN_8));
	printf("PC9 (D1): %d\n", HAL_GPIO_ReadPin(GPIOC, GPIO_PIN_9));
	printf("PC10 (D2): %d\n", HAL_GPIO_ReadPin(GPIOC, GPIO_PIN_10));
	printf("PC11 (D3): %d\n", HAL_GPIO_ReadPin(GPIOC, GPIO_PIN_11));
	printf("PB13 (DET): %d\n", HAL_GPIO_ReadPin(GPIOB, GPIO_PIN_13));

	// Check if pins are configured as alternate function
	printf("4. Checking pin configurations:\n");
	printf("PC12 MODER: 0x%lX (should be 0x2 for AF)\n",
			(GPIOC->MODER >> (12 * 2)) & 0x3);
	printf("PD2 MODER: 0x%lX (should be 0x2 for AF)\n",
			(GPIOD->MODER >> (2 * 2)) & 0x3);
	printf("PC8 MODER: 0x%lX (should be 0x2 for AF)\n",
			(GPIOC->MODER >> (8 * 2)) & 0x3);

	// Check alternate function settings (should be AF12 for SDMMC1)
	printf("PC12 AFR: 0x%lX (should be 0xC for SDMMC1)\n",
			(GPIOC->AFR[1] >> ((12 - 8) * 4)) & 0xF);
	printf("PD2 AFR: 0x%lX (should be 0xC for SDMMC1)\n",
			(GPIOD->AFR[0] >> (2 * 4)) & 0xF);
	printf("PC8 AFR: 0x%lX (should be 0xC for SDMMC1)\n",
			(GPIOC->AFR[1] >> ((8 - 8) * 4)) & 0xF);
}

void TestSDBasicCommunication(void) {
	printf("=== Testing Basic SD Communication ===\n");

	// Try to get more detailed error information
	uint32_t errorstate = HAL_SD_GetError(&hsd1);
	printf("SD Error State: 0x%08lX\n", errorstate);

	// Check SDMMC peripheral registers
	printf("SDMMC1->POWER: 0x%08lX\n", SDMMC1->POWER);
	printf("SDMMC1->CLKCR: 0x%08lX\n", SDMMC1->CLKCR);
	printf("SDMMC1->STA: 0x%08lX\n", SDMMC1->STA);

	// Try manual power cycle
	printf("Attempting manual SD power cycle...\n");

	// Power off
	SDMMC1->POWER &= ~SDMMC_POWER_PWRCTRL;
	HAL_Delay(100);

	// Power on
	SDMMC1->POWER |= SDMMC_POWER_PWRCTRL;
	HAL_Delay(100);

	printf("After power cycle - STA: 0x%08lX\n", SDMMC1->STA);
}

void DebugSDMMCClocks(void) {
	printf("=== SDMMC Clock Debug ===\n");

	// Check if SDMMC1 peripheral clock is enabled
	uint32_t ahb2enr = RCC->APB2ENR;
	printf("RCC->AHB2ENR: 0x%08lX\n", ahb2enr);
	printf("SDMMC1 Clock Enabled: %s\n",
			(ahb2enr & RCC_APB2ENR_SDMMC1EN) ? "YES" : "NO");

	// Check GPIO clocks
	uint32_t ahb1enr = RCC->AHB1ENR;
	printf("GPIOC Clock: %s\n", (ahb1enr & RCC_AHB1ENR_GPIOCEN) ? "YES" : "NO");
	printf("GPIOD Clock: %s\n", (ahb1enr & RCC_AHB1ENR_GPIODEN) ? "YES" : "NO");
	printf("GPIOB Clock: %s\n", (ahb1enr & RCC_AHB1ENR_GPIOBEN) ? "YES" : "NO");

	// Check SDMMC1 clock source
	uint32_t dckcfgr2 = RCC->DCKCFGR2;
	printf("RCC->DCKCFGR2: 0x%08lX\n", dckcfgr2);
	uint32_t sdmmc1sel = (dckcfgr2 & RCC_DCKCFGR2_SDMMC1SEL)
			>> RCC_DCKCFGR2_SDMMC1SEL_Pos;
	printf("SDMMC1 Clock Source: %s\n", sdmmc1sel ? "SYSCLK" : "48MHz");

	// Manual clock enable if needed
	if (!(ahb2enr & RCC_APB2ENR_SDMMC1EN)) {
		printf("Enabling SDMMC1 clock manually...\n");
		__HAL_RCC_SDMMC1_CLK_ENABLE();
		printf("SDMMC1 clock enabled.\n");
	}
}

// Enhanced SD initialization with proper clock setup
HAL_StatusTypeDef InitSDWithClockFix(void) {
	printf("=== Enhanced SD Initialization ===\n");

	// Ensure clocks are enabled
	__HAL_RCC_SDMMC1_CLK_ENABLE();
	__HAL_RCC_GPIOC_CLK_ENABLE();
	__HAL_RCC_GPIOD_CLK_ENABLE();
	__HAL_RCC_GPIOB_CLK_ENABLE();

	// Force SDMMC1 reset and release
	__HAL_RCC_SDMMC1_FORCE_RESET();
	HAL_Delay(10);
	__HAL_RCC_SDMMC1_RELEASE_RESET();
	HAL_Delay(10);

	printf("Clocks enabled, attempting SD init...\n");

	// Re-initialize the SD peripheral
	HAL_StatusTypeDef status = HAL_SD_Init(&hsd1);
	printf("HAL_SD_Init result: %d\n", status);

	if (status == HAL_OK) {
		printf("SDMMC1->POWER after init: 0x%08lX\n", SDMMC1->POWER);
		printf("SDMMC1->CLKCR after init: 0x%08lX\n", SDMMC1->CLKCR);
	} else {
		uint32_t error = HAL_SD_GetError(&hsd1);
		printf("SD Error: 0x%08lX\n", error);
	}

	return status;
}

// Alternative: Check if we need 48MHz clock
void Setup48MHzClock(void) {
	printf("Setting up 48MHz clock for SDMMC...\n");

	RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = { 0 };

	// Configure 48MHz clock (usually from PLL48M)
	PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_SDMMC1;
	PeriphClkInitStruct.Sdmmc1ClockSelection = RCC_SDMMC1CLKSOURCE_CLK48;

	if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK) {
		printf("Failed to configure SDMMC1 clock!\n");
	} else {
		printf("SDMMC1 configured for 48MHz clock.\n");
	}
}
/* USER CODE END 4 */

/* USER CODE BEGIN Header_StartDefaultTask */
/**
 * @brief  Function implementing the defaultTask thread.
 * @param  argument: Not used
 * @retval None
 */
/* USER CODE END Header_StartDefaultTask */
void StartDefaultTask(void *argument) {
	/* USER CODE BEGIN 5 */
	printf("starting default task.\n");
	printf("Starting SD card debug...\n");

	// Check card detection first
	if (HAL_GPIO_ReadPin(Card_Detect_GPIO_Port, Card_Detect_Pin)
			== GPIO_PIN_SET) {
		printf("SD card detected.\n");
	} else {
		printf("SD card not detected. Check card insertion and wiring.\n");
		for (;;)
			osDelay(1000);
	}

	// Debug wiring and pin states
	DebugSDWiring();

	// Test basic communication
	TestSDBasicCommunication();

	// Debug and fix clock issues
	DebugSDMMCClocks();

	// Try enhanced SD initialization
	printf("Attempting enhanced SD initialization...\n");

	HAL_StatusTypeDef status = InitSDWithClockFix();
	printf("HAL_SD_Init result: %d\n", status);

	if (status != HAL_OK) {
		uint32_t errorstate = HAL_SD_GetError(&hsd1);
		printf("Detailed SD Error: 0x%08lX\n", errorstate);

		// Decode common error flags
		if (errorstate & HAL_SD_ERROR_CMD_CRC_FAIL)
			printf("- CMD CRC Failed\n");
		if (errorstate & HAL_SD_ERROR_CMD_RSP_TIMEOUT)
			printf("- CMD Response Timeout\n");
		if (errorstate & HAL_SD_ERROR_DATA_CRC_FAIL)
			printf("- Data CRC Failed\n");
		if (errorstate & HAL_SD_ERROR_DATA_TIMEOUT)
			printf("- Data Timeout\n");
		if (errorstate & HAL_SD_ERROR_ADDR_MISALIGNED)
			printf("- Address Misaligned\n");
		if (errorstate & HAL_SD_ERROR_BLOCK_LEN_ERR)
			printf("- Block Length Error\n");
		if (errorstate & HAL_SD_ERROR_WRITE_PROT_VIOLATION)
			printf("- Write Protection Violation\n");

		printf("Check your wiring connections!\n");
		for (;;)
			osDelay(1000);
	}

	printf("SD initialization successful!\n");

	// Continue with rest of tests...
	for (;;) {
		osDelay(5000);
		printf("Debug task running...\n");
	}
	/* USER CODE END 5 */
}

/* MPU Configuration */

void MPU_Config(void) {
	MPU_Region_InitTypeDef MPU_InitStruct = { 0 };

	/* Disables the MPU */
	HAL_MPU_Disable();

	/** Initializes and configures the Region and the memory to be protected
	 */
	MPU_InitStruct.Enable = MPU_REGION_ENABLE;
	MPU_InitStruct.Number = MPU_REGION_NUMBER0;
	MPU_InitStruct.BaseAddress = 0x0;
	MPU_InitStruct.Size = MPU_REGION_SIZE_4GB;
	MPU_InitStruct.SubRegionDisable = 0x87;
	MPU_InitStruct.TypeExtField = MPU_TEX_LEVEL0;
	MPU_InitStruct.AccessPermission = MPU_REGION_NO_ACCESS;
	MPU_InitStruct.DisableExec = MPU_INSTRUCTION_ACCESS_DISABLE;
	MPU_InitStruct.IsShareable = MPU_ACCESS_SHAREABLE;
	MPU_InitStruct.IsCacheable = MPU_ACCESS_NOT_CACHEABLE;
	MPU_InitStruct.IsBufferable = MPU_ACCESS_NOT_BUFFERABLE;

	HAL_MPU_ConfigRegion(&MPU_InitStruct);
	/* Enables the MPU */
	HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);

}

/**
 * @brief  Period elapsed callback in non blocking mode
 * @note   This function is called  when TIM14 interrupt took place, inside
 * HAL_TIM_IRQHandler(). It makes a direct call to HAL_IncTick() to increment
 * a global variable "uwTick" used as application time base.
 * @param  htim : TIM handle
 * @retval None
 */
void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim) {
	/* USER CODE BEGIN Callback 0 */

	/* USER CODE END Callback 0 */
	if (htim->Instance == TIM14) {
		HAL_IncTick();
	}
	/* USER CODE BEGIN Callback 1 */

	/* USER CODE END Callback 1 */
}

/**
 * @brief  This function is executed in case of error occurrence.
 * @retval None
 */
void Error_Handler(void) {
	/* USER CODE BEGIN Error_Handler_Debug */
	/* User can add his own implementation to report the HAL error return state */
	__disable_irq();
	while (1) {
	}
	/* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
