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
#include "fatfs.h"

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
/**
 * Redirect stdout and stderr to our UART3 debug
 */
UART_HandleTypeDef huart3;
int _write(int fd, char *ptr, int len) {
	HAL_StatusTypeDef hstatus;

	if (fd == 1 || fd == 2) {
		hstatus = HAL_UART_Transmit(&huart3, (uint8_t*) ptr, len,
		HAL_MAX_DELAY);
		if (hstatus == HAL_OK)
			return len;
		else
			return -1;
	}
	return -1;
}
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

I2C_HandleTypeDef hi2c2;

SPI_HandleTypeDef hspi1;

UART_HandleTypeDef huart3;

/* Definitions for defaultTask */
osThreadId_t defaultTaskHandle;
const osThreadAttr_t defaultTask_attributes = { .name = "defaultTask",
		.stack_size = 1024 * 4, .priority = (osPriority_t) osPriorityNormal, };
/* USER CODE BEGIN PV */
osThreadId_t bme280SensorTaskHandle;
const osThreadAttr_t bme280SensorTask_attributes = { .name = "bme280SensorTask",
		.stack_size = 256 * 4, .priority = (osPriority_t) osPriorityNormal, };
FATFS fs;  // File system object
FIL fil;   // File object
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MPU_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART3_UART_Init(void);
static void MX_SPI1_Init(void);
static void MX_I2C2_Init(void);
void StartDefaultTask(void *argument);

/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

void test_basic_spi(void) {
	uint8_t tx = 0x55;
	uint8_t rx = 0;

	printf("Testing basic SPI...\n");

	// Manually control CS
	HAL_GPIO_WritePin(GPIOB, GPIO_PIN_6, GPIO_PIN_RESET);
	HAL_Delay(1);

	HAL_StatusTypeDef status = HAL_SPI_TransmitReceive(&hspi1, &tx, &rx, 1,
	HAL_MAX_DELAY);

	HAL_GPIO_WritePin(GPIOB, GPIO_PIN_6, GPIO_PIN_SET);

	printf("SPI status: %d, Sent: 0x55, Received: 0x%02X\n", status, rx);
}

void Test_SD_Card() {
	printf("\n=== SD Card Initialization ===\n");

	// 1. Power-up sequence - ensure CS is high for at least 1ms
	HAL_GPIO_WritePin(GPIOB, GPIO_PIN_6, GPIO_PIN_SET); // CS HIGH
	HAL_Delay(100); // Power stabilization delay

	// Send 80+ clocks (10 bytes of 0xFF) with CS high
	uint8_t dummy = 0xFF;
	for (int i = 0; i < 10; i++) {
		HAL_SPI_Transmit(&hspi1, &dummy, 1, 100);
	}
	HAL_Delay(10);

	// 2. Send CMD0 (GO_IDLE_STATE) with proper timing
	uint8_t response;
	int attempts = 0;

	do {
		// Select card (CS low)
		HAL_GPIO_WritePin(GPIOB, GPIO_PIN_6, GPIO_PIN_RESET);
		HAL_Delay(1); // Short delay after CS assertion

		// Send CMD0: 0x40, 0x00, 0x00, 0x00, 0x00, 0x95
		uint8_t cmd0[] = { 0x40, 0x00, 0x00, 0x00, 0x00, 0x95 };
		HAL_SPI_Transmit(&hspi1, cmd0, 6, 100);

		// Wait for response (R1) - should be 0x01
		// SD cards can take up to 8 bytes to respond
		response = 0xFF;
		for (int i = 0; i < 8; i++) {
			HAL_SPI_TransmitReceive(&hspi1, &dummy, &response, 1, 100);
			if (response != 0xFF)
				break;
		}

		// Deselect card (CS high)
		HAL_GPIO_WritePin(GPIOB, GPIO_PIN_6, GPIO_PIN_SET);

		// Send at least one more clock cycle after deselecting
		HAL_SPI_Transmit(&hspi1, &dummy, 1, 100);
		HAL_Delay(10);

		printf("CMD0 attempt %d: 0x%02X\n", ++attempts, response);

		if (response == 0x01) {
			printf("CMD0 success (0x01 response)\n");
			break;
		}

		if (attempts >= 10) {
			printf("Failed to get valid CMD0 response after %d attempts\n",
					attempts);
			return;
		}

		HAL_Delay(100); // Wait before retry
	} while (response != 0x01);

	// 3. Send CMD8 to check voltage compatibility (SDC V2)
	HAL_GPIO_WritePin(GPIOB, GPIO_PIN_6, GPIO_PIN_RESET); // CS LOW
	HAL_Delay(1);

	uint8_t cmd8[] = { 0x48, 0x00, 0x00, 0x01, 0xAA, 0x87 }; // 2.7-3.6V, check pattern 0xAA
	HAL_SPI_Transmit(&hspi1, cmd8, 6, 100);

	// Wait for R1 response first
	uint8_t r1_response = 0xFF;
	for (int i = 0; i < 8; i++) {
		HAL_SPI_TransmitReceive(&hspi1, &dummy, &r1_response, 1, 100);
		if (r1_response != 0xFF)
			break;
	}

	// Read remaining 4 bytes of R7 response
	uint8_t r7_data[4];
	for (int i = 0; i < 4; i++) {
		HAL_SPI_TransmitReceive(&hspi1, &dummy, &r7_data[i], 1, 100);
	}

	HAL_GPIO_WritePin(GPIOB, GPIO_PIN_6, GPIO_PIN_SET); // CS HIGH
	HAL_SPI_Transmit(&hspi1, &dummy, 1, 100);

	printf("CMD8 R1: 0x%02X, R7 data: %02X %02X %02X %02X\n", r1_response,
			r7_data[0], r7_data[1], r7_data[2], r7_data[3]);

	// Check if CMD8 was accepted and voltage is compatible
	bool is_sdc_v2 = (r1_response == 0x01) && (r7_data[2] == 0x01)
			&& (r7_data[3] == 0xAA);
	printf("SDC V2 compatible: %s\n", is_sdc_v2 ? "Yes" : "No");

	// 4. Initialize with ACMD41 (retry with different approaches)
	uint32_t init_timeout = HAL_GetTick();
	uint32_t hcs_bit = is_sdc_v2 ? 0x40000000 : 0x00000000; // Set HCS bit for SDC V2

	printf("Starting ACMD41 initialization (HCS bit: %s)...\n",
			is_sdc_v2 ? "Set" : "Clear");

	do {
		// First send CMD55 (APP_CMD)
		HAL_GPIO_WritePin(GPIOB, GPIO_PIN_6, GPIO_PIN_RESET); // CS LOW
		HAL_Delay(1);

		uint8_t cmd55[] = { 0x77, 0x00, 0x00, 0x00, 0x00, 0x01 };
		HAL_SPI_Transmit(&hspi1, cmd55, 6, 100);

		// Wait for R1 response
		uint8_t cmd55_response = 0xFF;
		for (int i = 0; i < 8; i++) {
			HAL_SPI_TransmitReceive(&hspi1, &dummy, &cmd55_response, 1, 100);
			if (cmd55_response != 0xFF)
				break;
		}

		HAL_GPIO_WritePin(GPIOB, GPIO_PIN_6, GPIO_PIN_SET); // CS HIGH
		HAL_SPI_Transmit(&hspi1, &dummy, 1, 100);
		HAL_Delay(1);

		if (cmd55_response > 0x01) {
			printf("CMD55 failed: 0x%02X\n", cmd55_response);
			return;
		}

		// Then send ACMD41
		HAL_GPIO_WritePin(GPIOB, GPIO_PIN_6, GPIO_PIN_RESET); // CS LOW
		HAL_Delay(1);

		uint8_t acmd41[] = { 0x69, (hcs_bit >> 24) & 0xFF, (hcs_bit >> 16)
				& 0xFF, (hcs_bit >> 8) & 0xFF, hcs_bit & 0xFF, 0x01 };
		HAL_SPI_Transmit(&hspi1, acmd41, 6, 100);

		// Wait for R1 response
		response = 0xFF;
		for (int i = 0; i < 8; i++) {
			HAL_SPI_TransmitReceive(&hspi1, &dummy, &response, 1, 100);
			if (response != 0xFF)
				break;
		}

		HAL_GPIO_WritePin(GPIOB, GPIO_PIN_6, GPIO_PIN_SET); // CS HIGH
		HAL_SPI_Transmit(&hspi1, &dummy, 1, 100);

		if ((HAL_GetTick() - init_timeout) > 2000) {
			printf("ACMD41 timeout! Last response: 0x%02X\n", response);

			// Try alternative initialization for older cards
			printf("Trying alternative initialization for SDC V1/MMC...\n");
			init_timeout = HAL_GetTick();
			hcs_bit = 0x00000000; // Clear HCS bit
			continue;
		}

		if (response == 0x00) {
			printf("ACMD41 successful: 0x%02X\n", response);
			break;
		}

		// Only print every 10th attempt to reduce spam
		static int acmd41_count = 0;
		if (++acmd41_count % 10 == 0) {
			printf("ACMD41 attempt %d: 0x%02X\n", acmd41_count, response);
		}

		HAL_Delay(50); // Longer delay between attempts

	} while (response != 0x00);

	printf("Card initialized successfully!\n");

	// 5. Read OCR to verify card type (optional)
	HAL_GPIO_WritePin(GPIOB, GPIO_PIN_6, GPIO_PIN_RESET); // CS LOW
	HAL_Delay(1);

	uint8_t cmd58[] = { 0x7A, 0x00, 0x00, 0x00, 0x00, 0x01 };
	HAL_SPI_Transmit(&hspi1, cmd58, 6, 100);

	// Read R3 response (5 bytes: R1 + 4 OCR bytes)
	uint8_t ocr_response[5];
	for (int i = 0; i < 5; i++) {
		HAL_SPI_TransmitReceive(&hspi1, &dummy, &ocr_response[i], 1, 100);
	}

	HAL_GPIO_WritePin(GPIOB, GPIO_PIN_6, GPIO_PIN_SET); // CS HIGH
	HAL_SPI_Transmit(&hspi1, &dummy, 1, 100);

	printf("CMD58 Response: 0x%02X, OCR: %02X %02X %02X %02X\n",
			ocr_response[0], ocr_response[1], ocr_response[2], ocr_response[3],
			ocr_response[4]);

	// Check if it's SDHC/SDXC (bit 30 of OCR)
	if (ocr_response[1] & 0x40) {
		printf("Card type: SDHC/SDXC\n");
	} else {
		printf("Card type: SDSC\n");
	}
}

void Test_SD_FileSystem(void) {
	printf("\n=== Testing SD Card File System ===\n");

	FRESULT res;
	UINT bytesWritten, bytesRead;
	char writeBuffer[] =
			"Hello SD Card! This is a test file.\r\nLine 2: Temperature and sensor data can be stored here.\r\n";
	char readBuffer[200];

	// 1. Try to mount the file system
	res = f_mount(&fs, "", 1);  // 1 = mount immediately
	if (res == FR_NO_FILESYSTEM) {
		printf("No file system found. Card may need formatting.\n");
		printf(
				"Since card shows as FAT32 on PC, let's try different mount options...\n");

		// Try mounting without immediate mount
		res = f_mount(&fs, "", 0);
		if (res != FR_OK) {
			printf("Alternative mount also failed: %d\n", res);

			// Only try formatting as last resort
			printf("Attempting to format SD card...\n");
			static BYTE work[4096];
			res = f_mkfs("", FM_FAT32, 0, work, sizeof(work));
			if (res != FR_OK) {
				printf("f_mkfs failed: %d\n", res);
				printf(
						"Error codes: FR_DISK_ERR=1, FR_NOT_READY=3, FR_WRITE_PROTECTED=10, FR_INVALID_PARAMETER=19\n");
				return;
			}
			printf("SD card formatted successfully\n");

			// Try to mount again after formatting
			res = f_mount(&fs, "", 1);
			if (res != FR_OK) {
				printf("f_mount after format failed: %d\n", res);
				return;
			}
		}
		printf("File system mounted successfully\n");
	} else if (res != FR_OK) {
		printf("f_mount failed: %d\n", res);
		printf(
				"Error codes: FR_DISK_ERR=1, FR_NOT_READY=3, FR_NO_FILESYSTEM=13\n");
		return;
	} else {
		printf("File system mounted successfully (existing FAT32)\n");
	}

	// 2. Create/Open a test file for writing
	res = f_open(&fil, "test.txt", FA_CREATE_ALWAYS | FA_WRITE);
	if (res != FR_OK) {
		printf("f_open for write failed: %d\n", res);
		f_mount(NULL, "", 0); // Unmount
		return;
	}
	printf("File opened for writing\n");

	// 3. Write data to file
	res = f_write(&fil, writeBuffer, strlen(writeBuffer), &bytesWritten);
	if (res != FR_OK) {
		printf("f_write failed: %d\n", res);
		f_close(&fil);
		f_mount(NULL, "", 0);
		return;
	}
	printf("Written %u bytes to file\n", bytesWritten);

	// 4. Close the file
	res = f_close(&fil);
	if (res != FR_OK) {
		printf("f_close after write failed: %d\n", res);
		f_mount(NULL, "", 0);
		return;
	}
	printf("File closed after writing\n");

	// 5. Open the file for reading
	res = f_open(&fil, "test.txt", FA_READ);
	if (res != FR_OK) {
		printf("f_open for read failed: %d\n", res);
		f_mount(NULL, "", 0);
		return;
	}
	printf("File opened for reading\n");

	// 6. Read data from file
	memset(readBuffer, 0, sizeof(readBuffer));
	res = f_read(&fil, readBuffer, sizeof(readBuffer) - 1, &bytesRead);
	if (res != FR_OK) {
		printf("f_read failed: %d\n", res);
		f_close(&fil);
		f_mount(NULL, "", 0);
		return;
	}
	printf("Read %u bytes from file\n", bytesRead);

	// 7. Display read data
	printf("File contents:\n%s\n", readBuffer);

	// 8. Verify data integrity
	if (strcmp(writeBuffer, readBuffer) == 0) {
		printf("✓ Data integrity verified - write/read successful!\n");
	} else {
		printf("✗ Data mismatch detected!\n");
		printf("Expected length: %d, Got length: %d\n", strlen(writeBuffer),
				strlen(readBuffer));
	}

	// 9. Test directory listing (optional)
	printf("\nTesting directory listing:\n");
	DIR dir;
	FILINFO fno;

	res = f_opendir(&dir, "/");
	if (res == FR_OK) {
		for (;;) {
			res = f_readdir(&dir, &fno);
			if (res != FR_OK || fno.fname[0] == 0)
				break;
			printf("  %s (%lu bytes)\n", fno.fname, fno.fsize);
		}
		f_closedir(&dir);
	}

	// 10. Close file and unmount
	f_close(&fil);
	f_mount(NULL, "", 0);
	printf("File system unmounted\n");

	printf("=== SD Card File System Test Complete ===\n\n");
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
	MX_USART3_UART_Init();
	MX_SPI1_Init();
	MX_I2C2_Init();
	MX_FATFS_Init();
	/* USER CODE BEGIN 2 */
	printf("Finished setup. Starting kernel and tasks.\n");

	/* USER CODE END 2 */

	/* Init scheduler */
	osKernelInitialize();

	/* USER CODE BEGIN RTOS_MUTEX */
	/* add mutexes, ... */
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
 * @brief SPI1 Initialization Function
 * @param None
 * @retval None
 */
static void MX_SPI1_Init(void) {

	/* USER CODE BEGIN SPI1_Init 0 */

	/* USER CODE END SPI1_Init 0 */

	/* USER CODE BEGIN SPI1_Init 1 */

	/* USER CODE END SPI1_Init 1 */
	/* SPI1 parameter configuration*/
	hspi1.Instance = SPI1;
	hspi1.Init.Mode = SPI_MODE_MASTER;
	hspi1.Init.Direction = SPI_DIRECTION_2LINES;
	hspi1.Init.DataSize = SPI_DATASIZE_8BIT;
	hspi1.Init.CLKPolarity = SPI_POLARITY_LOW;
	hspi1.Init.CLKPhase = SPI_PHASE_1EDGE;
	hspi1.Init.NSS = SPI_NSS_SOFT;
	hspi1.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_128;
	hspi1.Init.FirstBit = SPI_FIRSTBIT_MSB;
	hspi1.Init.TIMode = SPI_TIMODE_DISABLE;
	hspi1.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;
	hspi1.Init.CRCPolynomial = 7;
	hspi1.Init.CRCLength = SPI_CRC_LENGTH_DATASIZE;
	hspi1.Init.NSSPMode = SPI_NSS_PULSE_ENABLE;
	if (HAL_SPI_Init(&hspi1) != HAL_OK) {
		Error_Handler();
	}
	/* USER CODE BEGIN SPI1_Init 2 */

	/* USER CODE END SPI1_Init 2 */

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
	__HAL_RCC_GPIOA_CLK_ENABLE();
	__HAL_RCC_GPIOB_CLK_ENABLE();
	__HAL_RCC_GPIOD_CLK_ENABLE();

	/*Configure GPIO pin Output Level */
	HAL_GPIO_WritePin(MicroSD_CS_GPIO_Port, MicroSD_CS_Pin, GPIO_PIN_SET);

	/*Configure GPIO pin : MicroSD_CS_Pin */
	GPIO_InitStruct.Pin = MicroSD_CS_Pin;
	GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
	GPIO_InitStruct.Pull = GPIO_PULLUP;
	GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
	HAL_GPIO_Init(MicroSD_CS_GPIO_Port, &GPIO_InitStruct);

	/* USER CODE BEGIN MX_GPIO_Init_2 */
	__HAL_RCC_SPI1_CLK_ENABLE();  // Enable SPI1 clock

	// Configure SPI pins (PA5=SCK, PA6=MISO, PA7=MOSI)
	GPIO_InitStruct.Pin = GPIO_PIN_5 | GPIO_PIN_7;
	GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
	GPIO_InitStruct.Pull = GPIO_NOPULL;
	GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
	GPIO_InitStruct.Alternate = GPIO_AF5_SPI1;
	HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

	// MISO needs pull-up
	GPIO_InitStruct.Pin = GPIO_PIN_6;
	GPIO_InitStruct.Pull = GPIO_PULLUP;  // Critical for SD cards
	HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
	/* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

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

	// Run file system test once at startup
	static bool fileSystemTested = false;

	printf("Default task started\n");

	/* Infinite loop */
	for (;;) {
		// Only run the raw SD test if file system hasn't been tested yet
		if (!fileSystemTested) {
			printf("Running initial SD card test...\n");
			test_basic_spi();
			Test_SD_Card();

			// Give some time after initialization
			HAL_Delay(2000);

			// Test file system operations (only once)
			Test_SD_FileSystem();
			fileSystemTested = true;

			printf("SD card testing complete, entering normal operation\n");
		}

		// Normal operation - just delay
		osDelay(10000); // Long delay to avoid interference
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
