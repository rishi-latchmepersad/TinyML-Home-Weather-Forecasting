#ifndef FREERTOS_STUB_H
#define FREERTOS_STUB_H

#include <stdint.h>

// Basic type definitions
#define pdMS_TO_TICKS(ms) (ms)
#define portMAX_DELAY 0xFFFFFFFF
#define configTICK_RATE_HZ 1000

typedef void* TaskHandle_t;
typedef uint32_t TickType_t;
typedef int BaseType_t;
typedef unsigned int UBaseType_t;

// Mock functions
TickType_t xTaskGetTickCount(void);
void vTaskDelay(TickType_t ticks);
BaseType_t xTaskCreate(void (*pxTaskCode)(void*),
                     const char* pcName,
                     uint16_t usStackDepth,
                     void* pvParameters,
                     UBaseType_t uxPriority,
                     TaskHandle_t* pxCreatedTask);

// Stub control functions
void stub_set_tick_count(TickType_t new_count);
void stub_reset_all(void);

#endif // FREERTOS_STUB_H
