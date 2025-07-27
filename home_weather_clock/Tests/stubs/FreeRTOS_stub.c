#include "FreeRTOS_stub.h"

static TickType_t mock_tick_count = 0;

TickType_t xTaskGetTickCount(void) {
    return mock_tick_count;
}

void vTaskDelay(TickType_t ticks) {
    mock_tick_count += ticks;
}

BaseType_t xTaskCreate(void (*pxTaskCode)(void*),
                     const char* pcName,
                     uint16_t usStackDepth,
                     void* pvParameters,
                     UBaseType_t uxPriority,
                     TaskHandle_t* pxCreatedTask) {
    // Explicitly mark all unused parameters
    (void)pxTaskCode;
    (void)pcName;
    (void)usStackDepth;
    (void)pvParameters;
    (void)uxPriority;
    (void)pxCreatedTask;

    return 1; // Return success
}

void stub_set_tick_count(TickType_t new_count) {
    mock_tick_count = new_count;
}

void stub_reset_all(void) {
    mock_tick_count = 0;
}
