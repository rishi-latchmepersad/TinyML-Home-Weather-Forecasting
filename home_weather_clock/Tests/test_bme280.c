#include "unity.h"
#include "../Tasks/Sensors/bme280_task.h"

// Mock hardware-dependent functions
int8_t mock_bme280_setup() {
    return BME280_OK;
}

void setUp(void) {
    // Reset state before each test
}

void tearDown(void) {
    // Cleanup
}

void test_bme280_init_state(void) {
    bme280_task_data_t task_data = {0};
    bme280_task_init(&task_data);
    TEST_ASSERT_EQUAL(BME280_STATE_INIT, task_data.state);
}
