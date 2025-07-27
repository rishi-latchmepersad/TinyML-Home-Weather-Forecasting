#include "unity.h"

// Declare test functions
void test_bme280_init_state(void);
void test_bme280_should_initialize(void);
void test_bme280_should_return_temperature(void);

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_bme280_init_state);
    RUN_TEST(test_bme280_should_initialize);
    RUN_TEST(test_bme280_should_return_temperature);
    return UNITY_END();
}
