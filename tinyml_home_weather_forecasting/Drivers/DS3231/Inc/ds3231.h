#pragma once
#include <stddef.h>
#include "stm32f7xx_hal.h"

/****************************************************************************************
 * Function:    ds3231_read_time_iso8601_utc_i2c1
 * Purpose:     Read DS3231 via I²C1 and format UTC as "YYYY-MM-DDTHH:MM:SSZ".
 *
 * Parameters:  char *destination_buffer [out]  - Buffer for formatted string (>= 21 bytes).
 *              size_t destination_capacity_bytes [in] - Size of buffer in bytes.
 *
 * Returns:     HAL_StatusTypeDef - HAL_OK on success; HAL error code on failure.
 *
 * Side Effects:
 *                - Performs I²C transactions on I²C1.
 *
 * Preconditions:
 *                - hi2c1 initialized; DS3231 wired to the I²C1 pins and powered at 3.3 V.
 *
 * Postconditions:
 *                - destination_buffer contains a NUL-terminated ISO-8601 timestamp (on success).
 *
 * Concurrency:
 *                - Not reentrant with other I²C1 users; serialize access externally (e.g., mutex).
 *
 * Timing:
 *                - Typical < 1 ms.
 *
 * Errors:
 *                - Returns HAL error codes for NACK/bus faults/timeouts.
 *
 * Notes:
 *                - DS3231 stores BCD; this routine converts to binary and forces 24-hour mode.
 ****************************************************************************************/
HAL_StatusTypeDef ds3231_read_time_iso8601_utc_i2c1(char *destination_buffer,
                                                    size_t destination_capacity_bytes);

/****************************************************************************************
 * Function:    ds3231_set_time_from_components_utc_i2c1
 * Purpose:     Set DS3231 date/time over I²C1 from discrete UTC components.
 *
 * Parameters:  uint16_t year_yyyy   [in]  2000..2099
 *              uint8_t  month_1_12  [in]  1..12
 *              uint8_t  day_1_31    [in]  1..31
 *              uint8_t  hour_0_23   [in]  0..23
 *              uint8_t  minute_0_59 [in]  0..59
 *              uint8_t  second_0_59 [in]  0..59
 *
 * Returns:     HAL_StatusTypeDef - HAL_OK on success.
 *
 * Side Effects:
 *                - Writes DS3231 registers over I²C1.
 *
 * Preconditions:
 *                - I²C1 initialized; device responds at 7-bit address 0x68.
 *
 * Notes:
 *                - Day-of-week is written as 1 (unused).
 ****************************************************************************************/
HAL_StatusTypeDef ds3231_set_time_from_components_utc_i2c1(uint16_t year_yyyy,
                                                           uint8_t month_1_12,
                                                           uint8_t day_1_31,
                                                           uint8_t hour_0_23,
                                                           uint8_t minute_0_59,
                                                           uint8_t second_0_59);

/****************************************************************************************
 * Function:    i2c1_scan_and_log_devices
 * Purpose:     Scan I²C1 and print responding 7-bit addresses (bring-up aid).
 *
 * Parameters:  void
 * Returns:     void
 *
 * Side Effects:
 *                - Produces console prints; performs many tiny I²C ops.
 ****************************************************************************************/
void i2c1_scan_and_log_devices(void);
