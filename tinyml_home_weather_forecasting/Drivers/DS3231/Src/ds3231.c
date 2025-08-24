#include "ds3231.h"
#include <stdio.h>
#include <string.h>

#ifndef DS3231_ADDR_7BIT
#define DS3231_ADDR_7BIT  (0x68u)
#endif
#define DS3231_ADDR_8BIT  (DS3231_ADDR_7BIT << 1)

extern I2C_HandleTypeDef hi2c1;

static inline uint8_t bcd_to_bin(uint8_t b) {
	return (uint8_t) (((b >> 4) * 10u) + (b & 0x0Fu));
}
static inline uint8_t bin_to_bcd(uint8_t d) {
	return (uint8_t) (((d / 10u) << 4) | (d % 10u));
}

/****************************************************************************************
 * Function:    ds3231_read_time_iso8601_utc_i2c1
 * Purpose:     Read DS3231 via I²C1 and format UTC as "YYYY-MM-DDTHH:MM:SSZ".
 ****************************************************************************************/
HAL_StatusTypeDef ds3231_read_time_iso8601_utc_i2c1(char *dst, size_t cap) {
	uint8_t reg = 0x00u; /* seconds */
	uint8_t raw[7];

	HAL_StatusTypeDef st = HAL_I2C_Master_Transmit(&hi2c1, DS3231_ADDR_8BIT,
			&reg, 1u, 50u);
	if (st != HAL_OK)
		return st;

	st = HAL_I2C_Master_Receive(&hi2c1, DS3231_ADDR_8BIT, raw, sizeof raw, 50u);
	if (st != HAL_OK)
		return st;

	uint8_t sec_bcd = raw[0] & 0x7Fu;
	uint8_t min_bcd = raw[1] & 0x7Fu;
	uint8_t hr_raw = raw[2];
	uint8_t date_bcd = raw[4] & 0x3Fu;
	uint8_t mon_raw = raw[5];
	uint8_t yr_bcd = raw[6];

	uint8_t hours24;
	if (hr_raw & 0x40u) { /* 12h mode */
		uint8_t hr12 = bcd_to_bin((uint8_t) (hr_raw & 0x1Fu));
		uint8_t is_pm = (hr_raw & 0x20u) ? 1u : 0u;
		if (hr12 == 12u)
			hr12 = 0u;
		hours24 = (uint8_t) (hr12 + (is_pm ? 12u : 0u));
	} else {
		hours24 = bcd_to_bin((uint8_t) (hr_raw & 0x3Fu));
	}

	uint16_t year = (uint16_t) (2000u + bcd_to_bin(yr_bcd));
	uint8_t month = bcd_to_bin((uint8_t) (mon_raw & 0x1Fu));
	uint8_t day = bcd_to_bin(date_bcd);
	uint8_t mins = bcd_to_bin(min_bcd);
	uint8_t secs = bcd_to_bin(sec_bcd);

	(void) snprintf(dst, cap, "%04u-%02u-%02uT%02u:%02u:%02uZ", year, month,
			day, hours24, mins, secs);
	return HAL_OK;
}

/****************************************************************************************
 * Function:    ds3231_set_time_from_components_utc_i2c1
 * Purpose:     Set DS3231 date/time over I²C1 from discrete UTC components.
 ****************************************************************************************/
HAL_StatusTypeDef ds3231_set_time_from_components_utc_i2c1(uint16_t year_yyyy,
		uint8_t month_1_12, uint8_t day_1_31, uint8_t hour_0_23,
		uint8_t minute_0_59, uint8_t second_0_59) {
	uint8_t frame[8];
	frame[0] = 0x00u; /* start at seconds */
	frame[1] = bin_to_bcd((uint8_t) (second_0_59 & 0x7Fu));
	frame[2] = bin_to_bcd((uint8_t) (minute_0_59 & 0x7Fu));
	frame[3] = bin_to_bcd((uint8_t) (hour_0_23 & 0x3Fu)); /* force 24h */
	frame[4] = 0x01u; /* day-of-week (unused) */
	frame[5] = bin_to_bcd((uint8_t) (day_1_31 & 0x3Fu));
	frame[6] = bin_to_bcd((uint8_t) (month_1_12 & 0x1Fu));
	frame[7] = bin_to_bcd(
			(uint8_t) ((year_yyyy >= 2000u) ? (year_yyyy - 2000u) : 0u));

	return HAL_I2C_Master_Transmit(&hi2c1, DS3231_ADDR_8BIT, frame,
			sizeof frame, 100u);
}

/****************************************************************************************
 * Function:    i2c1_scan_and_log_devices
 * Purpose:     Scan I²C1 and print responding 7-bit addresses.
 ****************************************************************************************/
void i2c1_scan_and_log_devices(void) {
	printf("I2C1 scan:\r\n");
	for (uint8_t addr7 = 0x08u; addr7 < 0x78u; ++addr7) {
		uint8_t dummy = 0;
		if (HAL_I2C_Master_Transmit(&hi2c1, (uint16_t) (addr7 << 1), &dummy, 0u,
				5u) == HAL_OK) {
			printf("  - Found device at 0x%02X\r\n", addr7);
		}
	}
}
