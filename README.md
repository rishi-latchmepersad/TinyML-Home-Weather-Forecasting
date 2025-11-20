## Description

This project attempts to provide a real-time, short term weather forecast using an embedded system. It can be useful in remote, isolated areas where weather is critical for some process, and Internet bandwidth (and/or electrical power) is limited or unavailable. Due to the low power requirements, it could be run off a battery powered by a solar panel.

Use cases might involve offshore oil/gas rigs, offshore wind farms and remote power stations. Agricultural fields and greenhouses might also find this very useful.

This TinyML home weather forecasting project is our first real embedded project. An STM32 F767 board is used as the heart of the system.

We built this system to collect several bits of weather data, and store it in a csv-based logging system on a MicroSD card using SPI and TinyFS.

A DS3231 module is used to keep accurate UTC time for the log timestamps. 

We have collected temperature, humidity & pressure data using a BME280 sensor, ambient light data using a VEML7700 sensor and rain droplets using an LM393 sensor.

The board talks to most of the sensors using I2C, with the exception of the rain sensor, which uses a single GPIO pin.

The board is powered via USB.

After the data collection, we used TinyML to implement a CNN to forecast the temperature in the future.

## MicroSD troubleshooting

- Logged data is stored under `0:/logs/measurements_YYYY-MM-DD.csv`. After the first flush the firmware now performs a one-time sync, unmount/remount, and directory snapshot in the UART log so you can confirm the file exists as it would appear on a PC.
- If you move the card to a PC and do not see the file, check the log for the `DIR 0:/` and `DIR 0:/logs` listings that appear after the first flush. If those are empty, reformat the card as FAT32 and reseat it; the logger will recreate the directory and CSV header on the next boot.

## Diagrams

### Block Diagram

<img style="border-radius:5px;border:2px solid #ccc" alt="Block Diagram" width="600px" src="./tinyml_home_weather_forecasting-Block Diagram.png" />

### MicroSD Card State Machine Diagram

<img style="border-radius:5px;border:2px solid #ccc" alt="MicroSD Card State Machine Diagram" width="600px" src="./tinyml_home_weather_forecasting-MicroSD Card State Machine Diagram.png" />

### BME280 State Machine Diagram

<img style="border-radius:5px;border:2px solid #ccc" alt="BME280 State Machine Diagram" width="600px" src="./tinyml_home_weather_forecasting-BME280 State Machine Diagram.png" />

### VEML7700 State Machine Diagram

<img style="border-radius:5px;border:2px solid #ccc" alt="VEML7700 State Machine Diagram" width="600px" src="./tinyml_home_weather_forecasting-VEML7700 State Machine Diagram.png" />

### LM393 State Machine Diagram

<img style="border-radius:5px;border:2px solid #ccc" alt="LM393 State Machine Diagram" width="600px" src="./tinyml_home_weather_forecasting-LM393 State Machine Diagram.png" />
