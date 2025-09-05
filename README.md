## Description

This project attempts to provide a real-time, short term weather forecast (24 hours) using an embedded system. It can be useful in remote, isolated areas where weather is critical for some process, and Internet bandwidth (and/or electrical power) is limited or unavailable. Due to the low power requirements, it could be run off a battery powered by a solar panel.

Use cases might involve offshore oil/gas rigs, offshore wind farms and remote power stations. Agricultural fields and greenhouses might also find this very useful.

This TinyML home weather forecasting project is my first real embedded project (since my undergrad days). An STM32 F767 board is used as the heart of the system.

I have built a system to collect several bits of weather data, and store it in a csv-based logging system on a MicroSD card using SPI and TinyFS.

A DS3231 module is used to keep accurate UTC time for the log timestamps. 

I have collected temperature, humidity & pressure data using a BME280 sensor, ambient light data using a VEML7700 sensor and rain droplets using an LM393 sensor.

The board talks to most of the sensors using I2C, with the exception of the rain sensor, which uses a single GPIO pin.

The board is powered via USB.

Once I have the basics down, I will implement TinyML to try to predict the weather around my house using machine learning trained on the data I collect.

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