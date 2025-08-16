## Description

This TinyML home weather forecasting project is my first real STM32 project. An STM32 F767 board is used as the heart of the system.

I am building a system to collect temperature, humidity & pressure data using a Bosch BME280 sensor, and store it in a csv-based logging system on a MicroSD card.

I will later add additional sensors to store even more weather data. I will also connect a RTC sensor to store timestamps for the weather data.

Once I have the basics down, I will implement TinyML to try to predict the weather around my house using machine learning trained on the data I collect.

## Diagrams

### Block Diagram

<img alt="Block Diagram" width="600px" src="./tinyml_home_weather_forecasting-Block Diagram.jpg" />
