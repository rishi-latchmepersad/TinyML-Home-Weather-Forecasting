## Project Background

This project attempted to provide a real-time, short term weather forecast using an embedded system. Such a system can be useful in remote, isolated areas where weather is critical for some process, and Internet bandwidth (and/or electrical power) is limited or unavailable. Due to the low power requirements, it could be run off a battery, powered by a solar panel.

Use cases might involve offshore oil/gas rigs, offshore wind farms and remote power stations. Agricultural fields and greenhouses might also find this very useful.

This TinyML home weather forecasting project was our first real embedded project and our first project using TinyML. An STM32 F767 board was used as the heart of the system.

The following is an image of our system deployed in the field (in a rural home in Trinidad and Tobago):
<img src="./system_in_box_outdoors.jpg" width="400" height="400" />

Of course, the system had to be waterproof, so we used a plastic bag to cover it and put the VEML sensor in a plastic container 🙂

<img src="./full_system_outdoors.jpg" width="400" height="400" />

### Technical Details

We built this system to collect several bits of weather data (to serve as our input features), and store them in a CSV-based logging system on a MicroSD card using SPI and TinyFS.

A DS3231 module was used to keep accurate UTC time for the log timestamps.

We collected temperature, humidity & pressure data using a BME280 sensor, ambient light data using a VEML7700 sensor, and rain droplets using an LM393 sensor.

The board talked to most of the sensors using I2C, with the exception of the rain sensor, which used a single GPIO pin. The enter system was powered via USB from an outlet (we only got the solar panel later on ☹)

After the data collection, we used TinyML to implement a CNN to forecast the temperature in the future. We also implemented a simple RNN, to compare and contrast the results between both models.

We trained the model using a combination of this sensor data (collected in Q4 2025), and API data collected from OpenMeteo. We only had ~2 months of valid sensor data, so we needed to extend the training dataset.

All training was conducted on Google Colab, using Tensorflow and Keras. The model was then compressed using structured pruning and quantization, and deployed on the board using X-CUBE.AI.

Our models were able to forecast the temperature with a MAE of 1.8C for the RNN, and 2.4C for the CNN. We also measured other parameters such as inference time and power consumption. All findings are presented in the paper.

## Research and Writing

We have submitted the paper for this project to the West Indian Journal Of Engineering, and we are currently reviewing the paper. Please check back soon for updates.

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

## MicroSD troubleshooting

- Logged data is stored under `0:/logs/measurements_YYYY-MM-DD.csv`. After the first flush the firmware now performs a one-time sync, unmount/remount, and directory snapshot in the UART log so you can confirm the file exists as it would appear on a PC.
- If you move the card to a PC and do not see the file, check the log for the `DIR 0:/` and `DIR 0:/logs` listings that appear after the first flush. If those are empty, reformat the card as FAT32 and reseat it; the logger will recreate the directory and CSV header on the next boot.