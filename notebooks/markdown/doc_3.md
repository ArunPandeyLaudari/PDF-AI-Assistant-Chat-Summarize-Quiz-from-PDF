# MICROCONTROLLERS MADE EASY

## SERIAL INTERFACE
Serial interfaces are used to exchange data with the external world. Many microcontrollers have both asynchronous and synchronous communications peripherals built in. Usually, an asynchronous interface is called a serial communication interface (SCI or UART) while the synchronous interface is called a serial peripheral interface (SPI). A typical SCI application is to connect a PC for debugging purposes while a typical SPI application is to connect an external EEPROM.

A synchronous bus includes a separate line for the clock signal which simplifies the transmitter and receiver but is more susceptible to noise when used over long distances. With an asynchronous bus, the transmitter and receiver clocks are independent, and a resynchronization is performed for each byte at the start bit.

### Figure 3. Synchronous and Asynchronous Communication Principles
| Synchronous | Asynchronous |
|-------------|--------------|
| CLOCK       | CLOCK        |
| 0 b b b b b 1 + | Start               |
| ~~b~~ ~~b~~ ~~b~~ ~~b~~ ~~b~~ ~~b~~ ~~b~~ | DATA                |
| DATA        | Stop                |

## A/D CONVERTER
The A/D converter converts an external analog signal (typically relative to voltage) into a digital representation. Microcontrollers that have this feature can be used for instrumentation, environmental data logging, or any application that lives in an analog world.

### Figure 4. A/D Converter Principle
| Voltage | ANALOG | A / D | DIGITAL |
|---------|--------|-------|---------|
| 5       |        |       |         |
| 4       |        |       |         |
| 3       |        |       |         |
| 2       |        |       |         |
| 1       |        |       |         |
| Time    |        |       |         |

!A/D Converter Principle