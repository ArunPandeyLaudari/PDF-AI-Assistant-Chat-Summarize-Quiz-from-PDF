# MICROCONTROLLERS MADE EASY

## 2 ADDITIONAL MICROCONTROLLER FEATURES

### 2.1 TIMERS

**Watchdog Timer**
A watchdog timer provides a means of graceful recovery from a system problem. This could be a program that goes into an endless loop, or a hardware problem that prevents the program from operating correctly. If the program fails to reset the watchdog at some predetermined interval, a hardware reset will be initiated. The bug may still exist, but at least the system has a way to recover. This is especially useful for unattended systems.

**Auto Reload Timer**
Compared to a standard timer, this timer automatically reloads its counting value when the count is over, therefore sparing a waste of CPU resources.

#### Figure 9. Standard Timer and Auto-Reload Timer Principle

| STANDARD TIMER | AUTO RELOAD TIMER |
|----------------|-------------------|
| CLOCK          | CLOCK             |
|                | End of Count      |
| TIMER          | TIMER             |
| Load Register  | Load Register     |
|                | Reload            |

**Pulse Width Modulator**
Often used as a digital-to-analog conversion technique. A pulse train is generated and regulated with a low-pass filter to generate a voltage proportional to the duty cycle.

#### Figure 10. PWM Principle

| CLOCK | V | ANALOG VOLTAGE |
|-------|---|----------------|
| PWM   |   | time           |
|       |   |                |

**Pulse Accumulator**
A pulse accumulator is an event counter. Each pulse increments the pulse accumulator register, recording the number of times this event has occurred.

**Input Capture**
Input Capture can measure external frequencies or time intervals by copying the value from a free running timer into the input capture register when an external event occurs.