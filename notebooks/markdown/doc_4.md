# MICROCONTROLLERS MADE EASY

## 1 TYPICAL MICROCONTROLLER APPLICATIONS
Microcontrollers are frequently found in home appliances (microwave oven, refrigerators, television and VCRs, stereos), computers and computer equipment (laser printers, modems, disk drives), cars (engine control, diagnostics, climate control), environmental control (greenhouse, factory, home), instrumentation, aerospace, and thousands of other uses. In many items, more than one processor can be found.

!Typical MCU Applications

| TV SET          | BODY CONTROLLER      | TELEPHONE SET |
|------------------|---------------------|---------------|
|                  |                     | MONITOR       |
| CAR RADIO        | KEYBOARD            |               |
|                  |                     | DASHBOARD     |
|                  |                     | FRONT PANEL   |
|                  |                     |               |
| BATTERY CHARGER  | DIMMER              | REMOTE        |
| REMOTE CONTROL    |                     | SWITCH        |
|                  |                     | METER         |
|                  |                     | KEYLESS       |

While microprocessors target the maximum processing performance, the purpose of microcontrollers is to implement a set of control functions in the most cost-effective way. Although controlling a microwave oven with a Pentium™ might seem an attractive idea, it can be easily accomplished with an ST6.

In a typical application, the MCU has to manage several tasks according to their priority or to the occurrence of external events (new command sent by the keyboard, external temperature rise, etc.).

!Example of MCU Task Management

```
CENTRAL MCU
FUNCTION
KEYBOARD                                               INFORMATION
SCANNING                                                DISPLAY

MEASURE                    CHANGE
TEMPERATURE              TEMPERATURE
VR02101E
```

The ability to manage control tasks by hardware or by software is the main performance indicator for MCUs.