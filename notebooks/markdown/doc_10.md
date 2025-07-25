# MICROCONTROLLERS MADE EASY

## 2.5 INTERRUPTS

### Polling
Polling is what you have to do if your microcontroller does not have interrupts or if what you want to do is not time critical. It is a software technique whereby the controller continually asks a peripheral if it needs servicing. The peripheral sets a flag when it has data ready for transferring to the controller, which the controller notices on its next poll. Several peripherals can be polled in succession, with the controller jumping to different software routines, depending on which flags have been set.

!Polling versus Interrupt

| POLLING         | INTERRUPT       |
|------------------|-----------------|
| TASK 1          | TASK 1          |
| POLLING LOOP    |                 |
| TASK 2          | TASK 2          |
| POLLING LOOP    | TASK 3          |
| TASK 3          | Event occurs     |
| POLLING LOOP    | INTERRUPT       |
|                  | Save State      |
|                  | PROCESS EVENT   |
|                  | Restore State   |
| TASK 4          | TASK 4         |

### Interrupts
Rather than have the microcontroller continually polling—asking peripherals (timers, UARTs, A/Ds, external components) whether they have any data available (and finding most of the time they do not), a more efficient method is to have the peripherals tell the controller when they have data ready. The controller can be carrying out its normal function, only responding to peripherals when there is data to respond to.

On receipt of an interrupt, the controller suspends its current operation, identifies the interrupting peripheral, then jumps to the appropriate interrupt service routine. The advantage of interrupts, compared with polling, is the speed of response to external events and reduced software overhead (of continually asking peripherals if they have any data ready).

Most microcontrollers have at least one external interrupt, which can be edge selectable (rising or falling) or level triggered. Both systems have advantages. Edge is not time sensitive, but it is susceptible to glitches. Level must be held high (or low) for a specific duration (which can be a pain but is not susceptible to glitches).