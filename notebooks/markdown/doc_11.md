# MICROCONTROLLERS MADE EASY

## 3 POWER SUPPLY ISSUES IN MCUs

Since automotive applications have been the driving force behind most microcontrollers, and 5 Volts was very easy to do in a car, most microcontrollers only supported 4.5 - 5.5 V operation. In the recent past, as consumer goods were beginning to drive major segments of the microcontroller market, and became portable and lightweight, the requirement for 3 volt (and lower) microcontrollers became urgent. 3 volts means a 2 battery solution, lower voltage, and longer battery life. Most low voltage parts in the market today are simply 5 volt parts that were modified to operate at 3 volts (usually at a performance loss). Some micros being released now are designed from the ground up to operate properly at 3.0 (and lower) voltages, which offer a performance level comparable to 5 volt devices.

But why are voltages going down on ICs? There are a few interesting rules of thumb regarding transistors:

1. The amount of power they dissipate is proportional to their size. If you make a transistor half as big, it dissipates half as much power.
2. Their propagation delay is proportional to their size. If you make a transistor half as big, it’s twice as fast.
3. Their cost is proportional to the square of their size. If you make them half as big, they cost one quarter as much.

!Transistor Parameter Scheme

For years, people have been using 5 Volts to power integrated circuits. Because the transistors were large, there was little danger of damaging the transistor by putting this voltage across it. However, now that the transistors are getting so small, 5 Volts would now destroy them. The only way around this is to start lowering the voltage. This is also why people are now using 3 (actually 3.3) Volt logic, and this will certainly lead to lower voltages in the next few years.