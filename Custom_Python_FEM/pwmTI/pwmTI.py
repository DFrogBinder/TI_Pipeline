import numpy as np
import matplotlib.pyplot as plt
from math import trunc

# Corrected PWM generation function
def Generate_PWM(t, frequency, duty_cycle):
    pt = t * frequency
    tc = pt - trunc(pt)
    return 1 if tc < duty_cycle else 0

# Parameters for the two PWM signals
frequency1 = 2  # Frequency of first signal in Hz
frequency2 = 5  # Frequency of second signal in Hz
duty_cycle = 0.5  # Duty cycle as a fraction for both signals
time_end = 2  # Time period to generate the signal over, in seconds
sampling_rate = 1000  # Sampling rate in samples per second

# Generating the time array
t = np.linspace(0, time_end, int(time_end * sampling_rate))

# Generating the first PWM signal
signal1 = np.array([Generate_PWM(ti, frequency1, duty_cycle) for ti in t])

# Generating the second PWM signal
signal2 = np.array([Generate_PWM(ti, frequency2, duty_cycle) for ti in t])

# Summing the two signals
sum_signal = signal1 + signal2

# Plotting the signals in subplots
plt.figure(figsize=(10, 9))

# First PWM signal
plt.subplot(3, 1, 1)  # 3 rows, 1 column, 1st subplot
plt.plot(t, signal1, label=f'PWM Signal 1 - Frequency: {frequency1}Hz, Duty Cycle: {duty_cycle*100}%')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('PWM Signal 1')
plt.legend()
plt.grid(True)

# Second PWM signal
plt.subplot(3, 1, 2)  # 3 rows, 1 column, 2nd subplot
plt.plot(t, signal2, label=f'PWM Signal 2 - Frequency: {frequency2}Hz, Duty Cycle: {duty_cycle*100}%')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('PWM Signal 2')
plt.legend()
plt.grid(True)

# Sum of the two signals
plt.subplot(3, 1, 3)  # 3 rows, 1 column, 3rd subplot
plt.plot(t, sum_signal, label='Sum of PWM Signal 1 and 2')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Sum of PWM Signals')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


