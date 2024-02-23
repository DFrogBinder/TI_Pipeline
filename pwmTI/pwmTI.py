import numpy as np
import matplotlib.pyplot as plt
from math import trunc

# Corrected PWM generation function
def Generate_PWM(t, frequency, duty_cycle):
    pt = t * frequency
    tc = pt - trunc(pt)
    return 1 if tc < duty_cycle else 0

# Parameters for the PWM signal
frequency = [2,4]  # Frequency in Hz
duty_cycle = [0.5,1.5]  # Duty cycle as a fraction
time_end = [2]  # Time period to generate the signal over, in seconds
sampling_rate = 1000  # Sampling rate in samples per second

# Generating the time array
t = np.linspace(0, time_end[0], int(time_end[0] * sampling_rate))

# Generating the PWM signal
signal_1 = np.array([Generate_PWM(ti, frequency[0], duty_cycle[0]) for ti in t])
signal_2 = np.array([Generate_PWM(ti, frequency[1], duty_cycle[1]) for ti in t])


# Plotting the signal
plt.figure(figsize=(10, 3))
plt.plot(t, signal_1+signal_2, label=f'PWM Signal - Frequency: {frequency}Hz, Duty Cycle: {duty_cycle*100}%')
plt.title('PWM Signal')
plt.legend()
plt.grid(True)
plt.show()


