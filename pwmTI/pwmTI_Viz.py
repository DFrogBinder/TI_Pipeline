import numpy as np
import matplotlib.pyplot as plt

# Parameters
fs = 100e3  # Sampling frequency (for high resolution)
t = np.arange(0, 1, 1/fs)  # Time vector (1 second)
f1 = 2000  # Frequency of the first square wave
f2 = 2010  # Frequency of the second square wave
duty_cycle = 0.5  # Duty cycle

# Generate square waves
square1 = signal.square(2 * np.pi * f1 * t, duty=duty_cycle)
square2 = signal.square(2 * np.pi * f2 * t, duty=duty_cycle)

# PWM signal by adding the two square waves
pwm_signal = square1 + square2

# Modulation envelope (absolute value gives the envelope)
envelope = abs(signal.hilbert(pwm_signal))

# Plot
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, pwm_signal, label='PWM Signal')
plt.title('PWM Signal and Modulation Envelope')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, envelope, 'r', label='Modulation Envelope')
plt.xlabel('Time (s)')
plt.ylabel('Envelope Amplitude')
plt.legend()

plt.tight_layout()
plt.show()
