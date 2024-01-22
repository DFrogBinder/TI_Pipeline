import numpy as np
import matplotlib.pyplot as plt

# Parameters for the simulation
frequency = 1000  # Hz, frequency of the carrier wave
modulation_frequency = 50  # Hz, frequency of the phase inversion
duration = 0.1  # seconds, duration of the signal
sampling_rate = 10000  # Hz, sampling rate for the simulation

# Time array
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Carrier signal (unmodulated high-frequency wave)
carrier_signal = np.sin(2 * np.pi * frequency * t)

# Phase-shift keying (PSK) modulated signal
# Modulation happens at the modulation_frequency
# Sawtooth Phase Shift
# Gradual Phase Shift
phi = np.pi * (t * modulation_frequency % 1)
modulated_signal = np.sin(2 * np.pi * frequency * t + phi)
# modulated_signal = np.sin(2 * np.pi * frequency * t + np.pi * (t * modulation_frequency % 1 < 0.5))

# Resulting spTI signal (interference of carrier and modulated signals)
spti_signal = carrier_signal + modulated_signal

# Plotting
plt.figure(figsize=(12, 6))

# Carrier signal plot
plt.subplot(3, 1, 1)
plt.plot(t, carrier_signal)
plt.title('Carrier Signal (Unmodulated)')
plt.ylabel('Amplitude')
plt.grid(True)

# PSK modulated signal plot
plt.subplot(3, 1, 2)
plt.plot(t, modulated_signal)
plt.title('PSK Modulated Signal')
plt.ylabel('Amplitude')
plt.grid(True)

# spTI signal plot
plt.subplot(3, 1, 3)
plt.plot(t, spti_signal)
plt.title('Resulting spTI Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()

