import numpy as np
import Inversion_Library as inversion
import matplotlib.pyplot as plt

# Parameters for the simulation
frequency = 1000  # Hz, frequency of the carrier wave
mod_frequency = 1010
modulation_frequency = 50  # Hz, frequency of the phase inversion
duration = 0.1  # seconds, duration of the signal
sampling_rate = 10000  # Hz, sampling rate for the simulation

# Time array
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Carrier signal (unmodulated high-frequency wave)
carrier_signal = np.sin(2 * np.pi * frequency * t)

# Phase-shift keying (PSK) modulated signal
# Modulation happens at the modulation_frequency

Inversion_Name, modulated_signal = inversion.GradualInversion(t,frequency,modulation_frequency)

# Resulting spTI signal (interference of carrier and modulated signals)
spti_signal = carrier_signal + modulated_signal

# Plotting
plt.figure(figsize=(12, 6))

# Carrier signal plot
plt.subplot(3, 1, 1)
plt.plot(t, carrier_signal)
plt.title('Carrier Signal (Unmodulated)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# PSK modulated signal plot
plt.subplot(3, 1, 2)
plt.plot(t, modulated_signal)
plt.title('PSK Modulated Signal')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# spTI signal plot
plt.subplot(3, 1, 3)
plt.plot(t, spti_signal)
plt.title(f'{Inversion_Name} spTI Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

