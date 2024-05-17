import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# Sample rate
fs = 44100  # CD quality audio sample rate

# Duration of the signal in seconds
duration = 5  # 5 seconds of white noise

# Generate white noise
white_noise = np.random.normal(0, 1, int(fs * duration))

# Bandpass filter parameters
lowcut = 101  # Low frequency cutoff
highcut = 640  # High frequency cutoff
order = 6  # Order of the filter

# Create a bandpass Butterworth filter
b, a = signal.butter(order, [lowcut, highcut], btype='band', fs=fs)

# Apply the filter to the white noise
filtered_noise = signal.filtfilt(b, a, white_noise)

# Plot the time domain signal
plt.figure(figsize=(12, 6))
plt.plot(np.linspace(0, duration, len(filtered_noise)), filtered_noise, label='Filtered White Noise')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('White Noise Signal with Bandpass Filter (101-640 Hz)')
plt.grid(True)
plt.show()

# Plot the frequency spectrum
plt.figure(figsize=(12, 6))
frequencies, power_spectrum = signal.welch(filtered_noise, fs, nperseg=1024)
plt.semilogy(frequencies, power_spectrum, label='Filtered White Noise Spectrum')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power/Frequency [dB/Hz]')
plt.title('Frequency Spectrum of White Noise Signal (101-640 Hz)')
plt.grid(True)
plt.show()

