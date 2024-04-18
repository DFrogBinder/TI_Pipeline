# Re-importing necessary libraries and re-defining variables after code execution state reset
import numpy as np
import matplotlib.pyplot as plt

# Constants
fs = 44100  # Sampling frequency in Hz
duration = 0.3  # Duration in seconds for extended visualization
f1 = 1000  # Frequency of the first signal in Hz
f2 = 1010  # Frequency of the second signal in Hz
f3 = 1000  # Frequency of the third signal in Hz

# Time array for the extended duration
t_long = np.arange(0, duration, 1/fs)

# Generate the two original signals for the extended duration
signal1_long = np.sin(2 * np.pi * f1 * t_long)
signal2_long = np.sin(2 * np.pi * f2 * t_long)
signal3_long = np.sin(2 * np.pi * f3 * t_long)

# Mix the signals
mixed_signal_long = signal1_long + signal2_long + signal3_long

# Plot the two original signals along with the mixed signal for comparison
plt.figure(figsize=(15, 6))

# Plot the first signal
plt.subplot(4, 1, 1)  # 3 rows, 1 column, 1st subplot
plt.plot(t_long, signal1_long, label=f'{f1/1000} kHz Signal')
plt.title(f'{f1/1000} kHz Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

# Plot the second signal
plt.subplot(4, 1, 2)  # 3 rows, 1 column, 2nd subplot
plt.plot(t_long, signal2_long, label=f'{f2/1000} kHz Signal')
plt.title(f'{f2/1000} kHz Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

# Plot the third signal
plt.subplot(4, 1, 3)  # 3 rows, 1 column, 3nd subplot
plt.plot(t_long, signal3_long, label=f'{f3/1000} kHz Signal')
plt.title(f'{f3/1000} kHz Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

# Plot the mixed signal
plt.subplot(4, 1, 4)  # 3 rows, 1 column, 4rth subplot
plt.plot(t_long, mixed_signal_long, label='Mixed Signal')
plt.title('Mixed Signal (1 kHz and 1.01 kHz)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

plt.show()
