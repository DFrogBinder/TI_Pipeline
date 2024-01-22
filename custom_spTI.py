import numpy as np
import matplotlib.pyplot as plt

# Time array
t = np.linspace(0, 1, 1000)  # 1 second, 1000 points
frequency = 5  # Hz for the carrier wave
modulation_frequency = 1  # Hz for phase inversion

# Simple Phase Inversion
phi = np.pi * (t * modulation_frequency % 1 < 0.5)
signal = np.sin(2 * np.pi * frequency * t + phi)

# Gradual Phase Shift
phi = np.pi * (t * modulation_frequency % 1)
signal = np.sin(2 * np.pi * frequency * t + phi)

# Sawtooth Phase Shift
phi = 2 * np.pi * (t * modulation_frequency % 1)
signal = np.sin(2 * np.pi * frequency * t + phi)

# Customized Phase Shift
phi = np.pi * np.sin(2 * np.pi * modulation_frequency * t) + np.pi/2 * np.sin(4 * np.pi * modulation_frequency * t)
signal = np.sin(2 * np.pi * frequency * t + phi)

# Plot
plt.plot(t, signal)
plt.title('Simple Phase Inversion')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()
