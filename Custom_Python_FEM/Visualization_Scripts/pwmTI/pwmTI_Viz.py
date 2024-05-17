import numpy as np
import matplotlib.pyplot as plt

def generate_two_pwm_waveforms(f1, f2, duty_cycle, duration, sampling_rate):
    # Generate two PWM signals with slightly different frequencies
    t = np.arange(0, duration, 1/sampling_rate)
    period1 = 1 / f1
    period2 = 1 / f2
    pwm1 = (t % period1 < period1 * duty_cycle) * 2 - 1  # Convert to bipolar (-1 to 1)
    pwm2 = (t % period2 < period2 * duty_cycle) * 2 - 1  # Convert to bipolar (-1 to 1)
    
    # Sum the two PWM signals
    pwm_sum = pwm1 + pwm2
    
    return t, pwm1, pwm2, pwm_sum

# Parameters for the two PWM signals
f1 = 2000  # Frequency of first signal in Hz
f2 = 2010  # Frequency of second signal in Hz
duty_cycle = 0.5  # 50% duty cycle
duration = 0.02  # 10 ms duration for visualization
sampling_rate = 250000  # 250 kHz

# Generate the two PWM waveforms and their sum
t, pwm1, pwm2, pwm_sum = generate_two_pwm_waveforms(f1, f2, duty_cycle, duration, sampling_rate)

plt.rcParams.update({'font.size': 10})  # Sets the default font size to 10


# Plotting the PWM signals and their sum
plt.figure(figsize=(12, 5))
plt.subplot(3, 1, 1)
plt.plot(t, pwm1)
plt.title('PWM 2kHz')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(t, pwm2)
plt.title('PWM 2.01kHz')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, pwm_sum)
plt.title('Summed PWM Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()
