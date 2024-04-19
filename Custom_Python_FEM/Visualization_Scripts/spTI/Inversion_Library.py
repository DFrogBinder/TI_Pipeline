import numpy as np

'''Simple Phase Inversion'''
def SimpleInversion(t, frequency, modulation_frequency):
    phi = np.pi * (t * modulation_frequency % 1 < 0.5)
    signal = np.sin(2 * np.pi * frequency * t + phi)
    return 'Simple Inversion',signal


'''SawtoothInversion Phase Shift'''
def SawtoothInversion(t, frequency, modulation_frequency):
    phi = np.pi * (t * modulation_frequency % 1)
    signal = np.sin(2 * np.pi * frequency * t + phi)
    return 'SawtoothInversion Inversion', signal


'''Gradual Phase Shift'''
def GradualInversion(t, frequency, modulation_frequency):
    phi = 2 * np.pi * (t * modulation_frequency % 1)
    signal = np.sin(2 * np.pi * frequency * t + phi)
    return 'Gradual Inversion',signal

'''Customized Phase Shift'''
def CustomInversion(t, frequency, modulation_frequency):
    phi = np.pi * np.sin(2 * np.pi * modulation_frequency * t) + np.pi/2 * np.sin(4 * np.pi * modulation_frequency * t)
    signal = np.sin(2 * np.pi * frequency * t + phi)
    return 'Custom Inversion',signal

