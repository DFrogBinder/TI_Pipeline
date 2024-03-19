import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider, QLabel, QGridLayout
from PyQt5.QtCore import Qt
from matplotlib.animation import FuncAnimation

class SignalVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()

        # GUI Configuration
        self.setWindowTitle('Real-time Signal Visualizer')
        self.setGeometry(100, 100, 800, 600)

        # Initialize plot
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvas(self.figure)

        # Set up the main layout
        main_layout = QVBoxLayout()
        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

        # Add canvas (plot) to the main layout
        main_layout.addWidget(self.canvas)

        # Grid layout for sliders and labels
        grid_layout = QGridLayout()
        main_layout.addLayout(grid_layout)

        # Signal Parameters with default values
        self.frequency = 1000
        self.mod_frequency = 1010
        self.modulation_frequency = 50
        self.duration = 0.1
        self.sampling_rate = 10000

        # Adding sliders for signal parameters
        self.create_slider(grid_layout, 0, 'Frequency', self.frequency, 500, 2000, 100, self.update_frequency)
        self.create_slider(grid_layout, 1, 'Modulation Frequency', self.mod_frequency, 500, 2000, 100, self.update_mod_frequency)
        self.create_slider(grid_layout, 2, 'Phase Inversion Frequency', self.modulation_frequency, 10, 100, 5, self.update_modulation_frequency)
        self.create_slider(grid_layout, 3, 'Duration', int(self.duration * 1000), 100, 1000, 100, self.update_duration)
        self.create_slider(grid_layout, 4, 'Sampling Rate', self.sampling_rate, 5000, 20000, 1000, self.update_sampling_rate)

        # Initialize plot
        self.plot()

    def create_slider(self, layout, row, label_text, value, min_value, max_value, step, callback):
        slider_label = QLabel(f'{label_text}: {value}')
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_value)
        slider.setMaximum(max_value)
        slider.setValue(value)
        slider.setTickInterval(step)
        slider.setTickPosition(QSlider.TicksBelow)
        slider.valueChanged.connect(lambda: self.slider_changed(slider, slider_label, label_text, callback))
        layout.addWidget(slider_label, row, 0)
        layout.addWidget(slider, row, 1)

    def slider_changed(self, slider, label, name, callback):
        value = slider.value()
        label.setText(f'{name}: {value}')
        callback(value)

    def update_frequency(self, value):
        self.frequency = value
        self.plot()

    def update_mod_frequency(self, value):
        self.mod_frequency = value
        self.plot()

    def update_modulation_frequency(self, value):
        self.modulation_frequency = value
        self.plot()

    def update_duration(self, value):
        self.duration = value / 1000.0  # Convert to seconds
        self.plot()

    def update_sampling_rate(self, value):
        self.sampling_rate = value
        self.plot()

    def plot(self):
        # Update the signal based on current parameters
        t = np.linspace(0, self.duration, int(self.sampling_rate * self.duration), endpoint=False)
        carrier_signal = np.sin(2 * np.pi * self.frequency * t)
        phi = np.pi * np.sin(2 * np.pi * self.modulation_frequency * t) + np.pi/2 * np.sin(4 * np.pi * self.modulation_frequency * t)
        modulated_signal = np.sin(2 * np.pi * self.mod_frequency * t + phi)
        spti_signal = carrier_signal + modulated_signal

        # Clear and update the plot
        self.ax.clear()
        self.ax.plot(t, spti_signal)
        self.ax.set_title('Real-time Signal Visualization')
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Amplitude')
        self.canvas.draw()

# Main function to run the application
def main():
    app = QApplication(sys.argv)
    main_window = SignalVisualizer()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
