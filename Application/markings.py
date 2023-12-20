import os
import tkinter.messagebox

from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, RectangleSelector
from collections import deque
import numpy as np
import tkinter as tk
from tkinter import filedialog
from scipy.signal import filtfilt, butter
import matplotlib
from pathlib import Path
import tqdm
from csv_reader import csv_reader
from config import window_size, low_cutoff, high_cutoff, nyq, order

'''
Base program for adding markings to data. Automatically splits into text files containing only
ecg and markings for easier dataset creation.
'''

root = tk.Tk()  # Prompts user to select file
currdir = os.getcwd()
par = Path(currdir).parent
signal_dir = str(par) + r"\ECG_Data"
root.filename = filedialog.askopenfilename(initialdir=signal_dir, title="Select file")
matplotlib.use('Qt5Agg')
filepath = root.filename
if len(filepath) == 0:
    exit(0)
file_size = os.stat(filepath).st_size
root.destroy()

filename = os.path.split(filepath)[1]
filename = filename[:filename.index('.')]
ecg_filepath = os.path.join('..', 'Training', f'{filename}_ecg')
markings_filepath = os.path.join('..', 'Training', f'{filename}_markings')
markings_exist = os.path.exists(markings_filepath)
load_markings = False

if markings_exist:
    load_markings = tkinter.messagebox.askyesno(title='', message='Previous markings found, do you want to load them?')

file = open(filepath, 'r+')  # Gets an average line size for the progress bar
fig, axs = plt.subplots()
file_loc = file.tell()
temp_line = file.readline()
file.seek(file_loc)
line_size = len(temp_line.encode('utf-8'))

ecg = []  # Raw ECG signal
signal = []  # Raw signal (0 for non-peak and 1 for peak)
signals = []  # Indices of peaks in signals


class Events:
    """Class for an interactive pyplot to handle click, scroll, etc. events."""
    def __init__(self):
        self.default_length = 15 * window_size
        self.x_right = 15 * window_size  # Right and left bounds of window
        self.x_left = 0
        self.adding = False  # Following are toggling which of the modes are on
        self.removing = False
        self.clean = False

    def delete_onclick(self, event):
        """Delete a mark (actually deletes an area around the click since marks can be multiple values)."""
        if self.removing:
            try:
                ind = int(event.xdata)
            except TypeError:
                return
            if ind != 0:
                rad = 8
                remove(ind, radius=rad)
                line.set_ydata(signal)
                plt.draw()

    def scroll(self, event):
        """Just allows for scrolling through the plot."""
        dist = -event.step * (self.x_right - self.x_left)
        self.x_left += dist / 6
        self.x_right += dist / 6
        axs.axis([self.x_left, self.x_right, -0.5, 1])
        plt.draw()
        position_slider.valinit = (self.x_right + self.x_left) / 2
        position_slider.reset()

    def add_onclick(self, event):
        """Adds a signal on click (also has some radius)."""
        if self.adding:
            try:
                ind = int(event.xdata)
            except TypeError:
                return
            if ind != 0:
                add(ind)
                line.set_ydata(signal)
                plt.draw()

    def clean_region(self, eclick, erelease):
        """Deletes any signals in the selected region."""
        if self.clean:
            try:
                x1 = int(eclick.xdata)
                x2 = int(erelease.xdata)
            except TypeError:
                return
            if x1 > 0 and x2 > 0:
                clean(min(x1, x2), max(x1, x2))
                line.set_ydata(signal)
                plt.draw()

    def set_width(self, val):
        """Display width. How many datapoints are on screen at a time."""
        mid = (self.x_left + self.x_right) / 2
        self.x_left = mid - val
        self.x_right = mid + val
        axs.axis([self.x_left, self.x_right, -0.5, 1])
        plt.draw()

    def change_mode(self, label):
        """Handles changing modes."""
        if label == 'Browse':
            self.adding = False
            self.removing = False
            self.clean = False
        elif label == 'Add':
            self.adding = True
            self.removing = False
            self.clean = False
        elif label == 'Delete':
            self.removing = True
            self.adding = False
            self.clean = False
        elif label == 'Clean Region':
            self.removing = False
            self.adding = False
            self.clean = True

    def set_pos(self, val):
        """Slider which you can use to traverse through the plot."""
        width = self.x_right - self.x_left
        self.x_left = val - width / 2
        self.x_right = val + width / 2
        axs.axis([self.x_left, self.x_right, -0.5, 1])
        plt.draw()


def add(ind):
    """Adds a signal to the plot at some index."""
    for i in range(-2, 2):
        signal[ind + i] = 1


def remove(ind, radius=2):
    """Removes signals from the plot at some index with some 'radius' around it."""
    for i in range(-radius, radius):
        signal[ind + i] = 0


def clean(x1, x2):
    """Cleans a region on the plot, from x1 to x2."""
    for i in range(x1, x2 + 1):
        signal[i] = 0


def toggle_clean_selector(event):
    """Handles the selector for region selection for cleaning."""
    if events.clean:
        if toggle_clean_selector.RS.active:
            toggle_clean_selector.RS.set_active(False)
        if not toggle_clean_selector.RS.active:
            toggle_clean_selector.RS.set_active(True)
    else:
        toggle_clean_selector.RS.set_active(False)


events = Events()
last_few = deque(maxlen=8)
max_lines = 1000000

reader = csv_reader(file)
every_i = 10000

if load_markings:
    markings_file = open(markings_filepath, 'r+')
with tqdm.tqdm(total=min(file_size // line_size, max_lines)) as pbar:  # Progress bar
    pbar.set_description('Lines Read ')
    for i, value in enumerate(reader):
        if i % every_i == 0:
            pbar.update(every_i)
        ecg.append(value)
        if load_markings:
            try:
                signal.append(int(markings_file.readline()))
            except ValueError:
                signal.append(0)
        else:
            signal.append(0)
        if i > max_lines:
            break

if load_markings:
    markings_file.close()

axs.plot(range(len(ecg)), ecg, zorder=101)
line, = axs.plot(range(len(signal)), signal)

axs.legend(['ECG', 'Signal'], loc='upper left')
axs.axis([0, 6000, -0.5, 1])

rad = plt.axes([0.4, 0.01, 0.1, 0.075])  # Other interactive stuff, (buttons, sliders, event handlers, etc.)
width_slider_pos = plt.axes([0.2, 0.9, 0.65, 0.03])
position_slider_pos = plt.axes([0.2, 0.95, 0.65, 0.03])
stat = RadioButtons(rad, ('Browse', 'Add', 'Delete', 'Clean Region'))
toggle_clean_selector.RS = RectangleSelector(axs, events.clean_region, drawtype='box', button=1,
                                             rectprops=dict(facecolor='gray'))
stat.on_clicked(events.change_mode)
width_slider = Slider(width_slider_pos, 'Width', 200, 10000, valinit=3000)
width_slider.on_changed(events.set_width)
position_slider = Slider(position_slider_pos, 'Position', 0, len(signal), valinit=0)
position_slider.on_changed(events.set_pos)

scroll = fig.canvas.mpl_connect('scroll_event', events.scroll)
add_click = fig.canvas.mpl_connect('button_press_event', events.add_onclick)
delete_click = fig.canvas.mpl_connect('button_press_event', events.delete_onclick)
clean_region = fig.canvas.mpl_connect('button_press_event', toggle_clean_selector)

plt.show()

file.close()

if 1 in signal:
    ecg_file = open(ecg_filepath, 'w+')
    markings_file = open(markings_filepath, 'w+')

    with tqdm.tqdm(total=len(ecg)) as pbar:
        pbar.set_description('Lines written ')
        i = 0
        for ecg_point, marking in zip(ecg, signal):
            i += 1
            if i % every_i == 0:
                pbar.update(every_i)
            ecg_file.write(str(ecg_point) + '\n')
            markings_file.write(str(marking) + '\n')

    ecg_file.close()
    markings_file.close()
