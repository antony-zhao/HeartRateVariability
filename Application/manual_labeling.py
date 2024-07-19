import os
from matplotlib import pyplot as plt
import mmap
import re
from matplotlib.widgets import Button, Slider, RadioButtons, RectangleSelector
from collections import deque
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib
from pathlib import Path
import tqdm
from config import interval_length, low_cutoff, high_cutoff, nyq, order, max_dist_percentage, lines_per_file
from dataset import highpass_filter, bandpass_filter
import pandas as pd

'''
Plotting program for ECG signals, also marks some regions where the program might have messed up for optional human 
review, though the final post_processing.py program skips over the messed up regions.
'''

root = tk.Tk()  # Prompts user to select file
currdir = os.getcwd()
par = Path(currdir).parent
signal_dir = str(par) + r"\Training"
root.filename = filedialog.askopenfilename(initialdir=signal_dir, title="Select file",
                                           filetypes=(("txt files", "*.txt"), ("all files", "*.*")))
matplotlib.use('Qt5Agg')
filename = root.filename
if len(filename) == 0:
    exit(0)
file_size = os.stat(filename).st_size
root.destroy()

reader_pd = pd.read_csv(filename, header=None, usecols=[0, 1], engine='c', encoding_errors='ignore', sep=' ')

file = open(filename, 'r+')  # Gets an average line size for the progress bar
fig, axs = plt.subplots()
file_loc = file.tell()
temp_line = file.readline()
file.seek(file_loc)
line_size = len(temp_line.encode('utf-8'))

ecg = np.array(reader_pd[0])  # Raw ECG signal
signals = []  # Indices of peaks in signals
signal = np.array(reader_pd[1])
ind1 = 0
ind2 = 1000


class Events:
    """Class for an interactive pyplot to handle click, scroll, etc. events."""
    def __init__(self):
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
                remove(ind, radius=rad)

    def scroll(self, event):
        """Just allows for scrolling through the plot."""
        global ind1, ind2
        dist = -event.step * (ind2 - ind1)
        ind1 += dist // 6
        ind2 += dist // 6
        line.set_ydata(signal[ind1:ind2])
        plt.draw()
        position_slider.valinit = (ind1 + ind2) / 2
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
    signal[ind] = 1


def remove(ind, radius=2):
    """Removes signals from the plot at some index with some 'radius' around it."""
    signal[ind - radius:ind + radius] = 0


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


dist = 0
num = 0
events = Events()
first = True
last_few = deque(maxlen=1)

total_marks = 0  # Counts of how many marks there are (as well as errors)
ones = np.where(signal == 1)

filtered_ecg = highpass_filter(ecg, order, low_cutoff, nyq)

ecg_line, = axs.plot(ecg[ind1:ind2], zorder=101)
filtered_line, = axs.plot(filtered_ecg[ind1:ind2], zorder=101)
line, = axs.plot(signal[ind1:ind2])

leg = axs.legend(['ECG', 'Filtered_ECG', 'Signal'], loc='upper left')
axs.axis([0, 6000, -0.5, 1])

lines = [ecg_line, filtered_line, line]
map_legend_to_ax = {}  # Will map legend lines to original lines.

pickradius = 5  # Points (Pt). How close the click needs to be to trigger an event.

for legend_line, ax_line in zip(leg.get_lines(), lines):
    legend_line.set_picker(pickradius)  # Enable picking on the legend line.
    map_legend_to_ax[legend_line] = ax_line


def on_pick(event):
    # On the pick event, find the original line corresponding to the legend
    # proxy line, and toggle its visibility.
    legend_line = event.artist

    # Do nothing if the source of the event is not a legend line.
    if legend_line not in map_legend_to_ax:
        return

    ax_line = map_legend_to_ax[legend_line]
    visible = not ax_line.get_visible()
    ax_line.set_visible(visible)
    # Change the alpha on the line in the legend, so we can see what lines
    # have been toggled.
    legend_line.set_alpha(1.0 if visible else 0.2)
    fig.canvas.draw()


rad = plt.axes([0.4, 0.01, 0.1, 0.075])  # Other interactive stuff, (buttons, sliders, event handlers, etc.)
width_slider_pos = plt.axes([0.2, 0.9, 0.65, 0.03])
position_slider_pos = plt.axes([0.2, 0.95, 0.65, 0.03])
stat = RadioButtons(rad, ('Browse', 'Add', 'Delete', 'Clean Region'))
toggle_clean_selector.RS = RectangleSelector(axs, events.clean_region, button=1)
stat.on_clicked(events.change_mode)
width_slider = Slider(width_slider_pos, 'Width', 200, 10000, valinit=3000)
width_slider.on_changed(events.set_width)
position_slider = Slider(position_slider_pos, 'Position', 0, len(signal), valinit=0)
position_slider.on_changed(events.set_pos)

scroll = fig.canvas.mpl_connect('scroll_event', events.scroll)
add_click = fig.canvas.mpl_connect('button_press_event', events.add_onclick)
delete_click = fig.canvas.mpl_connect('button_press_event', events.delete_onclick)
clean_region = fig.canvas.mpl_connect('button_press_event', toggle_clean_selector)
fig.canvas.mpl_connect('pick_event', on_pick)

plt.show()

file.close()

df = pd.DataFrame([ecg, signal])
# df.to_csv(filename, sep=' ', header=False, index=False)
