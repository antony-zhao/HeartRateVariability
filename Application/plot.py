import os
from matplotlib import pyplot as plt
import mmap
import re
from matplotlib.widgets import Button, Slider, RadioButtons, RectangleSelector
from collections import deque
import numpy as np
import tkinter as tk
from tkinter import filedialog
from scipy.signal import filtfilt, butter
import matplotlib
from pathlib import Path
import tqdm
from config import interval_length, low_cutoff, high_cutoff, nyq, order, max_dist_percentage

'''
Plotting program for ECG signals, also marks some regions where the program might have messed up for optional human 
review, though the final post_processing.py program skips over the messed up regions.
'''
b, a = butter(N=order, Wn=[low_cutoff/nyq, high_cutoff/nyq], btype='bandpass', analog=False)  # Filter parameters

root = tk.Tk()  # Prompts user to select file
currdir = os.getcwd()
par = Path(currdir).parent
signal_dir = str(par) + r"\ECG_Data"
root.filename = filedialog.askopenfilename(initialdir=signal_dir, title="Select file",
                                           filetypes=(("txt files", "*.txt"), ("all files", "*.*")))
matplotlib.use('Qt5Agg')
filename = root.filename
file_size = os.stat(filename).st_size
root.destroy()

file = open(filename, 'r+')  # Gets an average line size for the progress bar
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
        self.ind_unmarked = -1  # Which error is being viewed (index of unmarked)
        self.ind_mismarked = -1  # Index of mismarked
        self.unmarked = []  # Indices of the unmarked and mismarked peaks
        self.mismarked = []
        self.default_length = 15 * interval_length
        self.x_right = 15 * interval_length  # Right and left bounds of window
        self.x_left = 0
        self.actions = []  # Actions to be processed into the file after the plot is closed
        self.adding = False  # Following are toggling which of the modes are on
        self.removing = False
        self.clean = False

    def next_unmarked(self, event):
        """Jumps to the next unmarked region in the plot"""
        self.ind_unmarked = (self.ind_unmarked + 1) % len(self.unmarked)
        val, dist = self.unmarked[self.ind_unmarked]
        length = min(self.default_length, dist * 50)
        if length < int(interval_length * 1.5):
            length = interval_length // 2
        axs.axis([val - length, val + length, -0.5, 1])
        self.x_left = val - length
        self.x_right = val + length
        if self.ind_unmarked == len(self.unmarked) - 1:
            axs.annotate('FINAL', (val, -0.25))
        width_slider.valinit = length
        width_slider.reset()
        position_slider.valinit = val
        position_slider.reset()
        plt.draw()

    def next_mismarked(self, event):
        """Jumps to next mismarked region in the graph."""
        self.ind_mismarked = (self.ind_mismarked + 1) % len(self.mismarked)
        val, dist = self.mismarked[self.ind_mismarked]
        length = min(self.default_length, dist * 50)
        if length < int(interval_length * 1.5):
            length = interval_length // 2
        axs.axis([val - length, val + length, -0.5, 1])
        self.x_left = val - length
        self.x_right = val + length
        if self.ind_mismarked == len(self.mismarked) - 1:
            axs.annotate('FINAL', (val, -0.25))
        width_slider.valinit = length
        width_slider.reset()
        position_slider.valinit = val
        position_slider.reset()
        plt.draw()

    def previous_unmarked(self, event):
        """Jumps to previous unmarked region"""
        self.ind_unmarked = (self.ind_unmarked - 1) % len(self.unmarked)
        val, dist = self.unmarked[self.ind_unmarked]
        length = min(self.default_length, dist * 50)
        if length < int(interval_length * 1.5):
            length = interval_length // 2
        axs.axis([val - length, val + length, -0.5, 1])
        self.x_left = val - length
        self.x_right = val + length
        if self.ind_unmarked == len(self.unmarked) - 1:
            axs.annotate('FINAL', (val, -0.25))
        width_slider.valinit = length
        width_slider.reset()
        position_slider.valinit = val
        position_slider.reset()
        plt.draw()

    def previous_mismarked(self, event):
        """Jumps to previous mismarked region"""
        self.ind_mismarked = (self.ind_mismarked - 1) % len(self.mismarked)
        val, dist = self.mismarked[self.ind_mismarked]
        length = min(self.default_length, dist * 50)
        if length < int(interval_length * 1.5):
            length = interval_length // 2
        axs.axis([val - length, val + length, -0.5, 1])
        self.x_left = val - length
        self.x_right = val + length
        if self.ind_mismarked == len(self.mismarked) - 1:
            axs.annotate('FINAL', (val, -0.25))
        width_slider.valinit = length
        width_slider.reset()
        position_slider.valinit = val
        position_slider.reset()
        plt.draw()

    def delete_onclick(self, event):
        """Delete a mark (actually deletes an area around the click since marks can be multiple values)."""
        if self.removing:
            try:
                ind = int(event.xdata)
            except TypeError:
                return
            if ind != 0:
                rad = 8
                self.actions.append(('delete', ind, rad))
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
                self.actions.append(('add', ind, 2))
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
                for i in range(min(x1, x2), max(x1, x2)):
                    self.actions.append(('delete', i, 1))
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


dist = 0
num = 0
events = Events()
first = True
last_few = deque(maxlen=8)
x1 = None
x2 = None

total_marks = 0  # Counts of how many marks there are (as well as errors)
mismarked = 0
unmarked_regions = 0
prev = 0

every_i = 100000
with tqdm.tqdm(total=file_size) as pbar:  # Progress bar
    pbar.set_description('Bytes ')
    for i, line in enumerate(file):
        if i % every_i == 0:
            pbar.update(line_size * every_i)
        dist += 1
        temp = re.findall('([-0-9.]+)', line)  # Regex, 2nd to last is the ECG, and last is the signal (others are
        # date values which are kept in for post_processing.py)
        ecg.append(float(temp[-2]))
        signal.append(int(temp[-1]))
        if int(temp[-1]) == 1:
            total_marks += 1
            if dist > (1 - max_dist_percentage) * interval_length or dist == 1 or first:  # This would mean that the
                # signal is correct
                if dist == 1:  # For the areas where the signal is marked multiple times
                    dist = 0
                    total_marks -= 1
                else:
                    if dist > (2 - 2 * max_dist_percentage) * interval_length:  # This indicates that the gap is too
                        # large
                        unmarked_regions += 1
                        events.unmarked.append((prev + interval_length, interval_length))
                        axs.annotate("*", (prev + interval_length, -0.2))
                    elif (1 + max_dist_percentage) * interval_length < dist < (2 - 2 * max_dist_percentage):    # For
                        # when one beat is missed and the next one is also wrong
                        if i - prev > 1:
                            events.mismarked.append((i, interval_length))
                            axs.annotate("#", (i, -0.2))
                            mismarked += 1
                        prev = i
                        continue
                    signals.append(i)  # Indices of signals

            else:
                events.mismarked.append((i, dist))  # These are the mismarked signals
                axs.annotate("#", (i, -0.2))
                mismarked += 1
            if (1 + max_dist_percentage) * interval_length > dist > (1 - max_dist_percentage) * interval_length:
                last_few.append(dist)  # Add the distance to the running average
            dist = 0  # Reset distance between last and current signal
            if first:
                first = False  # handling first signal
            if len(last_few) == 8:
                interval_length = np.mean(last_few)  # Running average of rr interval
            prev = i

plt.text(0.5, -0.3, 'Mismarked: {} \n Unmarked Regions : {} \n Total: {}'
         .format(mismarked, unmarked_regions, total_marks), bbox=dict(facecolor='red', alpha=0.5))

b, a = butter(N=order, Wn=low_cutoff/nyq, btype='low', analog=False)
ecg = filtfilt(b, a, np.asarray(ecg))
b, a = butter(N=order, Wn=high_cutoff/nyq, btype='high', analog=False)
ecg = filtfilt(b, a, np.asarray(ecg))

axs.plot(range(len(ecg)), ecg, zorder=101)
line, = axs.plot(range(len(signal)), signal)

axs.legend(['ECG', 'Signal'], loc='upper left')
axs.axis([0, 6000, -0.5, 1])

if len(events.unmarked) > 0:  # Only display unmarked and mismarked buttons if there are unmarked/mismarked signals
    button1 = plt.axes([0.6, 0.01, 0.1, 0.075])
    next_un_button = Button(button1, 'Next Unmarked')
    next_un_button.on_clicked(events.next_unmarked)
    button2 = plt.axes([0.5, 0.01, 0.1, 0.075])
    prev_un_button = Button(button2, 'Previous Unmarked')
    prev_un_button.on_clicked(events.previous_unmarked)

if len(events.mismarked) > 0:
    button3 = plt.axes([0.8, 0.01, 0.1, 0.075])
    next_mis_button = Button(button3, 'Next Mismarked')
    next_mis_button.on_clicked(events.next_mismarked)
    button4 = plt.axes([0.7, 0.01, 0.1, 0.075])
    prev_mis_button = Button(button4, 'Previous Mismarked')
    prev_mis_button.on_clicked(events.previous_mismarked)

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

file = open(os.path.join('..', 'Signal', filename), 'rb+')

file_mm = mmap.mmap(file.fileno(), 0)  # Allows us to quickly navigate through the file to modify it
line_length = len(file_mm.readline())

for action, val1, val2 in events.actions:  # Modifies the signal values depending on what the user did in the plot.
    if action == 'delete':
        ind = val1
        rad = val2
        for i in range(-rad, rad):
            if file_mm[line_length * (ind + 1 + i) - 3] == ord('2'):
                continue
            else:
                file_mm[line_length * (ind + 1 + i) - 3] = ord('0')
    elif action == 'add':
        ind = val1
        rad = val2
        for i in range(-rad, rad):
            if file_mm[line_length * (ind + 1 + i) - 3] == ord('2'):
                continue
            else:
                file_mm[line_length * (ind + 1 + i) - 3] = ord('1')
    elif action == 'clean':
        x1 = val1
        x2 = val2
        for i in range(x1, x2):
            file_mm[line_length * (1 + i) - 3] = ord('0')

file_mm.close()
file.close()
