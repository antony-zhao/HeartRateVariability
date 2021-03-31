import os
import tkinter
from matplotlib import pyplot as plt
import mmap
import re
from matplotlib.widgets import Button, Slider, RadioButtons, RectangleSelector
from collections import deque
import numpy as np
import tkinter as tk
from tkinter import filedialog
import json
from scipy.signal import filtfilt, butter
import matplotlib

'''

'''
config_file = open("config.json", "r")
config = json.load(config_file)
interval_length = config["interval_length"]
step = config["step"]
stack = config["stack"]
scale_down = config["scale_down"]
datapoints = config["datapoints"]
max_dist_percentage = config["max_dist_percentage"]


T = 0.1          # Sample Period
fs = 4000.0      # sample rate, Hz
low_cutoff = 5      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
high_cutoff = 200
nyq = 0.5 * fs   # Nyquist Frequency
order = 4        # sin wave can be approx represented as quadratic
n = int(T * fs)  # total number of samples
# b, a = butter(N=order, Wn=low_cutoff/nyq, btype='low', analog=False)
b, a = butter(N=order, Wn=[low_cutoff/nyq, high_cutoff/nyq], btype='bandpass', analog=False)

# b, a = butter(N=order, Wn=[low_cutoff/nyq, high_cutoff/nyq], btype='bandpass', analog=False)
root = tk.Tk()
currdir = os.getcwd()
root.filename = filedialog.askopenfilename(initialdir=currdir + "/../Signal", title="Select file",
                                           filetypes=(("txt files", "*.txt"), ("all files", "*.*")))
matplotlib.use('TkAgg')
filename = root.filename
root.destroy()

file = open(os.path.join('..', 'Signal', filename), 'r+')
fig, axs = plt.subplots()

ecg = []
signal = []

signals = []


class Events:
    def __init__(self):
        self.ind_unmarked = -1
        self.ind_mismarked = -1
        self.unmarked = []
        self.mismarked = []
        self.dist = []
        self.to_be_deleted = []
        self.prev_ann = None
        self.x_right = 6000
        self.x_left = 0
        self.actions = []
        self.adding = False
        self.removing = False
        self.clean = False
        self.width = 3000

    def next_unmarked(self, event):
        if self.prev_ann is not None:
            self.prev_ann.remove()
        self.ind_unmarked = (self.ind_unmarked + 1) % len(self.unmarked)
        val, dist = self.unmarked[self.ind_unmarked]
        length = min(2000, dist * 50)
        if length < 600:
            length = 200
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
        if self.prev_ann is not None:
            self.prev_ann.remove()
        self.ind_mismarked = (self.ind_mismarked + 1) % len(self.mismarked)
        val, dist = self.mismarked[self.ind_mismarked]
        length = min(2000, dist * 50)
        if length < 600:
            length = 200
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
        if self.prev_ann is not None:
            self.prev_ann.remove()
        self.ind_unmarked = (self.ind_unmarked - 1) % len(self.unmarked)
        val, dist = self.unmarked[self.ind_unmarked]
        length = min(2000, dist * 50)
        if length < 600:
            length = 200
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
        if self.prev_ann is not None:
            self.prev_ann.remove()
        self.ind_mismarked = (self.ind_mismarked - 1) % len(self.mismarked)
        val, dist = self.mismarked[self.ind_mismarked]
        length = min(2000, dist * 50)
        if length < 600:
            length = 200
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

    def prev(self, event):
        pass

    def delete_onclick(self, event):
        if self.removing:
            try:
                ind = int(event.xdata)
            except TypeError:
                return
            if ind != 0:
                # rad = int(self.dist[self.ind] // 50 + 3)
                rad = 8
                self.actions.append(('delete', ind, rad))
                remove(ind, radius=rad)
                line.set_ydata(signal)
                plt.draw()

    def scroll(self, event):
        dist = -event.step * (self.x_right - self.x_left)
        self.x_left += dist / 6
        self.x_right += dist / 6
        axs.axis([self.x_left, self.x_right, -0.5, 1])
        plt.draw()
        position_slider.valinit = (self.x_right + self.x_left) / 2
        position_slider.reset()

    def add_onclick(self, event):
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
        mid = (self.x_left + self.x_right) / 2
        self.x_left = mid - val
        self.x_right = mid + val
        axs.axis([self.x_left, self.x_right, -0.5, 1])
        plt.draw()

    def change_mode(self, label):
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
        width = self.x_right - self.x_left
        self.x_left = val - width / 2
        self.x_right = val + width / 2
        axs.axis([self.x_left, self.x_right, -0.5, 1])
        plt.draw()


def add(ind):
    for i in range(-2, 2):
        signal[ind + i] = 1


def remove(ind, radius=2):
    for i in range(-radius, radius):
        signal[ind + i] = 0


def clean(x1, x2):
    for i in range(x1, x2 + 1):
        signal[i] = 0


def toggle_clean_selector(event):
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

total_marks = 0
mismarked = 0
unmarked_regions = 0
prev = 0

for i, line in enumerate(file):
    dist += 1
    temp = re.findall('([-0-9.]+)', line)
    ecg.append(float(temp[-2]))
    signal.append(int(temp[-1]))
    if int(temp[-1]) == 1:
        total_marks += 1
        if dist > (1 - max_dist_percentage) * interval_length or dist == 1 or first:  # This would mean that the signal is correct
            if dist == 1:  # For the areas where the signal is marked multiple times
                dist = 0
                total_marks -= 1
            else:
                if dist > (2 - 2 * max_dist_percentage) * interval_length:  # This indicates that the gap is too large
                    unmarked_regions += 1
                    events.unmarked.append((prev + interval_length, interval_length))
                    axs.annotate("*", (prev + interval_length, -0.2))
                elif (1 + max_dist_percentage) * interval_length < dist < (2 - 2 * max_dist_percentage):    # For when one beat is missed and the next one is also wrong
                    if i - prev > 1:
                        events.mismarked.append((i, interval_length))
                        axs.annotate("#", (i, -0.2))
                        mismarked += 1
                    prev = i
                    continue
                signals.append(i)  # indices of signals

        else:
            events.mismarked.append((i, dist))  # These are the mismarked signals
            axs.annotate("#", (i, -0.2))
            mismarked += 1
        if (1 + max_dist_percentage) * interval_length > dist > (1 - max_dist_percentage) * interval_length:
            last_few.append(dist)  # add the distance to the running average
        dist = 0  # Reset distance between last and current signal
        if first:
            first = False  # handling first signal
        if len(last_few) == 8:
            interval_length = np.mean(last_few)  # running average of rr interval
        prev = i

    if int(temp[-1]) == 2:
        if x1 is None:
            x1 = i
        else:
            x2 = i

    if int(temp[-1]) == 0 and x2 is not None:
        axs.axvspan(xmin=x1, xmax=x2, ymin=-0.5, ymax=1, color='#d62728', zorder=100)
        x1 = None
        x2 = None

plt.text(0.5, -0.3, "Mismarked: {} \n Unmarked Regions : {} \n Total: {}".format(mismarked, unmarked_regions, total_marks),
         bbox=dict(facecolor='red', alpha=0.5))


T = 0.1          # Sample Period
fs = 4000.0      # sample rate, Hz
low_cutoff = 200      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
high_cutoff = 5
nyq = 0.5 * fs   # Nyquist Frequency
order = 4        # sin wave can be approx represented as quadratic
n = int(T * fs)  # total number of samples
b, a = butter(N=order, Wn=low_cutoff/nyq, btype='low', analog=False)
ecg = filtfilt(b, a, np.asarray(ecg))
b, a = butter(N=order, Wn=high_cutoff/nyq, btype='high', analog=False)
ecg = filtfilt(b, a, np.asarray(ecg))

axs.plot(range(len(ecg)), ecg, zorder=101)
line, = axs.plot(range(len(signal)), signal)

axs.legend(["ECG", "Signal"], loc='upper left')
axs.axis([0, 6000, -0.5, 1])

if len(events.unmarked) > 0:
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

rad = plt.axes([0.4, 0.01, 0.1, 0.075])
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

file_mm = mmap.mmap(file.fileno(), 0)
line_length = len(file_mm.readline())

for action, val1, val2 in events.actions:
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
    elif action == 'mark':
        x1 = val1
        x2 = val2
        for i in range(x1, x2):
            file_mm[line_length * (1 + i) - 3] = ord('2')
    elif action == 'clean':
        x1 = val1
        x2 = val2
        for i in range(x1, x2):
            file_mm[line_length * (1 + i) - 3] = ord('0')

file_mm.close()
file.close()
