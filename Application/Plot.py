import os
from matplotlib import pyplot as plt
import mmap
import re
from matplotlib.widgets import Button, Slider, TextBox
from collections import deque
import numpy as np

file = open(os.path.join('..', 'Signal', 'T21_transition example1_180s1.txt'), 'r+')
fig, axs = plt.subplots()

file.readline()
ecg = []
signal = []

signals = []
AVG_RR = 400


class Events:
    def __init__(self):
        self.ind = -1
        self.uncertain = []
        self.dist = []
        self.to_be_deleted = []
        self.prev_ann = None
        self.x_right = 6000
        self.x_left = 0
        self.actions = []
        self.adding = False
        self.removing = False
        self.width = 3000

    def append(self, ind, dist):
        self.uncertain.append(ind)
        self.dist.append(dist)

    def toggle_add(self, event):
        if not self.adding:
            self.removing = False
        self.adding = not self.adding
        if self.adding and not self.removing:
            text.set_val("Adding")
        if not self.adding and not self.removing:
            text.set_val("")

    def toggle_remove(self, event):
        if not self.removing:
            self.adding = False
        self.removing = not self.removing
        if self.removing and not self.adding:
            text.set_val("Deleting")
        if not self.adding and not self.removing:
            text.set_val("")

    def next(self, event):
        if self.prev_ann is not None:
            self.prev_ann.remove()
        self.ind = (self.ind + 1) % len(self.uncertain)
        val = self.uncertain[self.ind]
        dist = self.dist[self.ind]
        length = min(2000, dist * 50)
        if length < 600:
            length = 200
        axs.axis([val - length, val + length, -0.5, 1])
        self.x_left = val - length
        self.x_right = val + length
        self.prev_ann = axs.annotate("*", (val, -0.2))
        if self.ind == len(self.uncertain) - 1:
            axs.annotate('FINAL', (val, -0.25))
        slider.valinit = length
        slider.reset()
        plt.draw()

    def delete_onclick(self, event):
        if self.removing:
            ind = int(event.xdata)
            if ind != 0:
                rad = int(self.dist[self.ind] // 50 + 2)
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

    def add_onclick(self, event):
        if self.adding:
            ind = int(event.xdata)
            if ind != 0:
                self.actions.append(('add', ind, 2))
                add(ind)
                line.set_ydata(signal)
                plt.draw()

    def set_width(self, val):
        mid = (self.x_left + self.x_right) / 2
        self.x_left = mid - val
        self.x_right = mid + val
        axs.axis([self.x_left, self.x_right, -0.5, 1])
        plt.draw()


def add(ind):
    for i in range(-2, 2):
        signal[ind + i] = 1


def remove(ind, radius=2):
    for i in range(-radius, radius):
        signal[ind + i] = 0


dist = 0
num = 0
events = Events()
first = True
last_few = deque(maxlen=8)
for i, line in enumerate(file):
    dist += 1
    temp = re.findall('([-0-9.]+)', line)
    ecg.append(float(temp[0]))
    signal.append(int(temp[1]))
    if int(temp[1]) == 1:
        if dist > 0.8 * AVG_RR or dist == 1 or first:
            if dist == 1:
                dist = 0
            else:
                if dist > 1.3 * AVG_RR:
                    for j in range(prev + int(AVG_RR), i - int(AVG_RR)//2, int(AVG_RR)):
                        events.append(j, AVG_RR)
                signals.append(i)
        else:
            events.append(i, dist)
        if 1.5 * AVG_RR > dist > 0.7 * AVG_RR:
            last_few.append(dist)
        dist = 0
        if first:
            first = False
        prev = i
        if len(last_few) == 8:
            AVG_RR = np.mean(last_few)

axs.plot(range(len(ecg)), ecg)
line, = axs.plot(range(len(signal)), signal)
axs.legend(["ECG", "Signal"], loc='upper left')
axs.axis([0, 6000, -0.5, 1])

if len(events.uncertain) > 0:
    button1 = plt.axes([0.7, 0.01, 0.1, 0.075])
    next_button = Button(button1, 'Next')
    next_button.on_clicked(events.next)

button2 = plt.axes([0.6, 0.01, 0.1, 0.075])
button3 = plt.axes([0.5, 0.01, 0.1, 0.075])
slider_pos = plt.axes([0.2, 0.9, 0.65, 0.03])
status = plt.axes([0.1, 0.01, 0.1, 0.075])
delete_button = Button(button2, 'Delete')
add_button = Button(button3, 'Add')
slider = Slider(slider_pos, 'Width', 200, 10000, valinit=3000)
text = TextBox(status, "")
delete_button.on_clicked(events.toggle_remove)
add_button.on_clicked(events.toggle_add)
slider.on_changed(events.set_width)

figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

scroll = fig.canvas.mpl_connect('scroll_event', events.scroll)
click = fig.canvas.mpl_connect('button_press_event', events.add_onclick)
click = fig.canvas.mpl_connect('button_press_event', events.delete_onclick)

plt.show()

file.close()

file = open(os.path.join('..', 'Signal', 'T21_transition example1_180s1.txt'), 'rb+')

file_mm = mmap.mmap(file.fileno(), 0)

for action, val, rad in events.actions:
    print(action, val)
    if action == 'delete':
        for i in range(-rad, rad):
            file_mm[11 * (val + 1 + i) - 3] = ord('0')
    elif action == 'add':
        for i in range(-rad, rad):
            file_mm[11 * (val + 1 + i) - 3] = ord('1')


file_mm.close()
file.close()
