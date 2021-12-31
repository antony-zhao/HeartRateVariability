from matplotlib import pyplot as plt
import numpy as np
import os
import re
from scipy.signal import filtfilt, butter
import tensorflow as tf
from model import model
from dataset import preprocess_ecg
import time
from collections import deque
import tkinter as tk
from tkinter import filedialog
import tqdm
from pathlib import Path
from config import interval_length, step, stack, scale_down, datapoints, \
    lines_per_file, max_dist_percentage, low_cutoff, high_cutoff, nyq, order

tf.keras.backend.clear_session()
np.seterr(all='raise')

file_num = 1
update_freq = 10
signal = np.zeros(interval_length)
model.load_weights("model.h5")

# Opening file and choosing directory to save code in
root = tk.Tk()
currdir = os.getcwd()
par = Path(currdir).parent
root.filename = filedialog.askopenfilename(initialdir=str(par) + r"\ECG_Data", title="Select file",
                                           filetypes=(("ascii files", "*.ascii"), ("txt files", "*.txt"),
                                                      ("all files", "*.*")))
filename = root.filename
file = open(os.path.join('..', 'ECG_Data', filename), 'r')
file_size = os.stat(filename).st_size
root.destroy()

root = tk.Tk()
root.withdraw()
folder_selected = filedialog.askdirectory()
print(folder_selected)

filename = filename[len(filename) - filename[::-1].index("/"):filename.index(".")]  # Gets the name of the file itself
# without the full path
f = open(os.path.join(folder_selected, filename + '{:03}'.format(file_num) + '.txt'), 'w')  # Opens the first of the
# files of which the data will be saved into. Whenever the number of lines reaches a certain point, file_num gets
# incremented and a new file with the new number is created.

# Start timer (displays time elapsed in the end)
start = time.time()

ecg = []
lines = 0  # Tracks the number of lines for the current file
dist = 0
first = True  # ??
average_interval = deque(maxlen=10)  # Last 10 interval lengths, used to find the running average
average_interval.append(interval_length)


def read_ecg(ecg_file, count):
    """Reads 'count' lines from the ecg_file and returns them"""
    e = False  # End of file
    ecg = np.zeros(count)
    datetime = []
    for i in range(count):
        line = ecg_file.readline()
        if len(line) == 0:  # Signifies an end of the file
            e = True
            break
        temp = re.findall('([-0-9.x]+)', line)[-1]  # Sometimes x is in our data which is just an empty value,
        # otherwise this just reads the the signal value
        ecg[i] = 0 if temp == 'x' else float(temp)
        datetime.append(line[:line.index(',')])  # The time value, used for post_processing later so it is preserved
        # and transferred to the output file.
    return datetime, ecg.reshape(count, 1), e


def write_signal(sig_file, datetime, sig, ecg):
    """Writes the output signals (the peak detection) into a file as either a 1 or 0."""
    global dist
    global first
    # plt.plot(ecg)  # Just left in, uncomment if you want a visualization of the data for testing purposes.
    # plt.plot(sig)
    # plt.show()
    # The maximum amount we think the signal can differ by, our default is 0.2 so we don't believe any 'signal'
    # with a distance of 0.8-1.2 from the previous is real and so we omit it.
    min_dist = 1 - max_dist_percentage
    max_dist = 1 + max_dist_percentage
    for i in range(len(datetime)):
        d = datetime[i]
        e = ecg[i]
        s = sig[i]
        if s > 0.1:  # Minimum value of the signal before other checks.
            if dist < min_dist * np.mean(average_interval):
                if first:  # The very first signal
                    s = 1
                    first = False
                else:
                    s = 0
            else:
                s = 1
                if dist < np.mean(average_interval) * max_dist:
                    average_interval.append(dist)
                dist = 0
        else:
            s = 0
        sig_file.write('{},{:>8},{}\n'.format(d, '{:2.5f}'.format(e), int(s)))
        dist += 1
    return len(datetime)


ecg_temp = []
ecg_segment = []
datetime_segment = []
datetime = []
ecg_deque = deque(maxlen=stack)  # Variable which will eventually be converted to a numpy array and fed to the model.

for i in range(stack - 1):   # Initializes the values to just be 0's. (The dataset includes and the model has been
    # trained to deal with this, can also be treated the same as if the data were just blank in the case of missing
    # data)
    ecg_deque.append(np.zeros(datapoints, ))

# Skips through empty lines and comments, in our case comments are '#'
read = ""
while len(read) <= 1 or read[0] == "#":
    read = file.readline()

file_loc = file.tell()
temp_line = file.readline()
file.seek(file_loc)
line_size = len(temp_line.encode('utf-8'))  # Gets the average size of the line for the progress bar
# to track approximately where we are within the file.

# Reads a long segment of data for the filters, and slowly 'scrolls' through, removing the data that
datetime_segment, ecg_segment, EOF = read_ecg(file, interval_length * 10)
b, a = butter(N=order, Wn=low_cutoff / nyq, btype='low', analog=False)
ecg_segment = filtfilt(b, a, np.asarray(ecg_segment), axis=0)
b, a = butter(N=order, Wn=high_cutoff / nyq, btype='high', analog=False)
ecg_segment = filtfilt(b, a, np.asarray(ecg_segment), axis=0)
ecg_segment = ecg_segment.flatten()
ecg_temp = ecg_segment[:interval_length]
datetime = datetime_segment[:interval_length]
ecg_segment = ecg_segment[step:]
datetime_segment = datetime_segment[step:]

with tqdm.tqdm(total=file_size) as pbar:
    iter = 0  # Used for the progress bar to update it every so often.
    while True:
        num_lines = 0
        temp = preprocess_ecg(np.asarray(ecg_temp), scale_down)
        ecg_deque.append(temp)
        temp = np.swapaxes(np.asarray(ecg_deque)[np.newaxis, :, :], 1, 2)
        try:
            temp = temp / np.max(np.abs(temp))
        except FloatingPointError:
            pass
        # Blocked out code for visualizations on data
        # plt_temp = temp[0, :, 0]
        # for j in range(1, stack):
        #     plt_temp = np.append(plt_temp, temp[0, :, j][datapoints//(interval_length//step):])
        # plt.plot(plt_temp)
        temp = model(temp, training=False).numpy()
        # temp = np.zeros(interval_length,)
        # sig = np.sum(temp.reshape((-1, scale_down)), axis=1) / scale_down
        # ls = np.asarray([0] * (datapoints // (interval_length // step)) * (stack - 1))
        # temp_sig = np.append(ls, sig)
        # plt.plot(temp_sig)
        # plt.show(block=False)
        # plt.pause(0.5)
        # plt.close()
        temp = temp.reshape(interval_length, )
        max_ind = np.argmax(temp)
        # temp[min(interval_length, max_ind+1):] = 0
        # temp[:max(0, max_ind-1)] = 0

        # plt.plot(ecg_temp)
        # plt.plot(temp)
        # plt.show(block=False)
        # plt.pause(0.5)
        # plt.close()

        signal += temp / (interval_length / step)

        num_lines = write_signal(f, datetime[:step], signal[:step], ecg_temp[:interval_length - step])
        lines += num_lines

        signal[0:interval_length - step] = signal[step:]
        signal[interval_length - step:] = 0
        signal[signal < 0.1] = 0
        ecg_segment = ecg_segment[step:]
        if ecg_segment.size > 0:
            ecg_temp = ecg_temp[step:]
            ecg_temp = np.append(ecg_temp, ecg_segment[:step])
            datetime_segment = datetime_segment[step:]
            datetime = datetime[step:]
            datetime = np.append(datetime, datetime_segment[:step])
        elif not EOF:
            iter += 1
            datetime_segment, ecg_segment, EOF = read_ecg(file, interval_length * 10)
            if iter % update_freq == 0:
                pbar.update(line_size * len(datetime_segment) * update_freq)
            b, a = butter(N=order, Wn=low_cutoff / nyq, btype='low', analog=False)
            ecg_segment = filtfilt(b, a, np.asarray(ecg_segment), axis=0)
            b, a = butter(N=order, Wn=high_cutoff / nyq, btype='high', analog=False)
            ecg_segment = filtfilt(b, a, np.asarray(ecg_segment), axis=0)
            ecg_segment = ecg_segment.flatten()
            ecg_temp = ecg_temp[step:]
            ecg_temp = np.append(ecg_temp, ecg_segment[:step])
            datetime = datetime[step:]
            datetime = np.append(datetime, datetime_segment[:step])
        else:
            break

        if lines >= lines_per_file:  # Creates a new file in case the current one has enough lines.
            lines = 0
            file_num += 1
            f.close()

            f = open(os.path.join('..', folder_selected, filename + '{:03}'.format(file_num) + '.txt'), 'w')
write_signal(f, datetime[:interval_length - step], signal[:interval_length - step], ecg_temp[:interval_length - step])
end = time.time()
f.close()

print('elapsed time: ' + str(end - start) + ' seconds')
input("Press enter to continue")

del model
tf.keras.backend.clear_session()
