import scipy.special
from matplotlib import pyplot as plt
import numpy as np
import os
import re
from scipy.special import expit
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow.keras.models import load_model

from dataset import process_ecg, filters
import time
from collections import deque
import tkinter as tk
from tkinter import filedialog
import tqdm
from pathlib import Path
from config import window_size, stack, scale_down, datapoints, \
    lines_per_file, max_dist_percentage, low_cutoff, high_cutoff, nyq, order, interval_length, threshold, animal, \
    pad_behind

tf.keras.backend.clear_session()
np.seterr(all='raise')

file_num = 1
update_freq = 1
loading_size = 100
# model = keras.models.load_model(f'{animal}_model', compile=False)
model = load_model(f'{animal}_model_val_top_k', compile=False)
model.summary()

# Opening file and choosing directory to save code in
root = tk.Tk()
currdir = os.getcwd()
par = Path(currdir).parent
root.filename = filedialog.askopenfilename(initialdir=str(par) + r'\ECG_Data', title='Select file',
                                           filetypes=(('ascii files', '*.ascii'), ('txt files', '*.txt'),
                                                      ('all files', '*.*')))
filename = root.filename
if len(filename) == 0:
    exit(0)
file = open(os.path.join('..', 'ECG_Data', filename), 'r')
file_size = os.stat(filename).st_size
root.destroy()

root = tk.Tk()
root.withdraw()
folder_selected = filedialog.askdirectory()

filename = filename[len(filename) - filename[::-1].index('/'):filename.index('.')]  # Gets the name of the file itself
# without the full path
f = open(os.path.join(folder_selected, filename + '{:03}'.format(file_num) + '.txt'), 'w')  # Opens the first of the
# files of which the data will be saved into. Whenever the number of lines reaches a certain point, file_num gets
# incremented and a new file with the new number is created.

# Start timer (displays time elapsed in the end)
start = time.time()

lines = 0  # Tracks the number of lines for the current file
dist = 0
first = True  # For handling the first signal
average_interval = deque(maxlen=1)  # Last 10 interval lengths, used to find the running average
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
        # otherwise this just reads the signal value
        ecg[i] = 0 if temp == 'x' else float(temp)
        datetime.append(line[:line.index(',')])  # The time value, used for post_processing later so it is preserved
        # and transferred to the output file.
    return datetime, ecg, e


def write_signal(sig_file, datetime, sig, ecg, activation=None):
    """Writes the output signals (the peak detection) into a file as either a 1 or 0."""
    global dist
    global first
    # plt.plot(sig)
    if activation is not None:
        sig = activation(sig)

    # if lines >= 0:
    #     plt.plot(ecg)  # Just left in, uncomment if you want a visualization of the data for testing purposes.
    #     plt.plot(sig)
    #     plt.plot([threshold for _ in range(window_size)])
    #     plt.show()
    # The maximum amount we think the signal can differ by, our default is 0.2, so we don't believe any 'signal'
    # with a distance of 0.8-1.2 from the previous is real, and so we omit it.
    min_dist = 1 - max_dist_percentage
    max_dist = 1 + max_dist_percentage

    sorted_inds = np.argsort(sig)
    temp = []
    for i in range(len(datetime)):
        d = datetime[i]
        e = ecg[i]
        s_ = sig[i]
        if i == sorted_inds[-1]:  # Minimum value of the signal before other checks. May need to adjust this value.
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
        sig_file.write('{},{:>8},{}\n'.format(d, '{:2.5f}'.format(e), '{:1.2f}'.format(s_), int(s)))
        temp.append(s)
        dist += 1
    temp = np.array(temp)
    plt.plot(sig)
    plt.plot(ecg)
    plt.plot(temp)
    plt.show()
    return len(datetime)


ecg_segment = []  # Acts as a buffer for the ecg signals
datetime_segment = []  # Similarly for datetimes
ecg_temp = []  # ecg and datetimes of the portion being processed
datetime = []
signal = np.zeros(window_size)

# Skips through the header if it exists, goes until it reads a line of data (probably need to change regex depending on
# data)
read = ''
while re.match('\\s*[\\d\\s/:.APM]+,[-\\s\\dx.,]*[-\\s\\dx.]\n', read) is None:
    read = file.readline()

file_loc = file.tell()
temp_line = file.readline()
file.seek(file_loc)
line_size = len(temp_line.encode('utf-8'))  # Gets the average size of the line for the progress bar
# to track approximately where we are within the file.

# Reads a long segment of data for the filters, and slowly 'scrolls' through, removing the data that
datetime_segment, ecg_segment, EOF = read_ecg(file, window_size * loading_size)
filtered_segment = filters(ecg_segment, order, low_cutoff, high_cutoff, nyq)

# Pads the data with 0s
padding = (pad_behind * window_size, 0)
ecg_segment = np.pad(ecg_segment, padding, constant_values=(0, 0))
filtered_segment = np.pad(filtered_segment, padding, constant_values=(0, 0))

# Initializes the temporary variables and slides the window over
ecg_temp = ecg_segment[:window_size * stack]
filtered_temp = filtered_segment[:window_size * stack]
datetime = datetime_segment[:window_size]
ecg_segment = ecg_segment[window_size:]
filtered_segment = filtered_segment[window_size:]
datetime_segment = datetime_segment[window_size:]

ind1 = pad_behind * window_size
ind2 = (pad_behind + 1) * window_size
curr_segment = ecg_temp[ind1:ind2]
curr_filt = filtered_temp[ind1:ind2]
batch = []
datetimes = []
ecg_segments = []

with tqdm.tqdm(total=file_size) as pbar:  # Progress bar
    pbar.set_description('Bytes ')
    iter = 0  # Used for the progress bar to update it every so often.
    while True:
        num_lines = 0

        temp = process_ecg(ecg_temp, filtered_temp, scale_down, stack, datapoints)
        temp = temp[np.newaxis, :, :]
        temp = np.swapaxes(temp, 1, 2)
        batch.append(temp)
        datetimes.append(datetime)
        ecg_segments.append(curr_segment)
        if len(batch) == 256:
            batch = np.array(batch)[:, 0, :, :]
            signals = model(batch, training=False).numpy()

            for signal, datetime_iter, segment in zip(signals, datetimes, ecg_segments):
                num_lines = write_signal(f, datetime_iter, signal, segment, activation=scipy.special.softmax)
                lines += num_lines
            batch = []
            datetimes = []
            ecg_segments = []

        # Steps through all the different buffers, and if the buffer is empty reads more from the file.
        if ecg_segment.size > stack * window_size:
            ecg_segment = ecg_segment[window_size:]
            ecg_temp = ecg_segment[:window_size * stack]
            filtered_segment = filtered_segment[window_size:]
            filtered_temp = filtered_segment[:window_size * stack]
            datetime_segment = datetime_segment[window_size:]
            datetime = datetime_segment[:window_size]
            curr_segment = ecg_temp[ind1:ind2]
            curr_filt = filtered_temp[ind1:ind2]
        elif not EOF:
            iter += 1
            read = read_ecg(file, window_size * loading_size)
            if iter % update_freq == 0:
                pbar.update(line_size * len(read[0]) * update_freq)

            datetime_segment = read[0]
            ecg_segment = np.append(ecg_segment, read[1])
            EOF = read[2]

            filtered_segment = filters(ecg_segment, order, low_cutoff, high_cutoff, nyq)
            ecg_segment = np.pad(ecg_segment, padding, constant_values=(0, 0))
            filtered_segment = np.pad(filtered_segment, padding, constant_values=(0, 0))

            ecg_temp = ecg_segment[:window_size * stack]
            filtered_temp = filtered_segment[:window_size * stack]
            datetime = datetime_segment[:window_size]
            ecg_segment = ecg_segment[window_size:]
            filtered_segment = filtered_segment[window_size:]
            datetime_segment = datetime_segment[window_size:]

            ind1 = pad_behind * window_size
            ind2 = (pad_behind + 1) * window_size
            curr_segment = ecg_temp[ind1:ind2]
            curr_filt = filtered_temp[ind1:ind2]
        else:
            break

        if lines > lines_per_file:  # Creates a new file in case the current one has enough lines.
            lines = 0
            file_num += 1
            f.close()

            f = open(os.path.join('..', folder_selected, filename + '{:03}'.format(file_num) + '.txt'), 'w')


# Writes the final few datapoints into the file, and closes it
for signal, datetime_iter, segment in zip(signals, datetimes, ecg_segments):
    num_lines = write_signal(f, datetime_iter, signal, segment, activation=expit)
end = time.time()


f.close()

print('elapsed time: ' + str(end - start) + ' seconds')
# input('Press enter to continue (may need 2 times)')

del model
tf.keras.backend.clear_session()
