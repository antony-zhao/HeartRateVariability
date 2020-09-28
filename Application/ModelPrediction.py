from matplotlib import pyplot as plt
import numpy as np
import os
from Methods import *
import tensorflow as tf
from tensorflow.keras.models import load_model
from Model import model, load_model, interval_length, step, stack
import h5py
import time
from collections import deque
import tkinter as tk
from tkinter import filedialog
from scipy.signal import filtfilt, butter
import scipy.signal
from Parameters import lines_per_file

'''
Figure out why executable is slow


Deal with runaway averages.

Check logic with marking beats.
'''


T = 0.1          # Sample Period
fs = 4000.0      # sample rate, Hz
low_cutoff = 200      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
high_cutoff = 5
nyq = 0.5 * fs   # Nyquist Frequency
order = 4        # sin wave can be approx represented as quadratic
n = int(T * fs)  # total number of samples
b, a = butter(N=order, Wn=low_cutoff / nyq, btype='low', analog=False)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
model_file = 'Model.h5'

file_num = 1

root = tk.Tk()
currdir = os.getcwd()
root.filename = filedialog.askopenfilename(initialdir=currdir+"/../ECG_Data", title="Select file",
                                           filetypes=(("ascii files", "*.ascii"), ("txt files", "*.txt"),
                                                      ("all files", "*.*")))
filename = root.filename
file = open(os.path.join('..', 'ECG_Data', filename), 'r')
root.destroy()

root = tk.Tk()
root.withdraw()
folder_selected = filedialog.askdirectory()
print(folder_selected)

ecg = []
filepath = filename[:filename.index('ECG_Data')]
filename = filename[len(filename) - filename[::-1].index("/"):filename.index(".")]
f = open(os.path.join(filepath, 'Signal', filename + '{:03}'.format(file_num) + '.txt'), 'w')


signal = np.zeros(interval_length)
# with tf.device('/cpu:0'):
load_model(model_file)

start = time.time()
lines = 0
dist = 0
first = True
average_interval = deque(maxlen=10)
average_interval.append(interval_length)


def read_ecg(ecg_file, count):
    e = False
    sig = np.zeros(count)
    datetime = []
    for i in range(count):
        x = ecg_file.readline()
        if len(x) == 0:
            e = True
            break
        temp = re.findall('([-0-9.x]+)', x)[-1]
        sig[i] = 0 if temp == 'x' else float(temp)
        datetime.append(x[:x.index(',')])
    return datetime, sig.reshape(count, 1), e


def write_signal(sig_file, datetime, sig, ecg):
    global dist
    global first
    lines = 0
    # plt.plot(ecg)
    # plt.plot(sig)
    # plt.show()
    for i in range(len(datetime)):
        d = datetime[i]
        e = ecg[i][0]
        s = sig[i]
        lines += 1
        if s > 0.1:  # max(0.7, 1.5 / (interval_length / step)):
            if dist < 0.9 * np.mean(average_interval):
                if first:
                    s = 1
                    first = False
                else:
                    s = 0
            else:
                s = 1
                if 0.9 * np.mean(average_interval) < dist < np.mean(average_interval) * 1.1:
                    average_interval.append(dist)
                dist = 0
        else:
            s = 0
        sig_file.write('{},{:>8},{}\n'.format(d, '{:2.5f}'.format(e), int(s)))
        dist += 1
    return lines


ecg_temp = []
ecg_deque = deque(maxlen=stack)
for i in range(stack-1):
    ecg_deque.append(np.zeros(interval_length, ))

read = ""
while len(read) <= 1 or read[0] == "#":
    read = file.readline()

datetime, ecg_temp, EOF = read_ecg(file, interval_length)

while not EOF:
    num_lines = 0
    temp = np.asarray(ecg_temp).reshape(interval_length, )
    temp -= np.mean(temp)
    temp = filtfilt(b, a, np.asarray(temp))
    # b, a = butter(N=order, Wn=high_cutoff / nyq, btype='high', analog=False)
    # temp = filtfilt(b, a, np.asarray(temp))
    ecg_deque.append(temp)
    temp = np.swapaxes(np.asarray(ecg_deque)[np.newaxis, :, :], 1, 2)
    temp = (2 * temp / (np.nanmean(np.where(np.max(np.abs(temp), axis=1).reshape((-1, 1, stack)) != 0,
            np.max(np.abs(temp), axis=1).reshape((-1, 1, stack)), np.nan), axis=2).reshape((-1, 1, 1))
            + np.max(np.abs(temp), axis=1).reshape((-1, 1, stack))))
    # plt.plot(range(interval_length), temp[0, :, -1])
    temp = model.predict(temp)
    temp = temp.reshape(interval_length, )
    # plt.plot(range(interval_length), temp)
    max_ind = np.argmax(temp)
    temp[min(interval_length, max_ind+1):] = 0
    temp[:max(0, max_ind-1)] = 0

    # plt.plot(range(interval_length), temp)
    # plt.show(block=False)
    # plt.pause(0.5)
    # plt.close()

    signal += temp / (interval_length / step)

    num_lines = write_signal(f, datetime[:step], signal[:step], ecg_temp[:interval_length - step])
    lines += num_lines

    ecg_temp[:interval_length - step] = ecg_temp[step:]
    datetime[:interval_length - step] = datetime[step:]
    datetime[interval_length - step:], ecg_temp[interval_length - step:], EOF = read_ecg(file, step)
    signal[0:interval_length - step] = signal[step:]
    signal[interval_length - step:] = 0
    signal[signal < 0.1] = 0
    if lines >= lines_per_file:
        lines = 0
        file_num += 1
        f.close()

        f = open(os.path.join('..', 'Signal', filename + '{:03}'.format(file_num) + '.txt'), 'w')
write_signal(f, datetime[:interval_length - step], signal[:interval_length - step], ecg_temp[:interval_length - step])
end = time.time()

print('elapsed time: ' + str(end - start))
input()

del model
tf.keras.backend.clear_session()

f.close()
