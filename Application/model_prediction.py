from matplotlib import pyplot as plt
import numpy as np
import os
from methods import *
import tensorflow as tf
from tensorflow.keras.models import load_model
from model import model, Res1D, distance
from dataset import preprocess_ecg
import h5py
import time
from collections import deque
import tkinter as tk
from tkinter import filedialog
import json
import joblib
from sklearn.preprocessing import StandardScaler

'''
Figure out why executable is slow

Check logic with marking beats.
'''
scaler = joblib.load("scaler.save")

config_file = open("config.json", "r")
config = json.load(config_file)
interval_length = config["interval_length"]
step = config["step"]
stack = config["stack"]
scale_down = config["scale_down"]
datapoints = config["datapoints"]
lines_per_file = config["lines_per_file"]
T = config["T"]
fs = config["fs"]
low_cutoff = config["low_cutoff"]
high_cutoff = config["high_cutoff"]
nyq = config["nyq"]
order = config["order"]
n = config["n"]

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
model_file = 'val_top_k.h5'

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
f = open(os.path.join(folder_selected, filename + '{:03}'.format(file_num) + '.txt'), 'w')


signal = np.zeros(interval_length)
model.load_weights(model_file)

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
        e = ecg[i]
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
ecg_segment = []
datetime_segment = []
datetime = []
ecg_deque = deque(maxlen=stack)
for i in range(stack-1):
    ecg_deque.append(np.zeros(datapoints, ))

read = ""
while len(read) <= 1 or read[0] == "#":
    read = file.readline()

datetime_segment, ecg_segment, EOF = read_ecg(file, interval_length * 10)
b, a = butter(N=order, Wn=low_cutoff / nyq, btype='low', analog=False)
ecg_segment = filtfilt(b, a, np.asarray(ecg_segment), axis=0)
b, a = butter(N=order, Wn=high_cutoff / nyq, btype='high', analog=False)
ecg_segment = filtfilt(b, a, np.asarray(ecg_segment), axis=0)
ecg_segment = ecg_segment.flatten()
ecg_temp = ecg_segment[:interval_length]
datetime = datetime_segment[:interval_length]
ecg_segment = ecg_segment[step:]
datetime_segment = datetime_segment[:step]

while True:
    num_lines = 0
    temp = preprocess_ecg(np.asarray(ecg_temp), scale_down)
    ecg_deque.append(temp)
    temp = np.swapaxes(np.asarray(ecg_deque)[np.newaxis, :, :], 1, 2)
    temp = temp / np.max(np.abs(temp))
    # plt_temp = temp[0, :, 0]
    # for j in range(1, stack):
    #     plt_temp = np.append(plt_temp, temp[0, :, j][datapoints//(interval_length//step):])
    # plt.plot(plt_temp)
    temp = model.predict(temp)
    sig = np.sum(temp.reshape((-1, scale_down)), axis=1) / scale_down
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
        datetime_segment = datetime_segment[:step]
        datetime = datetime[step:]
        datetime = np.append(datetime, datetime_segment[:step])
    elif not EOF:
        datetime_segment, ecg_segment, EOF = read_ecg(file, interval_length * 10)
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