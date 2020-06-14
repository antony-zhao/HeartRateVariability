from matplotlib import pyplot as plt
import numpy as np
import os
from Methods import *
import tensorflow as tf
from Model import Model, train_model, load_model, interval_length, step
import h5py
import time
from collections import deque
from scipy.signal import filtfilt

n = 20  # https://stackoverflow.com/questions/37598986/reducing-noise-on-data
b = [1 / n] * n
a = 1

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
model_file = 'Model.h5'

lines_per_file = 10000000
file_num = 1

ecg = []
filename = "T21_transition example2_600s"
file = open(os.path.join('..', 'ECG_Data', filename + '.ascii'), 'r')
f = open(os.path.join('..', 'Signal', filename + str(file_num) + '.txt'), 'w')
commented = True

signal = np.zeros(interval_length)
load_model(model_file)

start = time.time()
lines = 0


def read_ecg(ecg_file, count):
    e = False
    sig = np.zeros(count)
    for i in range(count):
        x = ecg_file.readline()
        if len(x) == 0:
            e = True
            break
        sig[i] = float(re.findall('([-0-9.]+)', x)[-1])
    return sig.reshape(count, 1), e


def write_signal(sig_file, sig, ecg):
    lines = 0
    for e, s in zip(ecg, sig):
        lines += 1
        if s > 1 / (interval_length / step):
            s = 1
        else:
            s = 0
        e = str(float(e))
        e += '0' * (7 - len(e))
        sig_file.write('{},{}\n'.format(e, int(s)))
    return lines


if commented:
    read = ""
    while len(read) <= 1 or read[0] == "#":
        read = file.readline()

ecg_temp = []
ecg_temp, EOF = read_ecg(file, interval_length)

while not EOF:
    num_lines = 0
    temp = np.asarray(ecg_temp).reshape(interval_length, )
    # temp -= np.average(temp)
    # temp *= 100
    temp = filtfilt(b, a, temp)
    temp -= np.mean(temp)
    temp /= np.max(np.abs(temp))
    temp = Model.predict(temp.reshape(1, interval_length, 1))
    temp = temp.reshape(interval_length, )
    temp[temp < 0.4] = 0
    temp[temp >= 0.4] = 1
    signal += temp / (interval_length / step)

    num_lines = write_signal(f, signal[:step], ecg_temp[:interval_length - step])
    lines += num_lines

    ecg_temp[:interval_length - step] = ecg_temp[step:]
    ecg_temp[interval_length - step:], EOF = read_ecg(file, step)
    signal[0:interval_length - step] = signal[step:]
    signal[interval_length - step:] = 0
    signal[signal < 0.1] = 0
    if lines >= lines_per_file:
        lines = 0
        file_num += 1
        f.close()
        f = open(os.path.join('..', 'Signal', filename + str(file_num) + '.txt'), 'w')
write_signal(f, signal[:interval_length - step], ecg_temp[:interval_length - step])
end = time.time()

print('elapsed time: ' + str(end - start))

del Model
tf.keras.backend.clear_session()

f.close()
