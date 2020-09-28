import re
import random
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import lfilter, lfilter_zi, filtfilt, savgol_filter, butter
from collections import deque

T = 0.1  # Sample Period
fs = 4000.0  # sample rate, Hz
low_cutoff = 200  # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
high_cutoff = 5
nyq = 0.5 * fs  # Nyquist Frequency
order = 4  # sin wave can be approx represented as quadratic
n = int(T * fs)  # total number of samples


def ecg_from_file(ecg, filename, commented):
    f = open(filename, 'r')
    if commented:
        read = ""
        while len(read) <= 1 or read[0] == "#":
            read = f.readline()
        f.readline()

    for x in f:
        if x[0] != '#' and len(x) > 0:
            ecg.append(float(re.findall('([-0-9.]+)', x)[-1]))
    f.close()


def signal_from_file(signal, filename):
    f = open(filename,'r')
    for x in f:
        signal.append(float(x))
    f.close()
    return signal


def ecg_signal_from_file(ecg, filename):
    f = open(filename, 'r')
    for x in f:
        ecg.append(float(x[0:x.index('\t')]))
        ecg.append(int(x[x.index('\t') + 1:x.index('\n')]))
    f.close()


def random_sampling(ecg, signal, samples, interval_length, step, stack=1):
    x, y = [], []
    avg_max = 0
    for i in range(samples):
        j = random.randint(step*stack, len(ecg) - interval_length)
        if 1 not in signal[j:j + interval_length]:
            i -= 1
            continue
        ls = []
        if np.random.random() < 0.8:
            for k in range(stack):
                ind = j - (stack - k - 1) * step
                temp = ecg[ind:ind + interval_length]
                temp = np.asarray(temp).reshape(interval_length, )
                b, a = butter(N=order, Wn=low_cutoff / nyq, btype='low', analog=False)
                temp = filtfilt(b, a, np.asarray(temp))
                b, a = butter(N=order, Wn=high_cutoff / nyq, btype='high', analog=False)
                temp = filtfilt(b, a, np.asarray(temp))
                temp = temp.tolist()
                ls.append(temp)
        else:
            rand = np.random.randint(1, stack-1) if stack > 1 else 0
            for k in range(rand):
                ls.append([0]*interval_length)
            for k in range(rand, stack):
                ind = j - (stack - k - 1) * step
                temp = ecg[ind:ind + interval_length]
                temp = np.asarray(temp).reshape(interval_length, )
                b, a = butter(N=order, Wn=low_cutoff / nyq, btype='low', analog=False)
                temp = filtfilt(b, a, np.asarray(temp))
                b, a = butter(N=order, Wn=high_cutoff / nyq, btype='high', analog=False)
                temp = filtfilt(b, a, np.asarray(temp))
                temp = temp.tolist()
                ls.append(temp)

        x.append(ls)
        y.append(signal[j:j + interval_length])
    x = np.asarray(x)
    x = np.swapaxes(x, 1, 2)

    x = (2*x/(np.nanmean(np.where(np.max(np.abs(x), axis=1).reshape((-1, 1, stack)) != 0,
                                  np.max(np.abs(x), axis=1).reshape((-1, 1, stack)), np.nan), axis=2).reshape((-1, 1, 1))
              + np.max(np.abs(x), axis=1).reshape((-1, 1, stack))))

    y = np.asarray(y)
    return x, y


def sequential_sampling(ecg, signal, interval_length, step):
    x, y = [], []
    prev = np.zeros(shape=interval_length)
    for i in range(0, len(ecg) - interval_length, step):
        temp = np.asarray(ecg[i:i + interval_length])
        temp -= np.mean(temp)
        temp /= np.max(np.abs(temp))
        x.append(ecg[i:i + interval_length])
        y.append(signal[i:i + interval_length])
    x = np.asarray(x)
    y = np.asarray(y)

    return x.reshape(-1, interval_length, 1), y
