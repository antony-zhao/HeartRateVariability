import numpy as np
import re
import os
import json
import random
from scipy.signal import filtfilt, butter
from config import interval_length, step, stack, scale_down, datapoints, \
    lines_per_file, T, fs, low_cutoff, high_cutoff, nyq, order, n


def random_sampling(ecg, signal, samples, interval_length, step, scale_down, stack=1):
    datapoints = interval_length // scale_down
    x, y = [], []
    i = 0
    while i < samples:
        i += 1
        j = random.randint(step * stack, len(ecg) - interval_length)
        if 1 not in signal[j:j + interval_length]:
            i -= 1
            continue
        ls = []
        if np.random.random() < 0.95:
            for k in range(stack):
                ind = j - (stack - k - 1) * step
                temp = ecg[ind:ind + interval_length]
                temp = preprocess_ecg(np.asarray(temp), scale_down)
                temp = temp.tolist()
                ls.append(temp)
        else:
            rand = np.random.randint(1, stack - 1) if stack > 1 else 0
            for k in range(rand):
                ls.append([0] * datapoints)
            for k in range(rand, stack):
                ind = j - (stack - k - 1) * step
                temp = ecg[ind:ind + interval_length]
                temp = preprocess_ecg(np.asarray(temp), scale_down)
                temp = temp.tolist()
                ls.append(temp)
        ls = np.asarray(ls)
        ls = ls / np.max(np.abs(ls))
        x.append(ls.tolist())
        y.append(signal[j:j + interval_length])
    x = np.asarray(x)
    x = np.swapaxes(x, 1, 2)

    # Scaling, very scuffed currently, probably should look at x = (2*x/(np.nanmean(np.where(np.max(np.abs(x),
    # axis=1).reshape((-1, 1, stack)) != 0, np.max(np.abs(x), axis=1).reshape((-1, 1, stack)), np.nan),
    # axis=2).reshape((-1, 1, 1)) + np.max(np.abs(x), axis=1).reshape((-1, 1, stack))))

    y = np.asarray(y)
    return x, y


def preprocess_ecg(ecg, scale_down):
    ecg = ecg
    ecg = ecg.reshape(interval_length, )
    ecg = ecg - np.mean(ecg)
    # ecg = ecg / np.max(np.abs(ecg))
    ecg = np.sum(ecg.reshape((-1, scale_down)), axis=1) / scale_down
    return ecg


if __name__ == '__main__':
    ecg1, s1 = [], []
    ecg2, s2 = [], []

    lines = 100000000
    samples = 50000
    counter = 0

    for x in open(os.path.join('..', 'Training', 'ecg1.txt')):
        counter += 1
        ecg1.append(float(re.findall('([-0-9.]+)', x)[-1]))
        if counter >= lines:
            break

    b, a = butter(N=order, Wn=low_cutoff / nyq, btype='low', analog=False)
    ecg1 = filtfilt(b, a, np.asarray(ecg1))
    b, a = butter(N=order, Wn=high_cutoff / nyq, btype='high', analog=False)
    ecg1 = filtfilt(b, a, np.asarray(ecg1))

    counter = 0
    for x in open(os.path.join('..', 'Training', 'sig1.txt')):
        counter += 1
        s1.append(float(re.findall('([-0-9.]+)', x)[-1]))
        if counter >= lines:
            break

    counter = 0
    for x in open(os.path.join('..', 'Training', 'ecg6.txt')):
        counter += 1
        ecg2.append(float(re.findall('([-0-9.]+)', x)[-1]))
        if counter >= lines:
            break

    b, a = butter(N=order, Wn=low_cutoff / nyq, btype='low',  analog=False)
    ecg2 = filtfilt(b, a, np.asarray(ecg2))
    b, a = butter(N=order, Wn=high_cutoff / nyq, btype='high', analog=False)
    ecg2 = filtfilt(b, a, np.asarray(ecg2))

    counter = 0
    for x in open(os.path.join('..', 'Training', 'sig6.txt')):
        counter += 1
        s2.append(float(re.findall('([-0-9.]+)', x)[-1]))
        if counter >= lines:
            break

    x_train, y_train = random_sampling(ecg1, s1, samples, interval_length, step, scale_down, stack)

    x_train = np.append(x_train, -x_train, axis=0)
    y_train = np.append(y_train, y_train, axis=0)

    x_test, y_test = random_sampling(ecg2, s2, samples // 3, interval_length, step, scale_down, stack)
    x_test = np.append(x_test, -x_test, axis=0)
    y_test = np.append(y_test, y_test, axis=0)
    del s1
    del ecg1
    del ecg2
    del s2

    np.save("x_train", x_train)
    np.save("y_train", y_train)
    np.save("x_test", x_test)
    np.save("y_test", y_test)
