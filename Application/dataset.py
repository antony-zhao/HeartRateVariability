import numpy as np
import re
import os
import json
import random
from scipy.signal import filtfilt, butter
from config import interval_length, step, stack, scale_down, datapoints, \
    lines_per_file, T, fs, low_cutoff, high_cutoff, nyq, order


def random_sampling(ecg, signal, samples, interval_length, step, scale_down, stack=1):
    """Randomly creates a sample from somewhere within the data."""
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

    y = np.asarray(y)
    return x, y


def preprocess_ecg(ecg, scale_down):
    """Sets the baseline to be 0, and also averages every 'scale_down' datapoints to reduce the total amount of data
    per sample """
    ecg = ecg
    ecg = ecg.reshape(interval_length, )
    ecg = ecg - np.mean(ecg)
    ecg = np.sum(ecg.reshape((-1, scale_down)), axis=1) / scale_down
    return ecg


if __name__ == '__main__':
    """Creates the train and test datasets for the model to be trained on."""
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

    np.save(os.path.join('..', 'Training', 'x_train'), x_train)
    np.save(os.path.join('..', 'Training', 'y_train'), y_train)
    np.save(os.path.join('..', 'Training', 'x_test'), x_test)
    np.save(os.path.join('..', 'Training', 'y_test'), y_test)
