import re
import random
import numpy as np
import copy


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


def ecg_signal_from_file(ecg,filename):
    f = open(filename, 'r')
    for x in f:
        ecg.append(float(x[0:x.index('\t')]))
        ecg.append(int(x[x.index('\t') + 1:x.index('\n')]))
    f.close()


def random_sampling(ecg, signal, samples, interval_length):
    x, y = [], []
    for i in range(samples):
        j = random.randint(0, len(ecg) - interval_length)
        temp = ecg[j:j + interval_length]
        temp = np.asarray(temp)
        temp -= np.average(temp)
        temp *= 100
        temp = temp.tolist()
        x.append(temp)
        y.append(signal[j:j + interval_length])
    x = np.asarray(x)
    y = np.asarray(y)
    size = x.size
    x = np.concatenate(x).ravel()
    return x.reshape(size // interval_length, interval_length, 1), y


def sequential_sampling(ecg, signal, interval_length, step):
    x, y = [], []
    ecg = np.asarray(ecg)
    for i in range(0, len(ecg) - len(ecg) % interval_length - step, step):
        x.append(ecg[i:i + interval_length])
        y.append(signal[i:i + interval_length])
    x = np.asarray(x)
    for i in range(x.shape[0]):
        x[i] -= np.average(x[i])
        x[i] /= np.amax(np.abs(x[i]))
    y = np.asarray(y)
    size = x.size
    x = np.concatenate(x).ravel()
    return x.reshape(size//interval_length, interval_length,1),y