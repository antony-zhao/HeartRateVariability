import numpy as np
import re
import os
import random
from scipy.signal import filtfilt, butter, lfilter
from config import *

from matplotlib import pyplot as plt
from matplotlib import mlab
import glob

eps = 1e-9


def random_sampling(ecg, filtered_ecg, cleaned_ecg, signal, samples):
    """Randomly creates a sample from somewhere within the data."""
    x, y = [], []
    if len(ecg) - int(stack * window_size) <= 0:
        return [], []
    indices = np.linspace(0, len(ecg) - stack * window_size, num=samples, dtype=np.int32)
    for ind in indices:
        # j is the starting index for the block that is being labeled. We include 2 after and 2 before
        # second
        if len(x) >= samples:
            break
        y_i = signal[ind:ind + int(stack * window_size)]
        y_i = np.array(y_i).reshape((stack * datapoints // 4, 4))
        y_i = np.max(y_i, axis=-1)
        # if np.count_nonzero(y_i) < 3:
        #     continue
        # y_i = np.concatenate((y_i, 1 - np.max(y_i, axis=1).reshape(-1, 1)), axis=1)
        # y_i = np.argmax(y_i, axis=-1)
        x_i = process_sample(np.array(ecg[ind:ind + int(stack * window_size)]),
                             np.array(filtered_ecg[ind:ind + int(stack * window_size)]),
                             np.array(cleaned_ecg[ind:ind + int(stack * window_size)]))
        x.append(x_i)
        y.append(y_i)

    x = np.asarray(x)

    y = np.asarray(y)
    return x, y


def process_sample(ecg):
    """Sets the baseline to be 0, and also averages every 'scale_down' datapoints to reduce the total amount of data
    per sample """
    if mean_std_normalize:
        ecg = (ecg - np.mean(ecg, axis=0)) / (np.std(ecg, axis=0) + 1e-5)
    else:
        diff = np.max(ecg, axis=0) - np.min(ecg, axis=0)
        ecg = (ecg - np.mean(ecg, axis=0)) / (diff + 1e-5)

    return ecg


def process_segment(ecg):
    cleaned_ecg = highpass_filter(ecg, order, low_cutoff, nyq)
    bandpass_ecg = bandpass_filter(ecg, order, low_cutoff, high_cutoff, nyq)
    deriv_ecg = np.gradient(bandpass_ecg)
    squared_ecg = np.power(deriv_ecg, 2)
    moving_avg_ecg = np.convolve(squared_ecg, np.ones(40), mode='same')
    return np.array([ecg, cleaned_ecg, bandpass_ecg, deriv_ecg, squared_ecg, moving_avg_ecg]).T


def bandpass_filter(ecg, order, lowcut, highcut, nyq):
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, ecg)
    return y


def highpass_filter(ecg, order, high_cutoff, nyq):
    try:
        b, a = butter(N=order, Wn=high_cutoff / nyq, btype='highpass')
        ecg = filtfilt(b, a, np.asarray(ecg))
    except:
        return ecg

    return ecg


def filters_from_config(ecg):
    return bandpass_filter(ecg, order, low_cutoff, high_cutoff, nyq)


def read_file(file, lines):
    counter = 0
    ecg = []
    sig = []
    for line in file:
        counter += 1
        ecg.append(float(re.findall('([-\\d.]+)', line)[-2]))
        sig.append(float(re.findall('([-\\d.]+)', line)[-1]))
        if counter >= lines:
            break
        if not line:
            return ecg, sig, True

    return ecg, sig, False


def temp_plot(ecg, sig, start=0, size=2000):
    plt.plot(ecg[start:start + size])
    plt.plot(sig[start:start + size])
    plt.show()


def npy_to_tfrecords(inputs, labels, sample_weights, filename):
    pass


if __name__ == '__main__':
    """Creates the train and test datasets for the model to be trained on."""
    lines = 800000  # Maximum number of lines to read
    samples = 128  # Number of samples to create, won't generate exactly this many however.
    counter = 0
    ensure_labels = True  # Only add samples that have an actual beat in them

    # Reads the data from the ecg and sig files (containing the ecg and markings). Then runs them through the filter,
    # before taking random samples from the data to create the datasets.
    train_file = open(os.path.join('..', 'Training', f'{animal}_train.txt'))
    with open(os.path.join('..', 'Training', f'{animal}_train.txt')) as f:
        count = sum(1 for _ in f)
    val_file = open(os.path.join('..', 'Training', f'{animal}_val.txt'))

    eof = False
    current = 0
    x_train, y_train = [], []
    x_test, y_test = [], []
    while not eof:
        ecg, sig, eof = read_file(train_file, lines)
        cleaned_ecg = highpass_filter(ecg, order, low_cutoff, nyq)
        if eof:
            break
        print(count)
        count -= lines

        if len(ecg) < lines // 2:
            break

        filtered_ecg = bandpass_filter(ecg, order, low_cutoff, high_cutoff, nyq)
        x, y = random_sampling(ecg, filtered_ecg, cleaned_ecg, sig, samples)
        if len(x) == 0:
            continue
        x_train.append(x)
        y_train.append(y)
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    eof = False
    with open(os.path.join('..', 'Training', f'{animal}_val.txt')) as f:
        count = sum(1 for _ in f)
    while not eof:
        ecg, sig, eof = read_file(val_file, lines)
        cleaned_ecg = highpass_filter(ecg, order, low_cutoff, nyq)

        if eof or len(ecg) == 0:
            break
        print(count)
        count -= lines
        filtered_ecg = bandpass_filter(ecg, order, low_cutoff, high_cutoff, nyq)
        x, y = random_sampling(ecg, filtered_ecg, cleaned_ecg, sig, samples // 10)
        if len(x) == 0:
            continue
        x_test.append(x)
        y_test.append(y)
    x_test = np.concatenate(x_test)
    y_test = np.concatenate(y_test)

    # # Creates the .npy files containing the data in the Training directory
    np.save(os.path.join('..', 'Training', f'{animal}_x_train'), x_train)
    np.save(os.path.join('..', 'Training', f'{animal}_y_train'), y_train)
    np.save(os.path.join('..', 'Training', f'{animal}_x_test'), x_test)
    np.save(os.path.join('..', 'Training', f'{animal}_y_test'), y_test)
