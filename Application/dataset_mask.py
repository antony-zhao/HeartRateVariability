import numpy as np
import re
import os
import random
from scipy.signal import filtfilt, butter, lfilter
from config import window_size, stack, scale_down, datapoints, \
    lines_per_file, T, fs, low_cutoff, high_cutoff, nyq, order, animal, pad_behind, pad_forward, mean_std_normalize

from matplotlib import pyplot as plt
from dataset import *


def random_sampling_masking(ecg, signal, samples, window=100):
    """Randomly creates a sample from somewhere within the data."""
    x, y = [], []
    padded_ecg = np.pad(ecg, (int(pad_behind * window_size), int(pad_forward * window_size)), constant_values=(0, 0))
    # padded_filter = np.pad(filtered_ecg, (int(pad_behind * window_size), int(pad_forward * window_size)),
    #                        constant_values=(0, 0))
    needed_marks = samples // 3
    indices = np.random.randint(0, len(ecg) - int(pad_forward * window_size), size=samples * 10)
    mark_count = 0
    for ind in indices:
        if len(x) >= samples:
            break
        has_mark = 1 if np.count_nonzero(signal[ind:ind + int(window_size)]) > 0 else 0
        if not has_mark and mark_count < needed_marks:
            continue
        if has_mark:
            mark_count += 1
        y_i = has_mark #np.append(signal[ind:ind + int(window_size)], has_mark)
        x_i = padded_ecg[ind:ind + int(stack * window_size)].reshape(-1, window)

        # The label for the (stack - 1)th block, so there is some look ahead and some look behind
        x.append(x_i)
        y.append(y_i)

    x = np.asarray(x)

    y = np.asarray(y)
    return x, y


if __name__ == '__main__':
    """Creates the train and test datasets for the model to be trained on."""
    lines = 400000  # Maximum number of lines to read
    samples = 3000  # Number of samples to create, won't generate exactly this many however.
    counter = 0
    ensure_labels = True  # Only add samples that have an actual beat in them

    # Reads the data from the ecg and sig files (containing the ecg and markings). Then runs them through the filter,
    # before taking random samples from the data to create the datasets.
    train_file = open(os.path.join('..', 'Training', f'{animal}_train.txt'))
    val_file = open(os.path.join('..', 'Training', f'{animal}_val.txt'))
    eof = False
    x_train, y_train = None, None
    x_test, y_test = None, None
    while not eof:
        ecg1, sig1, eof = read_file(train_file, lines)
        if len(ecg1) < lines // 2:
            break
        std = np.std(ecg1)
        for i in range(0, 20, 2):
            if i > 0:
                length = np.pi * 2 * i
                my_wave = (np.cos(np.linspace(0, length, lines)) * 0.05)
                ecg = ecg1 + my_wave
            else:
                ecg = ecg1
            if x_train is None:
                x_train, y_train = random_sampling_masking(ecg, sig1, samples)
            else:
                temp1, temp2 = random_sampling_masking(ecg, sig1, samples)
                if temp1.size == 0:
                    continue
                x_train = np.append(x_train, temp1, axis=0)
                y_train = np.append(y_train, temp2, axis=0)
    eof = False
    while not eof:
        ecg2, sig2, eof = read_file(val_file, lines)
        if x_test is None:
            x_test, y_test = random_sampling_masking(ecg2, sig2, samples)
        else:
            if len(ecg2) < lines // 2:
                break
            temp1, temp2 = random_sampling_masking(ecg2, sig2, samples)
            if len(temp1) > 0:
                x_test = np.append(x_test, temp1, axis=0)
                y_test = np.append(y_test, temp2, axis=0)

    # Creates the .npy files containing the data in the Training directory
    np.save(os.path.join('..', 'Training', f'{animal}_x_train_mask'), x_train)
    np.save(os.path.join('..', 'Training', f'{animal}_y_train_mask'), y_train)
    np.save(os.path.join('..', 'Training', f'{animal}_x_test_mask'), x_test)
    np.save(os.path.join('..', 'Training', f'{animal}_y_test_mask'), y_test)
