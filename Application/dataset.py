import numpy as np
import re
import os
import random
from scipy.signal import filtfilt, butter
from config import interval_length, step, stack, scale_down, datapoints, \
    lines_per_file, T, fs, low_cutoff, high_cutoff, nyq, order, animal

from matplotlib import pyplot as plt


def random_sampling(ecg, filtered_ecg, signal, samples):
    """Randomly creates a sample from somewhere within the data."""
    x, y = [], []
    i = 0
    padded_ecg = np.pad(ecg, (int(2 * interval_length), int(2 * interval_length)), constant_values=(0, 0))
    padded_filter = np.pad(filtered_ecg, (int(2 * interval_length), int(2 * interval_length)), constant_values=(0, 0))
    while i < samples:
        i += 1
        # j is the starting index for the block that is being labeled. We include 2 after and 2 before
        # second
        j = random.randint(int(2 * interval_length), len(ecg) - int(2 * interval_length))
        x_i = np.concatenate((padded_ecg[j:j + int(4 * interval_length)],
                             padded_filter[j:j + int(4 * interval_length)]))
        # The label for the 3rd block of 100 ms, so there is some look ahead and some look behind
        y_i = signal[j:j + int(1 * interval_length)]
        x_i = preprocess_ecg(x_i, scale_down)
        x_i = x_i.reshape((4 * 2, datapoints))
        x.append(x_i.T)
        y.append(y_i)

    x = np.asarray(x)

    y = np.asarray(y)
    return x, y


def preprocess_ecg(ecg, scale_down):
    """Sets the baseline to be 0, and also averages every 'scale_down' datapoints to reduce the total amount of data
    per sample """
    ecg = np.sum(ecg.reshape((-1, scale_down)), axis=1) / scale_down
    return ecg


def filters(ecg, order, low_cutoff, high_cutoff, nyq):
    b, a = butter(N=order, Wn=low_cutoff / nyq, btype='low')
    ecg = filtfilt(b, a, np.asarray(ecg))
    b, a = butter(N=order, Wn=high_cutoff / nyq, btype='high')
    ecg = filtfilt(b, a, np.asarray(ecg))
    max_val = np.max(ecg)
    if max_val == 0:
        max_val = 1
    return ecg / max_val


def read_file(file_name, lines):
    counter = 0
    temp = []
    for line in open(os.path.join('..', 'Training', file_name)):
        counter += 1
        temp.append(float(re.findall('([-\\d.]+)', line)[-1]))
        if counter >= lines:
            break

    return temp


def temp_plot(ecg):
    plt.plot(ecg)
    plt.ylim(-0.5, 0.5)
    plt.xlim(0, 3000)
    plt.show()


if __name__ == '__main__':
    """Creates the train and test datasets for the model to be trained on."""
    lines = 5000000  # Maximum number of lines to read
    samples = 300000  # Number of samples to create
    counter = 0

    # Reads the data from the ecg and sig files (containing the ecg and markings). Then runs them through the filter,
    # before taking random samples from the data to create the datasets.
    ecg1 = read_file(f'{animal}_ecg.txt', lines)

    filtered_ecg1 = filters(ecg1, order, low_cutoff, high_cutoff, nyq)

    sig1 = read_file(f'{animal}_sig.txt', lines)

    ecg2 = read_file(f'{animal}_ecg_val.txt', lines)

    filtered_ecg2 = filters(ecg2, order, low_cutoff, high_cutoff, nyq)

    sig2 = read_file(f'{animal}_sig_val.txt', lines)

    x_train, y_train = random_sampling(ecg1, filtered_ecg1, sig1, samples)

    x_train = np.append(x_train, -x_train, axis=0)  # In our case we have inverted signals, so we just double the
    # dataset by adding more inverted signals
    y_train = np.append(y_train, y_train, axis=0)

    x_test, y_test = random_sampling(ecg2, filtered_ecg2, sig2, samples // 3)
    x_test = np.append(x_test, -x_test, axis=0)
    y_test = np.append(y_test, y_test, axis=0)

    # Creates the .npy files containing the data in the Training directory
    np.save(os.path.join('..', 'Training', f'{animal}_x_train'), x_train)
    np.save(os.path.join('..', 'Training', f'{animal}_y_train'), y_train)
    np.save(os.path.join('..', 'Training', f'{animal}_x_test'), x_test)
    np.save(os.path.join('..', 'Training', f'{animal}_y_test'), y_test)
