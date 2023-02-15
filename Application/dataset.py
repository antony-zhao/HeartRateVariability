import numpy as np
import re
import os
import random
from scipy.signal import filtfilt, butter
from config import interval_length, step, stack, scale_down, datapoints, \
    lines_per_file, T, fs, low_cutoff, high_cutoff, nyq, order


def random_sampling(ecg, signal, samples):
    """Randomly creates a sample from somewhere within the data."""
    x, y = [], []
    i = 0
    while i < samples:
        i += 1
        # j is the starting index for the block that is being labeled. We include one after and 3 before for a half
        # second
        j = random.randint(0, len(ecg))
        x_i = ecg[j:j + int(0.5 * fs)]
        # The label for the 4th block of 100 ms, so there is some look ahead and some look behind
        y_i = signal[j + int(0.3 * fs):j + int(0.4 * fs)]
        if 1 not in y_i:
            i -= 1
            continue
        x_i = preprocess_ecg(x_i, scale_down)
        x_i.reshape((5, 100))
        x.append(x_i)

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
    return ecg / np.max(ecg)


def read_file(file_name, lines):
    counter = 0
    temp = []
    for line in open(os.path.join('..', 'Training', file_name)):
        counter += 1
        temp.append(float(re.findall('([-\\d.]+)', line)[-1]))
        if counter >= lines:
            break

    return temp


if __name__ == '__main__':
    """Creates the train and test datasets for the model to be trained on."""
    lines = 10000  # Maximum number of lines to read
    samples = 500  # Number of samples to create
    counter = 0

    # Reads the data from the ecg and sig files (containing the ecg and markings). Then runs them through the filter,
    # before taking random samples from the data to create the datasets.
    ecg1 = read_file('ecg1.txt', lines)

    ecg1 = filters(ecg1, order, low_cutoff, high_cutoff, nyq)

    sig1 = read_file('sig1.txt', lines)

    ecg2 = read_file('ecg6.txt', lines)

    ecg2 = filters(ecg2, order, low_cutoff, high_cutoff, nyq)

    sig2 = read_file('sig6.txt', lines)

    x_train, y_train = random_sampling(ecg1, sig1, samples)

    x_train = np.append(x_train, -x_train, axis=0)  # In our case we have inverted signals, so we just double the
    # dataset by adding more inverted signals
    y_train = np.append(y_train, y_train, axis=0)

    x_test, y_test = random_sampling(ecg2, sig2, samples // 3)
    x_test = np.append(x_test, -x_test, axis=0)
    y_test = np.append(y_test, y_test, axis=0)

    # Creates the .npy files containing the data in the Training directory
    np.save(os.path.join('..', 'Training', 'x_train'), x_train)
    np.save(os.path.join('..', 'Training', 'y_train'), y_train)
    np.save(os.path.join('..', 'Training', 'x_test'), x_test)
    np.save(os.path.join('..', 'Training', 'y_test'), y_test)
