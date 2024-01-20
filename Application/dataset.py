import numpy as np
import re
import os
import random
from scipy.signal import filtfilt, butter
from config import window_size, stack, scale_down, datapoints, \
    lines_per_file, T, fs, low_cutoff, high_cutoff, nyq, order, animal, pad_behind, pad_forward, mean_std_normalize

from matplotlib import pyplot as plt

eps = 1e-9


def random_sampling(ecg, filtered_ecg, signal, samples, ensure_labels=False):
    """Randomly creates a sample from somewhere within the data."""
    x, y = [], []
    padded_ecg = np.pad(ecg, (int(pad_behind * window_size), int(pad_forward * window_size)), constant_values=(0, 0))
    padded_filter = np.pad(filtered_ecg, (int(pad_behind * window_size), int(pad_forward * window_size)), constant_values=(0, 0))
    indices = np.random.randint(0, len(ecg) - int(pad_forward * window_size), size=samples)
    for ind in indices:
        # j is the starting index for the block that is being labeled. We include 2 after and 2 before
        # second
        y_i = signal[ind:ind + int(1 * window_size)]
        if ensure_labels:
            if max(y_i) != 1:
                continue
        x_i = process_ecg(padded_ecg[ind:ind + int(stack * window_size)],
                          padded_filter[ind:ind + int(stack * window_size)],
                          scale_down, stack, datapoints)
        # The label for the (stack - 1)th block, so there is some look ahead and some look behind
        x.append(x_i.T)
        y.append(y_i)

    x = np.asarray(x)

    y = np.asarray(y)
    return x, y


def process_ecg(ecg, filtered_ecg, scale_down, stack, datapoints):
    """Sets the baseline to be 0, and also averages every 'scale_down' datapoints to reduce the total amount of data
    per sample """
    ecg = np.sum(ecg.reshape((-1, scale_down)), axis=1) / scale_down
    if mean_std_normalize:
        ecg = (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-5)
    else:
        diff = np.max(ecg) - np.min(ecg)
        if diff != 0:
            ecg = (ecg - np.min(ecg)) / diff

    filtered_ecg[np.abs(filtered_ecg) < eps] = 0
    filtered_ecg = np.sum(filtered_ecg.reshape((-1, scale_down)), axis=1) / scale_down
    if mean_std_normalize:
        filtered_ecg = (filtered_ecg - np.mean(filtered_ecg)) / (np.std(filtered_ecg) + 1e-5)
    else:
        diff = np.max(filtered_ecg) - np.min(filtered_ecg)
        if diff != 0:
            filtered_ecg = (filtered_ecg - np.min(filtered_ecg)) / diff

    ecg = np.concatenate((ecg, filtered_ecg))

    return ecg.reshape((stack * 2, datapoints))


def filters(ecg, order, low_cutoff, high_cutoff, nyq):
    try:
        b, a = butter(N=order, Wn=low_cutoff / nyq, btype='low')
        ecg = filtfilt(b, a, np.asarray(ecg))
        b, a = butter(N=order, Wn=high_cutoff / nyq, btype='high')
        ecg = filtfilt(b, a, np.asarray(ecg))
    except:
        return ecg
    return ecg


def filters_from_config(ecg):
    return filters(ecg, order, low_cutoff, high_cutoff, nyq)


def read_file(file_name, lines):
    counter = 0
    temp = []
    for line in open(os.path.join('..', 'Training', file_name)):
        counter += 1
        temp.append(float(re.findall('([-\\d.]+)', line)[-1]))
        if counter >= lines:
            break

    return temp


def temp_plot(ecg, sig, start=0, size=2000):
    plt.plot(ecg[start:start + size])
    plt.plot(sig[start:start + size])
    plt.show()


if __name__ == '__main__':
    """Creates the train and test datasets for the model to be trained on."""
    lines = 40000000  # Maximum number of lines to read
    samples = 400000  # Number of samples to create, won't generate exactly this many however.
    counter = 0
    ensure_labels = True  # Only add samples that have an actual beat in them

    # Reads the data from the ecg and sig files (containing the ecg and markings). Then runs them through the filter,
    # before taking random samples from the data to create the datasets.
    ecg1 = read_file(f'{animal}_ecg.txt', lines)

    filtered_ecg1 = filters(ecg1, order, low_cutoff, high_cutoff, nyq)

    sig1 = read_file(f'{animal}_sig.txt', lines)

    ecg2 = read_file(f'{animal}_ecg_val.txt', lines)

    filtered_ecg2 = filters(ecg2, order, low_cutoff, high_cutoff, nyq)

    sig2 = read_file(f'{animal}_sig_val.txt', lines)

    x_train, y_train = random_sampling(ecg1, filtered_ecg1, sig1, samples, ensure_labels)

    # x_train = np.append(x_train, -x_train, axis=0)  # In our case we have inverted signals, so we just double the
    # dataset by adding more inverted signals
    # y_train = np.append(y_train, y_train, axis=0)

    x_test, y_test = random_sampling(ecg2, filtered_ecg2, sig2, samples // 3, ensure_labels)

    for i in range(10):
        plt.plot(x_train[i, :, pad_behind], label='filtered')
        sig = y_train[i]
        sig = np.sum(sig.reshape((-1, scale_down)), axis=1)
        sig /= np.max(sig)
        plt.plot(sig)
        plt.show()
    x_test = np.append(x_test, -x_test, axis=0)
    y_test = np.append(y_test, y_test, axis=0)

    # Creates the .npy files containing the data in the Training directory
    np.save(os.path.join('..', 'Training', f'{animal}_x_train'), x_train)
    np.save(os.path.join('..', 'Training', f'{animal}_y_train'), y_train)
    np.save(os.path.join('..', 'Training', f'{animal}_x_test'), x_test)
    np.save(os.path.join('..', 'Training', f'{animal}_y_test'), y_test)
