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
    count = 0
    padded_ecg = np.pad(ecg, (int(pad_behind * window_size), int(pad_forward * window_size)), constant_values=(0, 0))
    padded_filter = np.pad(filtered_ecg, (int(pad_behind * window_size), int(pad_forward * window_size)), constant_values=(0, 0))
    indices = np.random.randint(0, len(ecg) - int(pad_forward * window_size), size=samples * 2)
    for ind in indices:
        # j is the starting index for the block that is being labeled. We include 2 after and 2 before
        # second
        if len(x) >= samples:
            break
        y_i = signal[ind:ind + int(1 * window_size)]
        if ensure_labels or count > 20:
            if max(y_i) != 1:
                continue
            # else:
            #     y_i.append(0)
        # else:
        #     if max(y_i) != 1:
        #         y_i.append(1)
        #         count += 1
        #     else:
        #         y_i.append(0)
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
            ecg = 2 * (ecg - np.min(ecg)) / diff - 1

    filtered_ecg[np.abs(filtered_ecg) < eps] = 0
    filtered_ecg = np.sum(filtered_ecg.reshape((-1, scale_down)), axis=1) / scale_down
    if mean_std_normalize:
        filtered_ecg = (filtered_ecg - np.mean(filtered_ecg)) / (np.std(filtered_ecg) + 1e-5)
    else:
        diff = np.max(filtered_ecg) - np.min(filtered_ecg)
        if diff != 0:
            filtered_ecg = 2 * (filtered_ecg - np.min(filtered_ecg)) / diff - 1

    ecg = np.concatenate((ecg, filtered_ecg))

    return ecg.reshape((2 * stack, datapoints))


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


if __name__ == '__main__':
    """Creates the train and test datasets for the model to be trained on."""
    lines = 400000  # Maximum number of lines to read
    samples = 2500  # Number of samples to create, won't generate exactly this many however.
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
                gaussian_noise = np.random.normal(0, std / 10, lines)
                ecg = ecg1 + my_wave + gaussian_noise
                # if np.random.random() > 0.5:
                #     ecg *= -1
            else:
                gaussian_noise = np.random.normal(0, np.std(ecg1) / 10, lines)
                ecg = ecg1 + gaussian_noise
            filtered_ecg1 = filters(ecg, order, low_cutoff, high_cutoff, nyq)
            if x_train is None:
                x_train, y_train = random_sampling(ecg, filtered_ecg1, sig1, samples, ensure_labels)
            else:
                temp1, temp2 = random_sampling(ecg, filtered_ecg1, sig1, samples, ensure_labels)
                if temp1.size == 0:
                    continue
                x_train = np.append(x_train, temp1, axis=0)
                y_train = np.append(y_train, temp2, axis=0)
    eof = False
    while not eof:
        ecg2, sig2, eof = read_file(val_file, lines)
        filtered_ecg2 = filters(ecg2, order, low_cutoff, high_cutoff, nyq)
        if x_test is None:
            x_test, y_test = random_sampling(ecg2, filtered_ecg2, sig2, samples * 3, ensure_labels)
        else:
            if len(ecg2) < lines // 2:
                break
            temp1, temp2 = random_sampling(ecg2, filtered_ecg2, sig2, samples * 3, ensure_labels)
            if len(temp1) > 0:
                x_test = np.append(x_test, temp1, axis=0)
                y_test = np.append(y_test, temp2, axis=0)

    # x_train = np.append(x_train, -x_train, axis=0)  # In our case we have inverted signals, so we just double the
    # dataset by adding more inverted signals
    # y_train = np.append(y_train, y_train, axis=0)


    for i in range(10):
        plt.plot(x_test[i, :, pad_forward], label='filtered')
        sig = y_test[i]
        sig = np.sum(sig.reshape((-1, scale_down)), axis=1)
        sig /= np.max(sig)
        plt.plot(sig)
        plt.show()

    # x_test = np.append(x_test, -x_test, axis=0)
    # y_test = np.append(y_test, y_test, axis=0)

    # Creates the .npy files containing the data in the Training directory
    np.save(os.path.join('..', 'Training', f'{animal}_x_train'), x_train)
    np.save(os.path.join('..', 'Training', f'{animal}_y_train'), y_train)
    np.save(os.path.join('..', 'Training', f'{animal}_x_test'), x_test)
    np.save(os.path.join('..', 'Training', f'{animal}_y_test'), y_test)
