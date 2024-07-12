import numpy as np
import re
import os
import random
from scipy.signal import filtfilt, butter, lfilter
from config import *

from matplotlib import pyplot as plt
from matplotlib import mlab

eps = 1e-9

#
# def pad_to_match(array, width):
#     return (np.append(array, np.zeros(width - len(array) % width))
#                 .reshape(-1, width))
#
# def absmaxND(a, axis=None):
#     amax = a.max(axis)
#     amin = a.min(axis)
#     return (np.where(-amin > amax, amin, amax))
#
#
# def standardize(array, width=window_size * stack):
#     array_len = len(array)
#     array_ = pad_to_match(array, width)
#     sign = np.sign(absmaxND(array_, axis=1))
#     std = np.std(array_, axis=1)
#     array /= np.repeat(sign * std, width)[:array_len]
#     return array


def random_sampling(ecg, filtered_ecg, signal, samples, ensure_labels=False):
    """Randomly creates a sample from somewhere within the data."""
    x, y = [], []
    count = 0
    # padded_ecg = np.pad(ecg, (int(pad_behind * window_size), int(pad_forward * window_size)), constant_values=(0, 0))
    # padded_filter = np.pad(filtered_ecg, (int(pad_behind * window_size), int(pad_forward * window_size)),
    #                        constant_values=(0, 0))
    # padded_sig = np.pad(sig, (int(pad_behind * window_size), int(pad_forward * window_size)), constant_values=(0, 0))
    indices = np.random.randint(0, len(ecg) - int(stack * window_size), size=samples * 4)
    for ind in indices:
        # j is the starting index for the block that is being labeled. We include 2 after and 2 before
        # second
        if len(x) >= samples:
            break
        y_i = signal[ind:ind + int(stack * window_size)] #signal[ind:ind + int(1 * window_size)]
        # if ensure_labels:
        #     if max(y_i) != 1:
        #         continue
        #     y_i.append(1)
        # else:
        #     y_i.append(0)
        y_i = np.array(y_i).reshape((stack, datapoints))
        if np.count_nonzero(y_i) < 3:
            continue
        y_i = np.concatenate((y_i, 1 - np.max(y_i, axis=1).reshape(-1, 1)), axis=1)
        # y_i = y_i.flatten()
        # y_i = np.argmax(y_i, axis=1)
        # else:
        #     if max(y_i) != 1:
        #         y_i.append(1)
        #         count += 1
        #     else:
        #         y_i.append(0)
        x_i = process_ecg(np.array(ecg[ind:ind + int(stack * window_size)]),
                          np.array(filtered_ecg[ind:ind + int(stack * window_size)]),
                          scale_down, stack, datapoints)
        # img = plt.specgram(ecg[:6000], Fs=4000)[3].get_array()
        # The label for the (stack - 1)th block, so there is some look ahead and some look behind
        x.append(x_i)
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
            ecg = (ecg - np.mean(ecg)) / diff

    filtered_ecg[np.abs(filtered_ecg) < eps] = 0
    filtered_ecg = np.sum(filtered_ecg.reshape((-1, scale_down)), axis=1) / scale_down
    if mean_std_normalize:
        filtered_ecg = (filtered_ecg - np.mean(filtered_ecg)) / (np.std(filtered_ecg) + 1e-5)
    else:
        diff = np.max(filtered_ecg) - np.min(filtered_ecg)
        if diff != 0:
            filtered_ecg = 2 * (filtered_ecg - np.min(filtered_ecg)) / diff - 1

    ecg = np.stack((ecg, filtered_ecg))#, np.gradient(ecg), np.gradient(filtered_ecg)))
    # ecg = np.stack((filtered_ecg, np.gradient(filtered_ecg)))

    return ecg.reshape((2 * stack, datapoints))


def bandpass_filter(ecg, order, lowcut, highcut, nyq):
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, ecg)
    return y


def cascaded_filters(ecg, order, low_cutoff, high_cutoff, nyq):
    try:
        b, a = butter(N=order, Wn=low_cutoff / nyq, btype='lowpass')
        ecg = filtfilt(b, a, np.asarray(ecg))
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


if __name__ == '__main__':
    """Creates the train and test datasets for the model to be trained on."""
    lines = 400000  # Maximum number of lines to read
    samples = 1500  # Number of samples to create, won't generate exactly this many however.
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
        # ecg1 = cascaded_filters(ecg1, 1, interval_length, 10, nyq)
        # mlab.specgram(ecg1[20000:26000], Fs=4000)[0][:45]

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
            filtered_ecg1 = bandpass_filter(ecg, order, low_cutoff, high_cutoff, nyq)
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
        # ecg2 = cascaded_filters(ecg2, 1, interval_length, 10, nyq)
        filtered_ecg2 = bandpass_filter(ecg2, order, low_cutoff, high_cutoff, nyq)
        if x_test is None:
            x_test, y_test = random_sampling(ecg2, filtered_ecg2, sig2, samples, ensure_labels)
        else:
            if len(ecg2) < lines // 2:
                break
            temp1, temp2 = random_sampling(ecg2, filtered_ecg2, sig2, samples, ensure_labels)
            if len(temp1) > 0:
                x_test = np.append(x_test, temp1, axis=0)
                y_test = np.append(y_test, temp2, axis=0)

    # for i in range(10):
    #     plt.plot(x_test[i, :, pad_forward], label='filtered')
    #     sig = y_test[i]
    #     sig = np.sum(sig.reshape((-1, scale_down)), axis=1)
    #     sig /= np.max(sig)
    #     plt.plot(sig)
    #     plt.show()

    # x_test = np.append(x_test, -x_test, axis=0)
    # y_test = np.append(y_test, y_test, axis=0)

    # Creates the .npy files containing the data in the Training directory
    np.save(os.path.join('..', 'Training', f'{animal}_x_train'), x_train)
    np.save(os.path.join('..', 'Training', f'{animal}_y_train'), y_train)
    np.save(os.path.join('..', 'Training', f'{animal}_x_test'), x_test)
    np.save(os.path.join('..', 'Training', f'{animal}_y_test'), y_test)

