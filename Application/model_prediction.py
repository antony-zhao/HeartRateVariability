import gc
from argparse import ArgumentParser

import scipy.special
from matplotlib import pyplot as plt
import numpy as np
import os
import re
from keras import backend as K
import polars as pl
import pandas as pd
from scipy.fft import fft, fftfreq
import ctypes
from process_signal_cython import process_signal_cython

import tensorflow as tf
from tensorflow.keras.models import load_model

from dataset import process_sample, highpass_filter, bandpass_filter, process_segment
import time
from collections import deque
import tkinter as tk
from tkinter import filedialog
import tqdm
from pathlib import Path
from config import *
import scipy.integrate as integrate
from scipy.fft import fft, ifft

signal_level_i, noise_level_i, threshold_i, signal_level_f, noise_level_f, threshold_f = (None, None, None,
                                                                                          None, None, None)


def ensemble_predict(batch):
    signals = []
    argmaxes = []
    for model in ensemble:
        signal = model.predict(batch, verbose='0')
        signal = scipy.special.expit(signal)
        # signal = signal.reshape(-1, window_size + 1)
        # signal = mask_signal(signal)
        signal = np.asarray(signal).flatten()
        temp_signals = np.append(signal, np.zeros(interval_length - signal.size % interval_length)).reshape(-1,
                                                                                                            interval_length)
        argmax = np.argmax(temp_signals, axis=1)
        argmax = argmax + np.arange(argmax.size) * np.prod(argmax.shape[1:]) * interval_length

        argmaxes.append(argmax)
        signals.append(signal)
    return signals, argmaxes


def fft_filter(ecg, axis=1):
    fft_signal = fft(ecg, axis=axis)

    # Identify noise frequencies and mask them out
    # For example, setting frequencies with too high amplitude to zero
    n_samples = ecg.shape[0]
    amplitudes = 2 / n_samples * np.abs(fft_signal)
    frequencies = np.fft.fftfreq(n_samples) * n_samples

    ecg = ifft(fft_signal)
    return ecg, amplitudes, frequencies


def pad_to_match(array, width):
    return (np.append(array, np.zeros(width - len(array) % width))
            .reshape(-1, width))


def read_ecg_polars(reader, count):
    batches = reader.next_batches(1)
    if batches is None:
        return None, None, True

    batches = batches[0].to_numpy()
    return batches[:, 0], batches[:, 1].astype(np.double), False


def process_signal_c(dataframe, datetime, sig, ecg, argmax):
    """Writes the output signals (the peak detection) into a file as either a 1 or 0."""
    global dist
    global first
    global average_interval_c
    # The maximum amount we think the signal can differ by, our default is 0.2, so we don't believe any 'signal'
    # with a distance of 0.8-1.2 from the previous is real, and so we omit it.
    min_dist = 1 - max_dist_percentage
    max_dist = 1 + max_dist_percentage

    offsets = np.arange(argmax.size) * np.prod(argmax.shape[1:]) * interval_length
    argmax = (argmax + offsets).astype(np.int32)

    processed_sig, average_interval_c, first, dist = process_signal_cython(sig, len(sig), argmax, len(argmax),
                                                                           average_interval_c,
                                                                           len(average_interval_c),
                                                                           threshold, min_dist, max_dist, first,
                                                                           dist)

    temp_df = pl.DataFrame({"date": datetime, "ecg": ecg.astype(np.float32), "signal": processed_sig})

    dataframe.vstack(temp_df, in_place=True)
    return len(datetime)


def process_signal_v2(dataframe, datetime, sig, ecg, argmax, radius=200):
    """Writes the output signals (the peak detection) into a file as either a 1 or 0."""
    global signal_level_i, noise_level_i, threshold_i, signal_level_f, noise_level_f, threshold_f, interval_final
    curr_ind = None
    curr_interval = None
    prev_interval = None
    ecg_len = len(ecg)
    sig = sig[0]
    radius = 100
    index_range = np.arange(-radius, radius)
    # sig[1:][sig[:-1] == sig[1:]] = 0
    signal_inds = np.flatnonzero((sig > 0.2).astype(np.int32))
    processed_sig_final = [0] * ecg_len
    area_around_indices = np.clip(signal_inds[:, None] + index_range, 0, ecg_len - 1)

    # bandpass_ecg = bandpass_filter(ecg, order, low_cutoff, high_cutoff, nyq)
    # prominences = (scipy.signal.peak_prominences(bandpass_ecg * 10, signal_inds, 10)[0] /
    #                np.max(np.abs(bandpass_ecg[area_around_indices]), axis=-1))
    # deriv_ecg = np.gradient(bandpass_ecg)
    # squared_ecg = np.power(deriv_ecg, 2)
    # moving_avg_ecg = np.convolve(squared_ecg, np.ones(40), mode='same')

    # processed_sig_final[signal_inds] = prominences / np.max(prominences)

    for i, ind in enumerate(signal_inds):
        if curr_ind is None:
            curr_ind = ind
            processed_sig_final[ind] = 1
            continue
        interval = ind - curr_ind
        if curr_interval is None:
            upper_thresh = 1.4
            lower_thresh = 0.6
        else:
            upper_thresh = 1.2
            lower_thresh = 0.8
        if upper_thresh * interval_length > interval:
            curr_ind = None
            curr_interval = None
            # prev_interval = None
        if interval > lower_thresh * interval_length:
            curr_interval = interval
            curr_ind = ind
            processed_sig_final[ind] = 1


    temp_df = pl.DataFrame({"date": datetime, "ecg": ecg.astype(np.float32),
                            "signal": np.array(processed_sig_final, dtype=np.int32),
                            "ensemble": np.array(sig, dtype=np.float32)})

    dataframe.vstack(temp_df, in_place=True)
    return ecg_len


def find_candidates(argmaxes, signals, candidate_index, support_dist=5):
    '''
    :param argmaxes:
    :param candidate_index:
    :return: A list of tuples. Each tuple consists of the candidate index as well as a list of which argmaxes are
    supporting it .
    '''
    candidates = []
    if candidate_index is None:
        candidate_index = np.median(argmaxes)
    argsort_ind = np.argsort(np.abs(argmaxes - candidate_index))
    sorted_by_dist = argmaxes[argsort_ind]
    processed_ind = set()
    for i, ind in enumerate(sorted_by_dist):
        if i in processed_ind:
            continue
        processed_ind.add(i)
        num_supporters = 1
        supporting = signals[argsort_ind[i]][ind]
        for j, support_ind in enumerate(sorted_by_dist):
            if i == j or j in processed_ind:
                continue
            elif abs(ind - support_ind) < support_dist:
                supporting += signals[argsort_ind[j]][support_ind]
                processed_ind.add(j)
                num_supporters += 1
        candidates.append((ind, supporting, num_supporters))
    return candidates


def cascaded_mean(signals, ecg, datetime, step=window_size // 2):
    signals_len = signals.shape[0]
    output_array = np.zeros(stack * window_size + step * (signals_len - 1))
    mean = np.zeros(stack * window_size + step * (signals_len - 1))

    for i in range(signals_len):
        output_array[i * step: i * step + stack * window_size] += signals[i]
        mean[i * step: i * step + stack * window_size] += np.ones(stack * window_size)

    return output_array / mean


def mask_signal(signals):
    return np.where(np.repeat(np.argmax(signals, axis=-1), window_size).reshape(-1, window_size) == window_size,
                    0, signals[:, :-1])


def step_through_data(segments, step=window_size):
    batch = []
    for i in range(len(segments)):
        batch.append(segments[i][:window_size * stack])
        segments[i] = segments[i][window_size * stack:]

    return batch


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--filename', '-f', type=str, help='Filename to load', default=None)
    args = parser.parse_args()
    tf.keras.backend.clear_session()
    np.seterr(all='ignore')

    file_num = 1
    update_freq = 1
    loading_size = 512
    batch_size = 256
    num_offsets = 6
    offset_size = 256

    # Opening file and choosing directory to save code in
    if args.filename is None:
        root = tk.Tk()
        currdir = os.getcwd()
        par = Path(currdir).parent
        root.filename = filedialog.askopenfilename(initialdir=str(par) + r'\ECG_Data', title='Select file',
                                                   filetypes=(('ascii files', '*.ascii'), ('txt files', '*.txt'),
                                                              ('all files', '*.*')))
        filename = os.path.basename(root.filename)
        if len(filename) == 0:
            exit(0)
        root.destroy()
    else:
        filename = args.filename
    file = open(os.path.join('..', 'ECG_Data', filename), 'r')
    temp_file = os.path.join('..', 'ECG_Data', filename)

    file_size = os.stat(os.path.join('..', 'ECG_Data', filename)).st_size

    folder_selected = 'ECG_Data'

    filename = filename[:filename.index('.')]  # Removes the type of file (i.e. ascii suffix)
    # without the full path
    f = os.path.join('..', folder_selected, filename + '{:03}'.format(file_num) + '.txt')  # Opens the first of the
    # files of which the data will be saved into. Whenever the number of lines reaches a certain point, file_num gets
    # incremented and a new file with the new number is created.
    writer = pl.DataFrame()

    # Start timer (displays time elapsed in the end)
    start = time.time()
    # model = keras.models.load_model(f'{animal}_model', compile=False)
    K.clear_session()
    model1 = load_model(f'{animal}_model_val_recall', compile=False)
    K.clear_session()
    model2 = load_model(f'{animal}_model_val_top_k', compile=False)
    K.clear_session()
    ensemble = [model1]
    model1.summary()

    lines = 0  # Tracks the number of lines for the current file
    dist = 0
    first = True
    average_interval = deque([interval_length] * 4, maxlen=4)

    interval = None
    curr_ind = None
    curr_ind_final = None
    interval_final = None

    # Skips through the header if it exists, goes until it reads a line of data (probably need to change regex
    # depending on data)
    read = ''
    count = 0
    while re.match('\\s*[\\d\\s/:.APM]+,[-\\s\\dx.,]*[-\\s\\dx.]\n', read) is None:
        read = file.readline()
        count += 1

    reader = pl.read_csv_batched(temp_file, skip_rows=count, has_header=False, batch_size=window_size * loading_size,
                                 infer_schema_length=0, null_values='     x', raise_if_empty=True, rechunk=True)
    reader_pd = pd.read_csv(temp_file, skiprows=count, header=None, chunksize=window_size * loading_size,
                            na_values='     x')

    file_loc = file.tell()
    temp_line = file.readline()
    file.seek(file_loc)
    line_size = len(temp_line.encode('utf-8'))  # Gets the average size of the line for the progress bar
    # to track approximately where we are within the file.

    # Reads a long segment of data for the filters, and slowly 'scrolls' through, removing the data that
    datetime_segment, ecg_segment, EOF = read_ecg_polars(reader, window_size * loading_size)
    processed_segment = process_segment(ecg_segment)
    segments = [ecg_segment, processed_segment, datetime_segment]  # Last element will always be
    # datetime and first element the ecg data, other elements are data to pass into the model.

    # offsets = [np.pad(ecg_segment, (256 * i, 0), constant_values=(0, 0)) for i in range(num_offsets)]
    # filtered_offsets = [bandpass_filter(offsets[i], order, low_cutoff, high_cutoff, nyq) for i in range(num_offsets)]

    # Initializes the temporary variables and slides the window over
    batch = step_through_data(segments)
    batches = []
    datetimes = []
    ecg_segments = []

    with tqdm.tqdm(total=file_size) as pbar:  # Progress bar
        pbar.set_description('Bytes ')
        iter = 0  # Used for the progress bar to update it every so often.
        while True:
            # curr = time.time()
            num_lines = 0

            temp = process_sample(batch[1])
            batches.append(temp)
            datetimes.append(batch[-1])
            ecg_segments.append(batch[0])
            if len(batches) >= batch_size:
                batch = np.array(batches).astype(np.float32)[:, :]
                signals, argmaxes = ensemble_predict(batch)

                datetime_iter = np.concatenate(datetimes).ravel()
                segment = np.asarray(ecg_segments).flatten()

                # num_lines = process_signal_c(writer, datetime_iter, signal, segment, argmax)
                num_lines = process_signal_v2(writer, datetime_iter, signals, segment, argmaxes, None)
                # num_lines = process_signal_c(writer, datetime_iter, signal, segment, argmax)
                lines += num_lines
                batches = []
                datetimes = []
                ecg_segments = []

            # Steps through all the different buffers, and if the buffer is empty reads more from the file.
            if len(segments[0]) > stack * window_size * 2:
                batch = step_through_data(segments)
            elif not EOF:
                iter += 1
                # temp_datetime, temp_ecg, EOF = read_ecg_pandas(reader_pd, window_size * loading_size)
                temp_datetime, temp_ecg, EOF = read_ecg_polars(reader, window_size * loading_size)
                # temp_datetime, temp_ecg, EOF = read_ecg(file, window_size * loading_size)
                datetime_segment = np.append(segments[-1], temp_datetime)
                ecg_segment = np.append(segments[0], temp_ecg)
                if EOF:
                    break
                if iter % update_freq == 0:
                    pbar.update(line_size * len(datetime_segment) * update_freq)

                processed_segment = process_segment(ecg_segment)
                segments = [ecg_segment, processed_segment, datetime_segment]

                batch = step_through_data(segments)
            else:
                break

            if lines > lines_per_file:  # Creates a new file in case the current one has enough lines.
                writer.write_csv(f, include_header=False)
                writer = pl.DataFrame()
                gc.collect()
                lines = 0
                file_num += 1

                f = os.path.join('..', folder_selected, filename + '{:03}'.format(file_num) + '.txt')

    # # Writes the final few datapoints into the file, and closes it
    batch = np.array(batches).astype(np.float32)[:, :]
    signals, argmaxes = ensemble_predict(batch)

    datetime_iter = np.concatenate(datetimes).ravel()
    segment = np.asarray(ecg_segments).flatten()

    # num_lines = process_signal_c(writer, datetime_iter, signal, segment, argmax)
    num_lines = process_signal_v2(writer, datetime_iter, signals, segment, argmaxes)
    # num_lines = process_signal_c(writer, datetime_iter, signal, segment, argmax)
    writer.write_csv(f, include_header=False)
    # lines += num_lines
    end = time.time()

    print('elapsed time: ' + str(end - start) + ' seconds')
    # input('Press enter to continue (may need 2 times)')

    tf.keras.backend.clear_session()
