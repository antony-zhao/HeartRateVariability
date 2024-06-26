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
import ctypes
from process_signal_cython import process_signal_cython

import tensorflow as tf
from tensorflow.keras.models import load_model

from dataset import process_ecg, filters
import time
from collections import deque
import tkinter as tk
from tkinter import filedialog
import tqdm
from pathlib import Path
from config import window_size, stack, scale_down, datapoints, \
    lines_per_file, max_dist_percentage, low_cutoff, high_cutoff, nyq, order, interval_length, threshold, animal, \
    pad_behind

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--filename', '-f', type=str, help='Filename to load', default=None)
    args = parser.parse_args()
    tf.keras.backend.clear_session()
    np.seterr(all='ignore')

    file_num = 1
    update_freq = 1
    loading_size = 128

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
    model1 = load_model(f'{animal}_model_conv_1', compile=False)
    K.clear_session()
    model2 = load_model(f'{animal}_model_conv_2', compile=False)
    K.clear_session()
    model3 = load_model(f'{animal}_model_conv_3', compile=False)
    K.clear_session()
    model4 = load_model(f'{animal}_model_conv_4', compile=False)
    K.clear_session()
    model5 = load_model(f'{animal}_model_conv_5', compile=False)
    ensemble = [model1, model2, model3, model4, model5]
    model1.summary()


    def ensemble_predict(batch):
        signals = []
        argmaxes = []
        for model in ensemble:
            signal = model(batch, training=False).numpy()
            signal = scipy.special.softmax(signal, axis=1)
            signal = np.asarray(signal).flatten()
            temp_signals = np.append(signal, np.zeros(interval_length - signal.size % interval_length)).reshape(-1,
                                                                                                                interval_length)
            argmax = np.argmax(temp_signals, axis=1)[:batch.shape[0]]
            argmax = argmax + np.arange(argmax.size) * np.prod(argmax.shape[1:]) * interval_length

            argmaxes.append(argmax)
            signals.append(signal)
        return signals, argmaxes

    lines = 0  # Tracks the number of lines for the current file
    dist = 0  #ctypes.c_int(0)
    first = True  #ctypes.c_bool(True)  # For handling the first signal
    average_interval = deque([interval_length] * 4, maxlen=4)  #np.array([interval_length] * 4, dtype=np.float32)
    dist_c = ctypes.c_int(0)
    first_c = ctypes.c_bool(True)  # For handling the first signal
    average_interval_c = np.array([interval_length] * 4, dtype=np.int32)

    interval = None
    curr_ind = None


    def read_ecg_polars(reader, count):
        # eof = False
        batches = reader.next_batches(1)
        if batches is None:
            return None, None, True

        # batches.select(
        #     pl.col("column_2").cast(pl.Float32, strict=False)
        # )
        # batches.fill_null(0)
        batches = batches[0].to_numpy()
        return batches[:, 0], batches[:, 1].astype(np.double), False


    def process_signal(dataframe, datetime, sig, ecg, argmax):
        """Writes the output signals (the peak detection) into a file as either a 1 or 0."""
        global dist
        global first
        global average_interval
        # The maximum amount we think the signal can differ by, our default is 0.2, so we don't believe any 'signal'
        # with a distance of 0.8-1.2 from the previous is real, and so we omit it.
        min_dist = 1 - max_dist_percentage
        max_dist = 1 + max_dist_percentage

        offsets = np.arange(argmax.size) * np.prod(argmax.shape[1:]) * interval_length
        argmax = argmax + offsets
        # curr = time.time()
        average_interval_len = len(average_interval)
        sig_len = len(sig)
        curr_argmax = argmax[0]
        curr_ind = 1
        processed_sig = [0] * sig_len  #np.zeros(sig_len, dtype=np.int32)
        argmax_len = len(argmax)

        avg = np.mean(average_interval) / average_interval_len

        for i in range(sig_len):
            if i == curr_argmax and sig[
                i] > threshold:  # Minimum value of the signal before other checks. May need to adjust this value.
                if dist < min_dist * avg:
                    if first:  # The very first signal
                        s = 1
                        first = 0
                        dist = 0
                    else:
                        s = 0
                else:
                    s = 1
                    if dist < avg * max_dist:
                        avg *= average_interval_len
                        avg -= average_interval[0]
                        avg += dist
                        avg /= average_interval_len

                        for j in range(1, average_interval_len):
                            average_interval[j - 1] = average_interval[j]
                        average_interval[average_interval_len - 1] = dist

                    dist = 0
            else:
                s = 0

            processed_sig[i] = int(s)
            dist += 1
            if i >= curr_argmax and curr_ind < argmax_len:
                curr_argmax = argmax[curr_ind]
                curr_ind += 1

        # print(time.time() - curr, "Time Elapsed")
        temp_df = pl.DataFrame({"date": datetime, "ecg": ecg.astype(np.float32), "signal": processed_sig})

        dataframe.vstack(temp_df, in_place=True)
        return sig_len


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


    def process_signal_v2(dataframe, datetime, sig, ecg, argmax):
        """Writes the output signals (the peak detection) into a file as either a 1 or 0."""
        global interval
        global curr_ind
        curr_ind = None
        interval = None

        ecg_len = len(ecg)
        processed_sig = [0] * ecg_len
        argmax = np.asarray(argmax, dtype=np.int32)

        for i in range(len(argmax[0])):
            if interval is None:
                candidates = find_candidates(argmax[:, i], sig, None)
            else:
                approximate_ind = curr_ind + interval
                candidates = find_candidates(argmax[:, i], sig, approximate_ind)

            for (candidate_ind, candidate_val, num_supporters) in candidates:
                # Figure out how to determine if there is too much disagreement in a section and just ignore it
                # Examples, multiple 0.2 or multiple 0.4s

                if (num_supporters >= len(ensemble) // 2): #or
                #(candidate_val > threshold and num_supporters > len(ensemble) // 2)):
                    # processed_sig[candidate_ind] = candidate_val
                    if curr_ind is None:
                        processed_sig[candidate_ind] = 1
                        curr_ind = candidate_ind
                    elif interval is None:
                        curr_length = candidate_ind - curr_ind
                        if curr_length < 0.6 * interval_length:
                            continue
                        elif curr_length > 1.3 * interval_length:
                            curr_ind = None
                            interval = None
                        else:
                            processed_sig[candidate_ind] = 1
                            curr_ind = candidate_ind
                            interval = curr_length
                    else:
                        curr_length = candidate_ind - curr_ind
                        if curr_length < 0.8 * interval:
                            continue
                        elif curr_length > 1.2 * interval:
                            curr_ind = None
                            interval = None
                        else:
                            processed_sig[candidate_ind] = 1
                            interval = curr_length
                            curr_ind = candidate_ind

        # print(time.time() - curr, "Time Elapsed")
        temp_df = pl.DataFrame({"date": datetime, "ecg": ecg.astype(np.float32), "signal": np.array(processed_sig, dtype=np.float32)})

        dataframe.vstack(temp_df, in_place=True)
        return ecg_len


    def find_candidates(argmaxes, signals, candidate_index, support_dist=1):
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


    ecg_segment = []  # Acts as a buffer for the ecg signals
    datetime_segment = []  # Similarly for datetimes
    ecg_temp = []  # ecg and datetimes of the portion being processed
    datetime = []

    # Skips through the header if it exists, goes until it reads a line of data (probably need to change regex depending on
    # data)
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
    # datetime_segment, ecg_segment, EOF = read_ecg_pandas(reader_pd, window_size * loading_size)
    datetime_segment, ecg_segment, EOF = read_ecg_polars(reader, window_size * loading_size)
    # datetime_segment, ecg_segment, EOF = `read_ecg(file, window_size * loading_size)
    filtered_segment = filters(ecg_segment, order, low_cutoff, high_cutoff, nyq)

    # Pads the data with 0s
    padding = (pad_behind * window_size, 0)
    ecg_segment = np.pad(ecg_segment, padding, constant_values=(0, 0))
    filtered_segment = np.pad(filtered_segment, padding, constant_values=(0, 0))

    # Initializes the temporary variables and slides the window over
    ecg_temp = ecg_segment[:window_size * stack]
    filtered_temp = filtered_segment[:window_size * stack]
    datetime = datetime_segment[:window_size]
    ecg_segment = ecg_segment[window_size:]
    filtered_segment = filtered_segment[window_size:]
    datetime_segment = datetime_segment[window_size:]

    ind1 = pad_behind * window_size
    ind2 = (pad_behind + 1) * window_size
    curr_segment = ecg_temp[ind1:ind2]
    curr_filt = filtered_temp[ind1:ind2]
    batch = []
    datetimes = []
    ecg_segments = []
    inc_time = 0

    with tqdm.tqdm(total=file_size) as pbar:  # Progress bar
        pbar.set_description('Bytes ')
        iter = 0  # Used for the progress bar to update it every so often.
        while True:
            # curr = time.time()
            num_lines = 0

            temp = process_ecg(ecg_temp, filtered_temp, scale_down, stack, datapoints)
            temp = temp[np.newaxis, :, :]
            temp = np.swapaxes(temp, 1, 2)
            batch.append(temp)
            datetimes.append(np.asarray(datetime))
            ecg_segments.append(curr_segment)
            if len(batch) >= 512:
                batch = np.array(batch)[:, 0, :, :].astype(np.float32)
                signals, argmaxes = ensemble_predict(batch)

                datetime_iter = np.concatenate(datetimes).ravel()
                segment = np.asarray(ecg_segments).flatten()

                # num_lines = process_signal_c(writer, datetime_iter, signal, segment, argmax)
                num_lines = process_signal_v2(writer, datetime_iter, signals, segment, argmaxes)
                # num_lines = process_signal_c(writer, datetime_iter, signal, segment, argmax)
                lines += num_lines
                batch = []
                datetimes = []
                ecg_segments = []

            # Steps through all the different buffers, and if the buffer is empty reads more from the file.
            if ecg_segment.size > stack * window_size * 2:
                ecg_segment = ecg_segment[window_size:]
                ecg_temp = ecg_segment[:window_size * stack]
                filtered_segment = filtered_segment[window_size:]
                filtered_temp = filtered_segment[:window_size * stack]
                datetime = datetime_segment[:window_size]
                datetime_segment = datetime_segment[window_size:]
                curr_segment = ecg_temp[ind1:ind2]
                curr_filt = filtered_temp[ind1:ind2]
            elif not EOF:
                iter += 1
                # temp_datetime, temp_ecg, EOF = read_ecg_pandas(reader_pd, window_size * loading_size)
                temp_datetime, temp_ecg, EOF = read_ecg_polars(reader, window_size * loading_size)
                # temp_datetime, temp_ecg, EOF = read_ecg(file, window_size * loading_size)
                datetime_segment = np.append(datetime_segment, temp_datetime)
                ecg_segment = np.append(ecg_segment, temp_ecg)
                if EOF:
                    break
                if iter % update_freq == 0:
                    pbar.update(line_size * len(datetime_segment) * update_freq)

                filtered_segment = filters(ecg_segment, order, low_cutoff, high_cutoff, nyq)

                ecg_temp = ecg_segment[:window_size * stack]
                filtered_temp = filtered_segment[:window_size * stack]
                datetime = datetime_segment[:window_size]
                ecg_segment = ecg_segment[window_size:]
                filtered_segment = filtered_segment[window_size:]
                datetime_segment = datetime_segment[window_size:]

                curr_segment = ecg_temp[ind1:ind2]
                curr_filt = filtered_temp[ind1:ind2]
                inc_time = 0
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
    batch = np.array(batch)[:, 0, :, :].astype(np.float32)
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
