from argparse import ArgumentParser

import scipy.special
from matplotlib import pyplot as plt
import numpy as np
import os
import re
from scipy.special import expit
import os
import polars as pl

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

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
    loading_size = 512
    # model = keras.models.load_model(f'{animal}_model', compile=False)
    model = load_model(f'{animal}_model_1', compile=False)
    model2 = load_model(f'{animal}_model_2', compile=False)
    model3 = load_model(f'{animal}_model_3', compile=False)
    model.summary()

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
    writer = pl.DataFrame([[], []])

    # Start timer (displays time elapsed in the end)
    start = time.time()

    lines = 0  # Tracks the number of lines for the current file
    dist = 0
    first = True  # For handling the first signal
    average_interval = deque(maxlen=10)  # Last 10 interval lengths, used to find the running average
    average_interval.append(interval_length)


    def read_ecg(ecg_file, count):
        """Reads 'count' lines from the ecg_file and returns them"""
        eof = False  # End of file
        ecg = np.zeros(count)
        datetime = []
        for i in range(count):
            line = ecg_file.readline()
            if len(line) == 0:  # Signifies an end of the file 
                eof = True
                break
            temp = re.findall('([-0-9.x]+)', line)[-1]  # Sometimes x is in our data which is just an empty value,
            # otherwise this just reads the signal value
            ecg[i] = 0 if temp == 'x' else float(temp)
            datetime.append(line[:line.index(',')])  # The time value, used for post_processing later so it is preserved
            # and transferred to the output file.

        return datetime, ecg, eof

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
        return batches[:, 0], batches[:, 1].astype(np.float32), False

    def process_signal(dataframe, datetime, sig, ecg, argmax):
        """Writes the output signals (the peak detection) into a file as either a 1 or 0."""
        global dist
        global first
        # plt.plot(sig)

        # if lines >= 0:
        #     plt.plot(ecg)  # Just left in, uncomment if you want a visualization of the data for testing purposes.
        #     plt.plot(sig)
        #     plt.plot([threshold for _ in range(window_size)])
        #     plt.show()
        # The maximum amount we think the signal can differ by, our default is 0.2, so we don't believe any 'signal'
        # with a distance of 0.8-1.2 from the previous is real, and so we omit it.
        min_dist = 1 - max_dist_percentage
        max_dist = 1 + max_dist_percentage

        # peaks, _ = scipy.signal.find_peaks(sig, height=threshold, distance=300, prominence=0.15)
        # current_peak = peaks[0]
        # exclusion = current_peak
        # for i, peak in enumerate(peaks[1:]):
        #     if min_dist * (peak - current_peak) < peaks[i + 1] - peak:
        #         exclusion.append(peak)
        offsets = np.arange(argmax.size) * np.prod(argmax.shape[1:]) * interval_length
        argmax = argmax + offsets
        processed_sig = []
        for i in range(len(datetime)):
            d = datetime[i]
            e = ecg[i]
            # s_ = sig[i]
            if i in argmax and sig[i] > threshold:  # Minimum value of the signal before other checks. May need to adjust this value.
                if dist < min_dist * np.mean(average_interval):
                    if first:  # The very first signal
                        s = 1
                        first = False
                    else:
                        s = 0
                else:
                    s = 1
                    if dist < np.mean(average_interval) * max_dist:
                        average_interval.append(dist)
                    dist = 0
            else:
                s = 0
            # sig_file.write('{},{:>8},{}\n'.format(d, '{:2.5f}'.format(e), int(s)))
            processed_sig.append(int(s))
            dist += 1
        temp_df = pl.DataFrame({"date": datetime, "ecg": ecg, "signal": processed_sig})
        dataframe.vstack(temp_df, in_place=True)
        # ind = len(ecg) // 4
        # plt.figure(figsize=(40, 10))
        # plt.plot(ecg[:ind])
        # plt.plot(sig[:ind])
        # plt.show()
        return len(datetime), dataframe


    ecg_segment = []  # Acts as a buffer for the ecg signals
    datetime_segment = []  # Similarly for datetimes
    ecg_temp = []  # ecg and datetimes of the portion being processed
    datetime = []
    signal = np.zeros(window_size)

    # Skips through the header if it exists, goes until it reads a line of data (probably need to change regex depending on
    # data)
    read = ''
    count = 0
    while re.match('\\s*[\\d\\s/:.APM]+,[-\\s\\dx.,]*[-\\s\\dx.]\n', read) is None:
        read = file.readline()
        count += 1

    reader = pl.read_csv_batched(temp_file, skip_rows=count, has_header=False, batch_size=window_size * loading_size,
                                 infer_schema_length=0, null_values='     x', raise_if_empty=True)

    file_loc = file.tell()
    temp_line = file.readline()
    file.seek(file_loc)
    line_size = len(temp_line.encode('utf-8'))  # Gets the average size of the line for the progress bar
    # to track approximately where we are within the file.

    # Reads a long segment of data for the filters, and slowly 'scrolls' through, removing the data that
    datetime_segment, ecg_segment, EOF = read_ecg_polars(reader, window_size * loading_size) # read_ecg(file, window_size * loading_size)
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
            # inc_time += time.time() - curr
            if len(batch) >= 512:
                # curr = time.time()
                batch = np.array(batch)[:, 0, :, :].astype(np.float32)
                signals = model(batch, training=False).numpy()
                signals2 = model2(batch, training=False).numpy()
                signals3 = model3(batch, training=False).numpy()
                signals = scipy.special.expit(signals)
                signals2 = scipy.special.expit(signals2)
                signals3 = scipy.special.expit(signals3)
                # print(time.time() - curr, " Model time")
                # curr = time.time()
                # signals = np.where((signals[:, -1] > 0.8)[:, None], np.zeros_like(ecg_segments), signals[:, :-1])

                datetime_iter = np.concatenate(datetimes).ravel()
                signals = (signals + signals2 + signals3) / 3
                # signals = np.where(signals2 > threshold, signals, 0)
                # signals = np.where(signals3 > threshold, signals, 0)
                signal = np.asarray(signals).flatten()
                temp_signals = np.append(signal, np.zeros(interval_length - signal.size % interval_length)).reshape(-1, interval_length)
                argmax = np.argmax(temp_signals, axis=1)

                segment = np.asarray(ecg_segments).flatten()
                num_lines, writer = process_signal(writer, datetime_iter, signal, segment, argmax)
                lines += num_lines
                batch = []
                datetimes = []
                ecg_segments = []
                # print(time.time() - curr, " Process time")


            # Steps through all the different buffers, and if the buffer is empty reads more from the file.
            if ecg_segment.size > stack * window_size * 2:
                # curr = time.time()
                ecg_segment = ecg_segment[window_size:]
                ecg_temp = ecg_segment[:window_size * stack]
                filtered_segment = filtered_segment[window_size:]
                filtered_temp = filtered_segment[:window_size * stack]
                datetime = datetime_segment[:window_size]
                datetime_segment = datetime_segment[window_size:]
                curr_segment = ecg_temp[ind1:ind2]
                curr_filt = filtered_temp[ind1:ind2]
                # inc_time += time.time() - curr
            elif not EOF:
                iter += 1
                temp_datetime, temp_ecg, EOF = read_ecg_polars(reader, window_size * loading_size) # read_ecg(file, window_size * loading_size)
                datetime_segment = np.append(datetime_segment, temp_datetime)
                ecg_segment = np.append(ecg_segment, temp_ecg)
                if EOF:
                    break
                if iter % update_freq == 0:
                    pbar.update(line_size * len(datetime_segment) * update_freq)

                filtered_segment = filters(ecg_segment, order, low_cutoff, high_cutoff, nyq)
                # ecg_segment = np.pad(ecg_segment, padding, constant_values=(0, 0))
                # filtered_segment = np.pad(filtered_segment, padding, constant_values=(0, 0))

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
                # print(time.time() - curr, " Read time")
                # print(inc_time, "Increment time")
                inc_time = 0
            else:
                break

            if lines > lines_per_file:  # Creates a new file in case the current one has enough lines.
                # curr = time.time()
                writer.write_csv(f, include_header=False)
                writer = pl.DataFrame()
                lines = 0
                file_num += 1

                f = os.path.join('..', folder_selected, filename + '{:03}'.format(file_num) + '.txt')
                # print(time.time() - curr, " Write time")

    # # Writes the final few datapoints into the file, and closes it
    # batch = np.array(batch)[:, 0, :, :]
    # signals = model(batch, training=False).numpy()
    # signals = scipy.special.expit(signals)
    # # argmax = np.argmax(signals, axis=1)
    # # signals = np.where((signals[:, -1] > 0.8)[:, None], np.zeros_like(ecg_segments), signals[:, :-1])
    #
    # datetime_iter = np.concatenate(datetimes).ravel()
    # signal = np.asarray(signals).flatten()
    # segment = np.asarray(ecg_segments).flatten()
    # temp_signals = np.append(signal, np.zeros(interval_length - signal.size % interval_length)).reshape(-1, interval_length)
    # argmax = np.argmax(temp_signals, axis=1)
    # num_lines, writer = process_signal(writer, datetime_iter, signal, segment, argmax)
    # writer.write_csv(f, include_header=False)
    # lines += num_lines
    end = time.time()

    print('elapsed time: ' + str(end - start) + ' seconds')
    # input('Press enter to continue (may need 2 times)')

    del model
    tf.keras.backend.clear_session()
