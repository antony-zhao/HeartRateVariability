import multiprocessing
from pathlib import Path
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import re
from collections import deque
from xlsxwriter import Workbook
from datetime import datetime as dt
import datetime
import time
import multiprocessing as mp
import json
import tqdm
from functools import partial
from config import window_size, stack, scale_down, datapoints, \
    lines_per_file, max_dist_percentage, low_cutoff, high_cutoff, nyq, order, interval_length, threshold, animal, \
    pad_behind

"""
After the model_prediction.py program splits the data into multiple files, this
program uses multiprocessing to process multiple files in parallel and rejoins the
final data into a single excel sheet.
"""

min_dist = 1 - max_dist_percentage
max_double_dist = 2 * min_dist
max_dist = 1 + max_dist_percentage


def process_file(filenames, filename):
    """
    Processes a single file and returns
    Args:
        filenames (list): A list of all the file names, used
            for tracking which files are done processing
        filename (str): The file to be processed
    """
    global interval_length
    avg_interval_length = interval_length
    first = True  # For the first signal of the file
    last_few = deque(maxlen=8)
    dist = 0  # Distance from the last peak
    reset = True  # If the gap is too wide and it needs to reset the distance
    file = open(os.path.join('..', 'Signal', filename), 'r+')
    lines = []  # Lines that will be later written to the excel sheet,
    # contains tuples of the datetime, as well as the distance, which is an empty string if it's the first
    # signal of the file, or if the gap was too wide
    for i, line in enumerate(file):
        dist += 1
        temp = re.findall('([-0-9.]+)', line)
        date = line[:line.index(',')]
        try:
            date = dt.strptime(date, ' %m/%d/%Y %I:%M:%S.%f %p')
        except ValueError:
            date = dt.strptime(date, '%m/%d/%Y %I:%M:%S.%f %p')

        signal = int(temp[-1])
        if signal == 1:
            if dist > min_dist * avg_interval_length or dist == 1 or first:  # This would mean that the signal is correct
                if dist == 1:  # For the areas where the signal is marked multiple times
                    dist = 0
                else:
                    if dist > max_double_dist * avg_interval_length:  # This indicates that the gap is too large
                        lines.append((date, ''))  #
                    elif 1.2 * avg_interval_length < dist < max_double_dist * avg_interval_length:  # For when one beat is missed and the
                        # next one is also wrong
                        continue
                    else:
                        lines.append((date, '' if reset else dist / 4))  # dist/4 because ours is sampled as 4
                        # datapoints per milisecond
                        if reset:
                            reset = False

            else:
                reset = True
            if max_dist * avg_interval_length > dist > min_dist * avg_interval_length:
                last_few.append(dist)  # Add the distance to the running average
            dist = 0  # Reset distance between last and current signal
            if first:
                first = False  # handling first signal
            if len(last_few) == 8:
                avg_interval_length = np.mean(last_few)  # running average of rr interval
    print("{}/{} file has been completed".format(filenames.index(filename) + 1, len(filenames)))
    return lines


def write_to_excel(lines, sheet, row, formats, sheet_num):
    """
    Writes the lines to the sheet, as well as calculating some useful data which are
    related to heart rate variability.

    Args:
        lines (list): A list containing the datetimes of when the signal occured as well as
            the distance between it and the previous signal
        sheet (Workbook.worksheet): The excel sheet which the data will be written to
        row (int): The row to write the data to (also be returned for the next call as the
            function is called once for the data of each file
        formats (tuple): A tuple containing the different formats to write to cells.

    """
    interval_format = formats[0]
    date_format = formats[1]
    time_format = formats[2]
    diff_format = formats[3]
    prev_value = ''
    start_time = lines[0][0]
    intervals = []  # Tracks the last RR-intervals for a certain duration, which we get data from after
    for line in lines:
        if sheet_num == 1 and (line[0].time() < datetime.time(hour=8) or line[0].time() > datetime.time(hour=20)):
            continue
        if sheet_num == 2 and (datetime.time(hour=20) > line[0].time() > datetime.time(hour=8)):
            continue
        sheet.write_datetime(row, 0, line[0], date_format)  # Splits the datetime value
        sheet.write_datetime(row, 1, line[0], time_format)
        sheet.write(row, 4, row)  # Just the number of the signal (shouldn't be different from the row
        # in our version but can be changed)
        sheet.write(row, 5, line[1], interval_format)  # The number of ms between peaks
        sheet.write(row, 6, (line[1] - prev_value) ** 2 if not prev_value == '' and not line[1] == '' else '',
                    diff_format)  # The squared difference between the RR-interval of this and
        # the previous data (if possible)
        if line[0] - start_time >= datetime.timedelta(minutes=5):  # The average and standard deviation in the
            # RR-interval in 5 minutes
            if len(intervals) > 0:
                sheet.write(row, 7, np.mean(intervals), interval_format)
                sheet.write(row, 8, np.std(intervals), interval_format)
                start_time = start_time + datetime.timedelta(minutes=5)
                intervals.clear()
        if line[1] != '':
            intervals.append(float(line[1]))
        prev_value = line[1]
        row += 1
    return row


def main():
    # Handles selecting the file and selecting a place/name for the new file
    root = tk.Tk()
    currdir = os.getcwd()
    root.filename = filedialog.askopenfilenames(initialdir=currdir + "/../ECG_Data", title="Select file",
                                                filetypes=(("txt files", "*.txt"),
                                                           ("all files", "*.*")))
    filenames = list(root.filename)
    filenames.sort()
    total_size = 0
    for file_name in filenames:
        total_size += os.stat(file_name).st_size
    root.destroy()

    if len(filenames) == 0:
        print("Please select files")
        return

    root = tk.Tk()
    currdir = os.getcwd()
    par = Path(currdir).parent
    signal_dir = str(par) + r"\Signal"
    root.filename = filedialog.asksaveasfilename(initialdir=signal_dir,
                                                 filetypes=(("excel file", "*.xlsx"),), defaultextension=".xlsx")
    saveas = root.filename
    if saveas == '':
        print("Please input a filename")
        return
    root.destroy()

    start = time.time()

    out = os.path.join(signal_dir, saveas)

    # Handles the multiprocessing, and runs multiple instances of the process_file function
    pool = mp.Pool(processes=max(1, multiprocessing.cpu_count() - 2))
    func = partial(process_file, filenames)
    results = pool.map(func, filenames)

    pool.close()
    pool.join()

    # Adds labels to the excel sheet
    wb = Workbook(out)
    sheets = [wb.add_worksheet('All Data'), wb.add_worksheet('Light'), wb.add_worksheet('Dark')]
    for i, sheet in enumerate(sheets):
        sheet.set_column(1, 1, 12)
        sheet.write(0, 0, 'Date')
        sheet.write(0, 1, 'Time')
        sheet.write(0, 4, 'Num: ECG')
        sheet.write(0, 5, 'RR (ms)')
        sheet.set_column(6, 6, 16)
        sheet.write(0, 6, 'Squared Diff (ms)')
        sheet.set_column(7, 9, 14)
        sheet.write(0, 7, 'Avg 5 minutes')
        sheet.write(0, 8, 'STD 5 minutes')
        sheet.write(0, 9, 'Statistics')
        sheet.write(1, 9, 'STDev')
        sheet.write(2, 9, 'rMSSD')
        sheet.write(3, 9, 'STD of Avg (SDANN)')
        sheet.write(4, 9, 'Avg of STD (SDNNIDX)')
        sheet.write(5, 9, 'Avg RR Interval')
        sheet.write(6, 9, 'Total HR Data (Hours Min Sec)')
        interval_format = wb.add_format({'num_format': '#.00'})  # Formatting for the data
        date_format = wb.add_format({'num_format': 'm/d/y'})
        time_format = wb.add_format({'num_format': 'hh:mm:ss.000'})
        diff_format = wb.add_format({'num_format': '0.00'})
        formats = (interval_format, date_format, time_format, diff_format)

        row = 1  # Current row in the sheet

        for result in results:  # At this point results contains the list of tuples for each file, and we write
            # that data into the sheet.
            row = write_to_excel(result, sheet, row, formats, i)

        # Add formulas to calculate more useful data
        sheet.set_column(9, 9, 20)
        sheet.set_column(10, 10, 20)
        sheet.write_formula('K2', '=STDEV(F:F)')
        sheet.write_formula('K3', '=SQRT(AVERAGE(G:G))')
        sheet.write_formula('K4', '=STDEV(H:H)')
        sheet.write_formula('K5', '=AVERAGE(I:I)')
        sheet.write_formula('K6', '=AVERAGE(F:F)')
        sheet.write_formula('K7', '=COUNT(F:F)*K6/(24*60*60*1000)', cell_format=time_format)

    end = time.time()
    print('elapsed time: ' + str(end - start))

    wb.close()


if __name__ == "__main__":
    # More handling of multiprocessing
    mp.freeze_support()
    main()
    # input("Press enter to continue")
