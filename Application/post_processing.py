import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import re
from collections import deque
import xlsxwriter
from multiprocessing import Process
from xlsxwriter import Workbook
from datetime import datetime as dt
import datetime
import time
from parameters import interval_length
import multiprocessing as mp

"""
TODO
ModuleNotFoundError: No module named 'six'
"""


def process_file(filename):
    global interval_length
    first = True
    last_few = deque(maxlen=8)
    dist = 0
    reset = True
    file = open(os.path.join('..', 'Signal', filename), 'r+')
    lines = []
    for i, line in enumerate(file):
        # print(i)
        dist += 1
        temp = re.findall('([-0-9.]+)', line)
        date = line[:line.index(',')]

        date = dt.strptime(date, '\t%m/%d/%Y %I:%M:%S.%f %p')

        signal = int(temp[-1])
        if signal == 1:
            if dist > 0.8 * interval_length or dist == 1 or first:  # This would mean that the signal is correct
                if dist == 1:  # For the areas where the signal is marked multiple times
                    dist = 0
                else:
                    if dist > 1.7 * interval_length:  # This indicates that the gap is too large
                        # Write the datetime, ecg signal, but no RR interval
                        lines.append((date, ''))
                    elif 1.2 * interval_length < dist < 1.7 * interval_length:  # For when one beat is missed and the next one is also wrong
                        continue
                    else:
                        # Write the datetime, ecg signal, RR interval (or blank if reset is True)
                        lines.append((date, '' if reset else dist / 4))
                        if reset:
                            reset = False

            else:
                reset = True
            if 1.2 * interval_length > dist > 0.8 * interval_length:
                last_few.append(dist)  # add the distance to the running average
            dist = 0  # Reset distance between last and current signal
            if first:
                first = False  # handling first signal
            if len(last_few) == 8:
                interval_length = np.mean(last_few)  # running average of rr interval
    return lines


def write_to_excel(lines, sheet1, row, wb):
    interval_format = wb.add_format({'num_format': '#.00'})
    date_format = wb.add_format({'num_format': 'm/d/y'})
    time_format = wb.add_format({'num_format': 'hh:mm:ss.000'})
    diff_format = wb.add_format({'num_format': '0.00'})
    prev_value = ''
    start_time = lines[0][0]
    intervals = []
    for line in lines:
        sheet1.write_datetime(row, 0, line[0], date_format)
        sheet1.write_datetime(row, 1, line[0], time_format)
        sheet1.write(row, 4, row)
        sheet1.write(row, 5, line[1], interval_format)
        sheet1.write(row, 6, (line[1] - prev_value)**2 if not prev_value == '' and not line[1] == '' else '', diff_format)
        if line[0] - start_time >= datetime.timedelta(minutes=2):
            sheet1.write(row, 7, np.mean(intervals), interval_format)
            sheet1.write(row, 8, np.std(intervals), interval_format)
            start_time = line[0]
            intervals.clear()
        if line[1] != '':
            intervals.append(float(line[1]))
        prev_value = line[1]
        row += 1
    return row


def main():
    root = tk.Tk()
    currdir = os.getcwd()
    root.filename = filedialog.askopenfilenames(initialdir=currdir + "/../Signal", title="Select file",
                                                filetypes=(("txt files", "*.txt"),
                                                           ("all files", "*.*")))
    filenames = list(root.filename)
    root.destroy()

    if len(filenames) == 0:
        return

    root = tk.Tk()
    currdir = os.getcwd()
    root.filename = filedialog.asksaveasfilename(initialdir=currdir + "/../Signal",
                                                 filetypes=(("excel file", "*.xlsx"),), defaultextension=".xlsx")
    saveas = root.filename
    if saveas == '':
        return
    root.destroy()

    start = time.time()

    out = os.path.join('..', 'Signal', saveas)

    wb = Workbook(out)
    sheet1 = wb.add_worksheet('Sheet 1')
    sheet1.set_column(1, 1, 12)
    sheet1.write(0, 0, 'Date')
    sheet1.write(0, 1, 'Time')
    sheet1.write(0, 4, 'Num: ECG')
    sheet1.write(0, 5, 'RR (ms)')
    sheet1.set_column(6, 6, 16)
    sheet1.write(0, 6, 'Squared Diff (ms)')
    sheet1.set_column(7, 9, 14)
    sheet1.write(0, 7, 'Avg 2 minutes')
    sheet1.write(0, 8, 'STD 2 minutes')
    sheet1.write(0, 9, 'STDev')
    sheet1.write(1, 9, 'rMSSD')
    sheet1.write(2, 9, 'STD of Avg')
    sheet1.write(3, 9, 'Avg of STD')

    row = 1

    pool = mp.Pool()
    results = pool.map(process_file, filenames)

    pool.close()
    pool.join()

    for result in results:
        row = write_to_excel(result, sheet1, row, wb)

    sheet1.write_formula('K1', '=STDEV(F:F)')
    sheet1.write_formula('K2', '=SQRT(AVERAGE(G:G))')
    sheet1.write_formula('K3', '=STDEV(H:H)')
    sheet1.write_formula('K4', '=AVERAGE(I:I)')

    end = time.time()
    print('elapsed time: ' + str(end - start))

    wb.close()


if __name__ == "__main__":
    mp.freeze_support()
    main()
    input()
