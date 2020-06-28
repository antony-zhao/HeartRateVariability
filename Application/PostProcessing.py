import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
import re
from collections import deque
import xlsxwriter
from xlsxwriter import Workbook
from datetime import datetime as dt
import time
from Parameters import interval_length

root = tk.Tk()
currdir = os.getcwd()
root.filename = filedialog.askopenfilename(initialdir=currdir+"/../Signal", title="Select file",
                                           filetypes=(("txt files", "*.txt"),
                                                      ("all files", "*.*")))

filename = root.filename
root.destroy()

start = time.time()
file = open(os.path.join('..', 'Signal', filename), 'r+')
prefix = filename[len(filename) - filename[::-1].index("/"):filename.index(".")]
out = os.path.join('..', 'Signal', prefix + '.xlsx')
first = True
last_few = deque(maxlen=8)
dist = 0
reset = True

wb = Workbook(out)
sheet1 = wb.add_worksheet('Sheet 1')
sheet1.set_column(1, 1, 12)
sheet1.write(0, 0, 'Date')
sheet1.write(0, 1, 'Time')
sheet1.write(0, 4, 'Num: ECG')
sheet1.write(0, 5, 'RR (ms)')
row = 1

for i, line in enumerate(file):
    dist += 1
    temp = re.findall('([-0-9.]+)', line)
    date = line[:line.index(',')]
    # print(datetime.split())
    interval_format = wb.add_format({'num_format': '#.00'})
    date_format = wb.add_format({'num_format': 'm/d/y'})
    time_format = wb.add_format({'num_format': 'hh:mm:ss.000'})

    date = dt.strptime(date, '\t%m/%d/%Y %I:%M:%S.%f %p')

    ecg = float(temp[-2])
    signal = int(temp[-1])
    if signal == 1:
        if dist > 0.8 * interval_length or dist == 1 or first:  # This would mean that the signal is correct
            if dist == 1:  # For the areas where the signal is marked multiple times
                dist = 0
            else:
                if dist > 1.7 * interval_length:  # This indicates that the gap is too large
                    pass
                    # Write the datetime, ecg signal, but no RR interval
                    sheet1.write_datetime(row, 0, date, date_format)
                    sheet1.write_datetime(row, 1, date, time_format)
                    sheet1.write(row, 4, row)
                    row += 1
                elif 1.2 * interval_length < dist < 1.7 * interval_length:  # For when one beat is missed and the next one is also wrong
                    continue
                else:
                    # Write the datetime, ecg signal, RR interval (or blank if reset is True)
                    sheet1.write_datetime(row, 0, date, date_format)
                    sheet1.write_datetime(row, 1, date, time_format)
                    sheet1.write(row, 4, row)
                    sheet1.write(row, 5, ' ' if reset else dist / 4, interval_format)
                    row += 1
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
        prev = i

    if int(temp[-1]) == 2:
        pass

end = time.time()
print('elapsed time: ' + str(end - start))

wb.close()
