import xlrd
import re
from datetime import datetime as dt
import os
from csv_reader import csv_reader

"""
Reads from the ecg file and the excel file, and creates two new files. One containing the raw ECG signals, and the
other containing the R-peaks, each line corresponding to the same line in the other file. 1 for R-peak and 0 otherwise.
"""

total_count = 10000000  # Maximum lines to copy over
count = 0  # Current number of lines

xl_file = os.path.join('..', 'Signal', 'T21 - April 1st 6pm - 2nd 6pm - epoch data - Hand cleaned.xlsx')  # Data files
ecg_file_name = os.path.join('..', 'ECG_Data', 'T21_transition example3_900s.ascii')
'''
Rat Val: 'RAT #12_2016_WK4.ascii'
Rat Train: RAT #01_2021_baseline.ascii

Mouse Train: T22 - 2 hour data.ascii
Mouse Val: T21_transition example3_900s.ascii
'''
file = open(os.path.join('..', 'Training', 'mouse_val.txt'), 'w')  # Output files

wb = xlrd.open_workbook(xl_file)
page = wb.sheet_by_index(0)  # The excel page for the relevant data

row = 1  # The row of the dates in the excel sheet

xl_date = xlrd.xldate_as_datetime(page.cell_value(row, 0), wb.datemode)
header = True  # For handling if the line is a header or if it is actual data
reader = csv_reader(ecg_file_name, read_line=True)

for line in reader:
    # if header or re.match('\\s*[\\d\\s/:.APM]+,[-\\s\\dx.,]*[-\\s\\dx.]\n', line) is None:
    #     # Skips through the header in the beginning, probably need to edit regex to match data.
    #     continue
    # else:
    #     header = False
    if line == '':
        break
    reg = re.findall('([0-9:./APMx-]+)', line)
    date = reg[0] + ' ' + reg[1] + ' ' + reg[2]
    date = dt.strptime(date, "%m/%d/%Y %I:%M:%S.%f %p")  # The date, to match with the excel data, and output a
    # 1 if there is a peak at the time
    ecg = reg[-1]  # ECG signal
    count += 1

    while xl_date < date:  # In case the ascii file starts after the start of the excel file
        row = min(row + 1, page.nrows - 1)  # Just to make sure it doesn't go to a row that doesn't contain anything
        xl_date = xlrd.xldate_as_datetime(page.cell_value(row, 0), wb.datemode)
        if row == page.nrows - 1:
            break

    if ecg == 'x' or ecg == 'PM' or ecg == 'AM':  # Handles when there isn't a proper ECG signal
        file.write('0')
    else:
        file.write(str(ecg))
    file.write(' ')

    if xl_date != date:  # For when there isn't a peak
        file.write('0\n')
    else:
        file.write('1\n')
        row = min(row + 1, page.nrows - 1)
        xl_date = xlrd.xldate_as_datetime(page.cell_value(row, 0), wb.datemode)

    if count > total_count:
        break
