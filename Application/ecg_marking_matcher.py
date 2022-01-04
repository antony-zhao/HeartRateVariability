import xlrd
import re
from datetime import datetime as dt
import os

"""
Reads from the ecg file and the excel file, and creates two new files. One containing the raw ECG signals, and the
other containing the R-peaks, each line corresponding to the same line in the other file. 1 for R-peak and 0 otherwise.
"""

total_count = 10000000  # Maximum lines to copy over
count = 0  # Current number of lines

xl_file = os.path.join('..', 'Signal', 'T21 - April 1st 6pm - 2nd 6pm - epoch data - Hand cleaned.xlsx')  # Data files
ascii_file_name = os.path.join('..', 'ECG_Data', 'T21_transition example3_900s.ascii')
ecg_file = open(os.path.join('..', 'Training', 'ecg.txt'), 'w')  # Output files
signal_file = open(os.path.join('..', 'Training', 'sig.txt'), 'w')

wb = xlrd.open_workbook(xl_file)
page = wb.sheet_by_index(0)  # The excel page for the relevant data

row = 1  # The row of the dates in the excel sheet

xl_date = xlrd.xldate_as_datetime(page.cell_value(row, 0), wb.datemode)

with open(ascii_file_name, 'r') as ascii_file:
    for line in ascii_file:
        if line[0] == '#' or len(line) == 1:  # Skips through comments and empty lines in the beginning
            continue
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
            ecg_file.write('0\n')
        else:
            ecg_file.write(str(ecg) + '\n')

        if xl_date != date:  # For when there isn't a peak
            signal_file.write('0\n')
        else:
            signal_file.write('1\n')
            row = min(row + 1, page.nrows - 1)
            xl_date = xlrd.xldate_as_datetime(page.cell_value(row, 0), wb.datemode)

        if count > total_count:
            break
