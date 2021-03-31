from openpyxl import load_workbook
import xlrd
import re
from datetime import datetime as dt
import datetime
import os

total_count = 10000000
count = 0

xl_file = os.path.join('..', 'Signal', 'T21 - April 1st 6pm - 2nd 6pm - epoch data - Hand cleaned.xlsx') #  T22 - Hand clean Epoch Data.xlsx
ascii_file = open(os.path.join('..', 'ECG_Data', 'T21_transition example3_900s.ascii'), 'r') # T22 - 2 hour data.ascii
ecg2 = open(os.path.join('..', 'Training', 'ecg6.txt'), 'w')
sig2 = open(os.path.join('..', 'Training', 'sig6.txt'), 'w')

wb = xlrd.open_workbook(xl_file)
page = wb.sheet_by_index(0)

row = 1

xl_date = xlrd.xldate_as_datetime(page.cell_value(row, 0), wb.datemode)

for line in ascii_file:
    if line[0] == '#' or len(line) == 1:
        continue
    reg = re.findall('([0-9:./APMx-]+)', line)
    date = reg[0] + ' ' + reg[1] + ' ' + reg[2]
    date = dt.strptime(date, "%m/%d/%Y %I:%M:%S.%f %p")
    ecg = reg[-1]

    if xl_date < date:
        while xl_date < date:
            row = min(row + 1, page.nrows - 1)
            xl_date = xlrd.xldate_as_datetime(page.cell_value(row, 0), wb.datemode)
            if row == page.nrows - 1:
                break

    if xl_date != date:
        sig2.write('0\n')
        count += 1
        if ecg == 'x' or ecg == 'PM' or ecg == 'AM':
            ecg2.write('0\n')
        else:
            ecg2.write(str(ecg) + '\n')
        if len(line) == 0:
            break

    if xl_date == date:
        sig2.write('1\n')
        count += 1
        # else:
        #     sig2.write('0\n')
        if ecg == 'x' or ecg == 'PM' or ecg == 'AM':
            ecg2.write('0' + '\n')
        else:
            ecg2.write(str(ecg) + '\n')
        min(row + 1, page.nrows - 1)
        xl_date = xlrd.xldate_as_datetime(page.cell_value(row, 0), wb.datemode)

    if count > total_count:
        break
