from openpyxl import load_workbook
import xlrd
import re
from datetime import datetime as dt
import datetime
import os

total_count = 36000000
count = 0

xl_file = os.path.join('..','Signal','T21 - April 1st 6pm - 2nd 6pm - epoch data - Hand cleaned.xlsx')
ascii_file = open(os.path.join('..','ECG_Data', 'T21 - whole recording data.ascii'),'r')
ecg2 = open(os.path.join('..', 'Training', 'ecg4.txt'), 'w')
sig2 = open(os.path.join('..', 'Training', 'sig4.txt'), 'w')

wb = xlrd.open_workbook(xl_file)
page = wb.sheet_by_index(0)

line = ascii_file.readline()
while line[0] == '#' or len(line) < 2:
    line = ascii_file.readline()
line = ascii_file.readline()

date_initialized = False

for n in range(1, page.nrows):
    xl_date = xlrd.xldate_as_datetime(page.cell_value(n,0), wb.datemode)
    line = ascii_file.readline()

    if not date_initialized or date >= xl_date:
        line = ascii_file.readline()
        if len(line) == 0:
            break
        reg = re.findall('([0-9:./APM-]+)', line)
        date = reg[0] + ' ' + reg[1] + ' ' + reg[2]
        date = dt.strptime(date, "%m/%d/%Y %I:%M:%S.%f %p")
        ecg = reg[-1]

        while xl_date > date:
            #sig2.write(str(date) + ' ')
            sig2.write('0\n')
            count += 1
            if ecg == 'x' or ecg == 'PM' or ecg == 'AM':
                ecg2.write('0' + '\n')
                count += 1
            else:
                ecg2.write(str(ecg) + '\n')
                count += 1
            line = ascii_file.readline()
            if len(line) == 0:
                break
            reg = re.findall('([0-9:./APMx-]+)', line)
            date = reg[0] + ' ' + reg[1] + ' ' + reg[2]
            date = dt.strptime(date, "%m/%d/%Y %I:%M:%S.%f %p")
            ecg = reg[-1]
            if len(line) == 0:
                break

        while xl_date == date:
            #sig2.write(str(date) + ' ')
            sig2.write('1\n')
            if ecg == 'x' or ecg == 'PM' or ecg == 'AM':
                ecg2.write('0' + '\n')
            else:
                ecg2.write(str(ecg) + '\n')
            line = ascii_file.readline()
            if len(line) == 0:
                break
            reg = re.findall('([0-9:./APMx-]+)', line)
            date = reg[0] + ' ' + reg[1] + ' ' + reg[2]
            date = dt.strptime(date, "%m/%d/%Y %I:%M:%S.%f %p")
            ecg = reg[-1]



    if count > total_count:
        break