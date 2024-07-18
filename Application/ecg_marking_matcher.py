import xlrd
import re
from datetime import datetime as dt
import os
import pandas as pd

"""
Reads from the ecg file and the excel file, and creates two new files. One containing the raw ECG signals, and the
other containing the R-peaks, each line corresponding to the same line in the other file. 1 for R-peak and 0 otherwise.
"""

total_count = 10000000  # Maximum lines to copy over
count = 0  # Current number of lines

xl_file = os.path.join('..', 'Signal', 'RAT #01_2021_baseline_EPOCH DATA.xlsx')  # Data files
ecg_file_name = os.path.join('..', 'ECG_Data', 'RAT #01_2021_baseline.ascii')
'''
Rat Val: 'RAT #12_2016_WK4.ascii'
Rat Train: RAT #01_2021_baseline.ascii

Mouse Train: T22 - 2 hour data.ascii
Mouse Val: T21_transition example3_900s.ascii
'''

df_excel = pd.read_excel(xl_file, sheet_name=6)
df = pd.read_csv(ecg_file_name, comment='#', header=None)

print(df_excel)
