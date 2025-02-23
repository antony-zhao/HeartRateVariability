import re
from datetime import datetime as dt
import os
import pandas as pd
import numpy as np
from dataset import random_sampling

"""
Reads from the ecg file and the excel file, and creates two new files. One containing the raw ECG signals, and the
other containing the R-peaks, each line corresponding to the same line in the other file. 1 for R-peak and 0 otherwise.
"""

def make_csv(xl_file, ecg_file, out_file):
    df_excel = pd.read_excel(xl_file, sheet_name=6, usecols=['Date'])
    df = pd.read_csv(ecg_file, comment='#')
    df.columns = df.columns.str.replace(' ', '')

    df['Time'] = pd.to_datetime(df['Time'], format="%m/%d/%Y %I:%M:%S.%f %p", exact=False)
    df = df.merge(df_excel, left_on='Time', right_on='Date', how='outer')

    keys = df.columns.to_list()
    print(keys)

    df = df[[keys[1], keys[2]]]
    df[keys[1]] = pd.to_numeric(df[keys[1]], errors ='coerce').fillna(0)
    ecg = df[keys[1]]
    markings = 1 - df[keys[2]].isna().to_numpy()
    # markings[1:][markings[:-1]==markings[1:]] = 0
    diff = np.diff(np.r_[0, markings, 0])  # Add padding with zeros
    starts = np.where(diff == 1)[0]  # Indices where blocks of 1s start
    ends = np.where(diff == -1)[0]  # Indices where blocks of 1s end
    correct_markings = np.zeros_like(markings)

    for start, end in zip(starts, ends):
        block_values = ecg[start:end]
        baseline = np.gradient(ecg[[start - 20, end + 20]])
        max_index = start + np.argmin(np.abs(baseline))  # Find the index of the maximum value
        correct_markings[max_index] = 1  # Set only the max value's position to 1

    df[keys[2]] = correct_markings
    lower_bound = np.nonzero(markings == 1)[0][0] - 200
    upper_bound = np.nonzero(markings == 1)[0][-1] + 200
    df = df[lower_bound:upper_bound]

    df.to_csv(out_file, sep=',', header=False, index=False)

if __name__ == '__main__':
    # Training Data
    xl_files = ['WA-20_File4.xlsx', 'WA-20_File17.xlsx', 'WA-20_File30.xlsx']
    ecg_file = os.path.join('..', 'ECG_Data', 'WA20_pre1.ascii')
    out_files = ['finetune_train1.csv', 'finetune_train2.csv', 'finetune_train3.csv']
    for xl_file_name, out_file_name in zip(xl_files, out_files):
        xl_file = os.path.join('..', 'Signal', xl_file_name)  # Data files
        out_file = os.path.join('..', 'Training', out_file_name)
        if not os.path.exists(out_file):
            make_csv(xl_file, ecg_file, out_file)


    # Validation Data
    xl_files = ['WA-24_File2.xlsx', 'WA-24_File20.xlsx', 'WA-24_File32.xlsx']
    ecg_file = os.path.join('..', 'ECG_Data', 'WA24_pre1.ascii')
    out_files = ['finetune_val1.csv', 'finetune_val2.csv', 'finetune_val3.csv']
    for xl_file_name, out_file_name in zip(xl_files, out_files):
        xl_file = os.path.join('..', 'Signal', xl_file_name)  # Data files
        out_file = os.path.join('..', 'Training', out_file_name)
        if not os.path.exists(out_file):
            make_csv(xl_file, ecg_file, out_file)