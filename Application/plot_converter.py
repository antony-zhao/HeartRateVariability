import numpy as np
import tkinter as tk
from tkinter import filedialog
import os
from pathlib import Path
import re
from dataset import random_sampling, filters_from_config

# Converts the output of plot.py into data that can be used to improve the model

root = tk.Tk()  # Prompts user to select file
currdir = os.getcwd()
par = Path(currdir).parent
signal_dir = str(par) + r"\ECG_Data"
root.filename = filedialog.askopenfilename(initialdir=signal_dir, title="Select file",
                                           filetypes=(("txt files", "*.txt"), ("all files", "*.*")))
filename = root.filename
if len(filename) == 0:
    exit(0)
root.destroy()

file = open(filename, 'r+')

ecg = []
sig = []

for i, line in enumerate(file):
    temp = re.findall('([-0-9.]+)', line)  # Regex, 2nd to last is the ECG, and last is the signal (others are
    # date values which are kept in for post_processing.py)
    ecg.append(float(temp[-2]))
    sig.append(int(temp[-1]))

filtered_ecg = filters_from_config(ecg)
x_train, y_train = random_sampling(ecg, filtered_ecg, sig, samples=10000)

np.save(os.path.join('..', 'Training', f'extra_x_train'), x_train)
np.save(os.path.join('..', 'Training', f'extra_y_train'), y_train)

