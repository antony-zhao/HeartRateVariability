import numpy as np
from scipy.signal import resample
import re
import os
import methods
import time
import joblib
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, Normalizer
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, filtfilt, butter


ecg1 = []
lines = 100000
counter = 0
interval_length = 400
step = interval_length // 2
stack = 8 # normally 8
scale_down = 4
datapoints = interval_length // scale_down
T = 0.1          # Sample Period
fs = 4000.0      # sample rate, Hz
low_cutoff = 200      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
high_cutoff = 5
nyq = 0.5 * fs   # Nyquist Frequency
order = 4        # sin wave can be approx represented as quadratic
n = int(T * fs)  # total number of samples

for x in open(os.path.join('..', 'Training', 'ecg1.txt')):
    counter += 1
    ecg1.append(float(re.findall('([-0-9.]+)', x)[-1]))
    if counter >= lines:
        break

ecg1 = np.asarray(ecg1)

b, a = butter(N=order, Wn=low_cutoff/nyq, btype='low', analog=False)
ecg1 = filtfilt(b, a, np.asarray(ecg1))
b, a = butter(N=order, Wn=high_cutoff/nyq, btype='high', analog=False)
ecg1 = filtfilt(b, a, np.asarray(ecg1))


ecg1 = ecg1.reshape(400, -1)
# ecg1 = ecg1 - np.median(ecg1, axis=0).reshape(1, -1)
ecg1 = ecg1 / np.sqrt(np.sum(ecg1**2, axis=0).reshape(1, -1))
# ecg1 = ecg1.reshape(-1, 1)
scaler = MinMaxScaler()
ecg1 = scaler.fit_transform(ecg1)
plt.plot(ecg1.flatten())
plt.show()