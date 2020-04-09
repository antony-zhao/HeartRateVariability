from matplotlib import pyplot as plt
import numpy as np
import os
from Methods import *
import tensorflow as tf
from Model import Model, train_model, load_model, interval_length
import h5py
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

step = interval_length // 4

model_file = 'Model3.h5'

ecg = []
file = open(os.path.join('..', 'ECG_Data', 'T21 - whole recording data.ascii'), 'r')
f = open(os.path.join('..', 'Signal', 'T21Signal.txt'), 'w')

ecg_from_file(ecg, os.path.join('..', 'ECG_Data', 'T21_transition example2_600s.ascii'))

signal = np.zeros(interval_length)
load_model(model_file)

start = time.time()


def read_ecg(ecg_file, count):
    e = False
    x = ecg_file.readline()
    sig = np.zeros(count)
    if len(x) == 0:
        e = True
        return sig.reshape(count, 1), e
    sig[0] = float(re.findall('([-0-9.]+)', x)[-1])
    for i in range(1, count):
        x = ecg_file.readline()
        if len(x) == 0:
            e = True
            break
        sig[i] = float(re.findall('([-0-9.]+)', x)[-1])
    return sig.reshape(count, 1), e


def write_signal(sig_file, sig, ecg):
    for e, s in zip(sig, ecg):
        if s > 1 / (interval_length / step):
            s = 1
        else:
            s = 0
        sig_file.write('{},{}\n'.format(e, s))


read = ""
while len(read) < 1 or read[0] == "#":
    read = file.readline()
file.readline()

ecg_temp, EOF = read_ecg(file, interval_length)

for i in range(0, len(ecg) - interval_length, step):
    temp = np.copy(ecg_temp)
    temp -= np.average(temp)
    temp *= 100
    temp = Model.predict(temp.reshape(1, interval_length, 1))
    temp = temp.reshape(interval_length, )
    temp[temp < 0.4] = 0
    temp[temp >= 0.4] = 1
    #temp_max = temp.argmax()
    #temp[range(interval_length)] = 0
    #temp[temp_max] = 1
    signal += temp / (interval_length / step)
    '''
    plt.plot(ecg_temp)
    plt.plot(temp, color='green')
    plt.plot(signal, color='orange')
    plt.legend(["ECG", "Temp", "Signal"])
    plt.axis([0, interval_length, -0.5, 1])
    plt.show()
    '''
    ecg_temp[0:interval_length - step] = ecg_temp[step:]
    ecg_temp[interval_length - step:], EOF = read_ecg(file, step)
    write_signal(f, signal[:step])
    signal[0:interval_length - step] = signal[step:]
    signal[interval_length - step:] = 0
    signal[signal < 0.1] = 0
    '''
    plt.plot(ecg_temp)
    plt.plot(signal, color='orange')
    plt.legend(["ECG", "Signal"])
    plt.axis([0, interval_length, -0.5, 1])
    plt.show()
    '''
end = time.time()


print('elapsed time: ' + str(end - start))

signal = []
signal_from_file(signal, os.path.join('..', 'Signal', 'SignalPy.txt'))

plt.plot(range(len(ecg)), ecg)
# plt.plot(range(len(temp_ecg)), temp_ecg)
plt.plot(range(len(signal)), signal)
plt.axis([0, 6000, -0.5, 1])
plt.show()

del Model
tf.keras.backend.clear_session()

f.close()
