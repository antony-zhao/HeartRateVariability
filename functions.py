
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 10:43:04 2019

@author: tony
"""
from ctypes import *
from matplotlib import pyplot as plt
import numpy as np

def ecg_from_file(ecg, filename):
    f = open(filename, 'r')
    for x in f:
        ecg.append(float(x[x.index(',') + 1:x.index('\n')]))
    f.close()

def PT_from_file(PT,filename):
    f = open(filename,'r')
    f.readline()
    for x in f:
        PT.append(int(x[:x.index('\n')]))
    f.close()

def signal_from_file(signal, filename):
    f = open(filename,'r')
    for x in f:
        signal.append(float(x))
    f.close()
    return signal

def signal_guess(ecg):
    signal = np.zeros(len(ecg))
    for x in range(0,len(ecg),400):
        np.put(signal,ecg[x:x+400].argmax()+x,1)
    return signal

def find1s(signal):
    ind = []
    for x in range(len(signal)):
        if signal[x] == 1:
            ind.append(x)
    return ind

def plot(ecg, signal):
    x = range(ecg.shape[0])
    plt.plot(x, ecg)
    plt.plot(x, signal)
    plt.show()
    

def sample(ecg,signal,input_dim):
    x = []
    y = []
    for n in range(0, ecg.shape[0] - input_dim, int(input_dim/2)):
        x.append(ecg[n:n + input_dim])
        y.append(signal[n:n + input_dim])
    x = np.asarray(x)
    y = np.asarray(y)
    return x.reshape(x.shape[0], x.shape[1]), y.reshape(y.shape[0], y.shape[1])

def plot_prediction(x,model):
    y = model.predict(x.reshape(1, x.shape[0]))
    plt.plot(x)
    plt.plot(y)
    y = np.ndarray.tolist(y)[0]
    plt.show()


def heartbeat_yes_no(y_train):
    y = np.ndarray([0,1])
    for n in y_train:
        y = np.append(y,np.argwhere(n == 1).size)
    return y

def plot_y_n(x_train,model):
    print(model.predict(x_train.reshape(1,x_train.shape[0])))
    plt.plot(x_train)

ecg = []
signal = []
ecg2 = []
ecg_from_file(ecg,"Sample 7.ascii")
signal_from_file(signal, "Signal.txt")
#signal_from_file(ecg2,"ECG.txt")
""" signal = signal_from_file(signal,"Sample1Corrected.txt")
ecg = ecg_from_file(ecg,"Sample 2.ascii")
signal = signal_from_file(signal, "Sample2Corrected.txt")
ecg = ecg_from_file(ecg,"Sample 3.ascii")
signal = signal_from_file(signal,"Sample3Corrected.txt")
ecg = ecg_from_file(ecg,"Sample 4.ascii")
signal = signal_from_file(signal, "Sample4Corrected.txt")
ecg = np.asarray(ecg)
ecg = np.append(ecg,-ecg)
signal = np.asarray(signal)
signal = np.append(signal,signal)
ecg = ecg.reshape(ecg.shape[0],1)
signal = signal.reshape(signal.shape[0],1) """
plt.plot(range(len(signal)),signal)
plt.plot(range(len(ecg)),ecg)
#plt.plot(range(len(ecg2)),ecg2)
#plt.plot(range(len(ecg)),ecg)
plt.show()