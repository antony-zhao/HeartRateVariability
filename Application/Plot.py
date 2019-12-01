from Methods import *
import os.path


ecg = []
signal = []
ecg_from_file(ecg,os.path.join('..','ECG_Data','T21.ascii'))
signal_from_file(signal, os.path.join('..','Signal','Signal.txt'))
plot(ecg,signal)