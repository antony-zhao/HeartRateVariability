from Methods import *
import os.path


ecg = []
signal = []
ecg_from_file(ecg,os.path.join('..','ECG_Data','T21.ascii'))
signal_from_file(signal, os.path.join('..','Signal','Signal.txt'))
plt.plot(range(len(signal[0:1000000])),signal[0:1000000])
plt.plot(range(len(ecg[0:1000000])),ecg[0:1000000])
plt.show()