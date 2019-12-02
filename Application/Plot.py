from Methods import *
import os.path


ecg = []
signal = []
ecg_from_file(ecg,os.path.join('..','ECG_Data','T21.ascii'))
signal_from_file(signal, os.path.join('..','Signal','Signal.txt'))
plt.plot(range(len(ecg)), ecg)
plt.plot(range(len(signal)), signal)
plt.legend(('ECG','Signal'),loc = 'upper left')
plt.axis([0,6000,-1,1])
plt.show()