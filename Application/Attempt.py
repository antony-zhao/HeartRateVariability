from matplotlib import pyplot as plt
import numpy as np
import os
from Methods import *

interval_length = 400
step = 100

ecg = []
ecg_from_file(ecg, os.path.join('..','ECG_Data','T21.ascii'))
signal = np.zeros((len(ecg)))

for i in range(0,len(ecg) - interval_length,step):
    if i % 10000 == 0:
        print(i)
    temp = ecg[i:i+interval_length]
    if np.var(temp) > 0.01:
        continue
    else:
        temp -= np.average(temp)
        signal[np.argmax(temp) + i] = 1

f = open(os.path.join('..','Signal','SignalPy.txt'),'w')
for i in signal:
    f.write(str(i) + '\n')
f.close()

plt.plot(range(len(ecg)), ecg)
plt.plot(range(len(signal)), signal)
plt.axis([0,6000,-0.5,1])
plt.show()