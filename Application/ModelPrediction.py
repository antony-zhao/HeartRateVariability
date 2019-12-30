from matplotlib import pyplot as plt
import numpy as np
import os
from Methods import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D, Activation
import tensorflow as tf
from itertools import repeat
from Model import Model
import time

os.environ['KMP_DUPLICATE_LIB_OK']='True'

interval_length = 400
step = 100

ecg = []
ecg_from_file(ecg, os.path.join('..','ECG_Data','T21.ascii'))
signal = np.zeros((len(ecg)))
std = []

start = time.time()

with tf.device("/device:GPU:0"):
    for i in range(0,len(ecg) - interval_length,step):
        #if i % 10000 == 0:
        #    print(i)
        temp = np.asarray(ecg[i:i+interval_length])*100
        temp = Model.predict(temp.reshape(1,interval_length, 1))
        std.extend(repeat(np.std(temp*10),step))
        #if np.var(temp) > 0.002:
            #continue
        #else:
        temp -= np.average(temp)
        signal[i:i + interval_length] += temp.reshape(400,)
end = time.time()

print('elapsed time: ' + str(end-start))

signal /= (interval_length/step)

f = open(os.path.join('..','Signal','SignalPy.txt'),'w')
for i in signal:
    f.write(str(i) + '\n')
f.close()

fig, axs = plt.subplots(2, sharex = True)

axs[0].plot(range(len(ecg)), ecg)
axs[0].plot(range(len(signal)), signal)
axs[0].axis([0,6000,-0.5,1])
axs[1].plot(range(len(std)), std)
axs[1].axis([0,6000,-0.5,1])
plt.show()

del Model
tf.keras.backend.clear_session()