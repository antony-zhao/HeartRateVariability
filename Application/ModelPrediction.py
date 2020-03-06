from matplotlib import pyplot as plt
import numpy as np
import os
from Methods import *
import tensorflow as tf
from Model import Model, train_model, load_model, interval_length
import time

os.environ['KMP_DUPLICATE_LIB_OK']='True'

step = interval_length//2
epochs = 50

model_file = 'Model2.h5'

ecg = []
ecg_from_file(ecg, os.path.join('..','ECG_Data','T21.ascii'))
signal = np.zeros((len(ecg)))


load_model(model_file)

start = time.time()

for i in range(0,len(ecg) - interval_length,step):
    temp = np.asarray(ecg[i:i+interval_length])*100
    temp = Model.predict(temp.reshape(1,interval_length, 1))
    temp -= np.average(temp)
    signal[i:i + interval_length] += temp.reshape(400,)
end = time.time()

print('elapsed time: ' + str(end-start))

signal /= (interval_length/step)

f = open(os.path.join('..','Signal','SignalPy.txt'),'w')
for i in signal:
    f.write(str(i) + '\n')
f.close()

plt.plot(range(len(ecg)), ecg)
plt.plot(range(len(signal)), signal)
plt.axis([0,6000,-0.5,1])
plt.show()

del Model
tf.keras.backend.clear_session()