from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D, Activation
import tensorflow as tf
import re
from Methods import *
import os
from tensorflow import keras
from matplotlib import pyplot as plt
from datetime import datetime


ecg1 = []
s1 = []

interval_length = 400

for x in open(os.path.join('..','Training','Ecg1.txt')):
    ecg1.append(float(re.findall('([-0-9.]+)', x)[-1]))

for x in open(os.path.join('..','Training','Sig1.txt')):
    s1.append(float(re.findall('([-0-9.]+)', x)[-1]))

x_train, y_train = random_sampling(ecg1, s1, 5000, interval_length)
x_test, y_test = random_sampling(ecg1, s1, 100, interval_length)

date = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = os.path.join('Training', date)


Model = Sequential()
Model.add(Conv1D(input_shape = (interval_length,1), filters = 20, kernel_size = 4,strides = 2, activation = 'relu'))
Model.add(MaxPooling1D())
Model.add(Conv1D(50, kernel_size = 8, strides = 2, activation = 'relu'))
Model.add(MaxPooling1D())
Model.add(Flatten())
Model.add(Dense(600))
Model.add(Activation('relu'))
Model.add(Dropout(0.2))
Model.add(Dense(interval_length))
Model.add(Activation('softmax'))
modelFile = 'Model.h5'

try:
    Model.load_weights(modelFile)
except:
    Model.compile(optimizer = 'Adadelta', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()


    def lr_schedule(epoch):
        learning_rate = 0.2
        if epoch > 10:
            learning_rate = 0.02
        if epoch > 20:
            learning_rate = 0.01
        if epoch > 50:
            learning_rate = 0.005

        tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
        return learning_rate

    lr_callback = keras.callbacks.LearningRateScheduler(lr_schedule)
    tsbrd = tf.keras.callbacks.TensorBoard(log_dir = logdir)

    with tf.device("/device:CPU:0"):
        hist = Model.fit(x_train, y_train, batch_size = 64, validation_data = (x_test, y_test), callbacks = [tsbrd, lr_callback], epochs = 50, verbose = 2)

        Model.save(modelFile)
'''
with tf.device("/device:CPU:0"):
    plt.plot(range(len(x_test[0, :, 0])), x_test[0, :, 0])
    plt.plot(range(len(y_test[0, :])), y_test[0, :])
    plt.plot(range(len(x_test[0, :, 0])), Model.predict(x_test[0, :, 0].reshape(1, interval_length, 1))[0])
    plt.show()
'''