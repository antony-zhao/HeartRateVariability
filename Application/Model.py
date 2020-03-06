from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D, Activation, GRU
import tensorflow as tf
from Methods import *
import os
from matplotlib import pyplot as plt
from tensorflow import keras
from datetime import datetime


ecg1 = []
s1 = []

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

interval_length = 400

Model = Sequential()
Model.add(Conv1D(input_shape = (interval_length, 1), filters = 10, kernel_size = 4,strides = 2, activation = 'relu'))
Model.add(MaxPooling1D())
Model.add(Conv1D(20, kernel_size = 8, strides = 2, activation = 'relu'))
Model.add(MaxPooling1D())
Model.add(Flatten())
Model.add(Dense(200))
Model.add(Activation('relu'))
Model.add(Dropout(0.2))
Model.add(Dense(interval_length))
Model.add(Activation('sigmoid'))


def load_model(modelFile):
    Model.load_weights(modelFile)

def train_model(epochs, modelFile, samples = 10000, batch_size = 512):
    for x in open(os.path.join('..', 'Training', 'ecg2.txt')):
        ecg1.append(float(re.findall('([-0-9.]+)', x)[-1]))

    for x in open(os.path.join('..', 'Training', 'sig2.txt')):
        s1.append(float(re.findall('([-0-9.]+)', x)[-1]))

    x_train, y_train = random_sampling(ecg1, s1, samples, interval_length)
    #x_train, y_train = sequential_sampling(ecg1, s1, interval_length, interval_length//2)

    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join('Training', date)

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
    print(x_train.shape)
    Model.fit(x_train, y_train, batch_size = batch_size, callbacks = [tsbrd, lr_callback],epochs = epochs, verbose = 2)

    Model.save(modelFile)
"""
for x in open(os.path.join('..', 'Training', 'ecg2.txt')):
    ecg1.append(float(re.findall('([-0-9.]+)', x)[-1]))

for x in open(os.path.join('..', 'Training', 'sig2.txt')):
    s1.append(float(re.findall('([-0-9.]+)', x)[-1]))

plt.plot(range(len(ecg1)), ecg1)
plt.plot(range(len(s1)), s1)
plt.show()
"""
