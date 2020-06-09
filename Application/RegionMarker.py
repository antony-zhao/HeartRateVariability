from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D,\
                                    Activation, LeakyReLU, GRU, LSTM, TimeDistributed
from keras.preprocessing import sequence
import tensorflow as tf
from Methods import *
import os
import keras.backend as K
from matplotlib import pyplot as plt
from tensorflow import keras
from datetime import datetime


ecg1 = []
s1 = []

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

interval_length = 800
step = interval_length // 8


Model = Sequential()
Model.add(Conv1D(input_shape=(interval_length, 1), filters=16, kernel_size=6, strides=2, activation='relu'))
Model.add(MaxPooling1D())
Model.add(Conv1D(32, kernel_size=12, strides=2, activation='relu'))
Model.add(MaxPooling1D())
Model.add(Flatten())
Model.add(Dense(128))
Model.add(Activation('relu'))
Model.add(Dropout(0.4))
Model.add(Dense(56))
Model.add(Activation('relu'))
Model.add(Dropout(0.4))
Model.add(Dense(2))
Model.add(Activation('softmax'))


def load_model(model_file):
    Model.load_weights(model_file)


def train_marker(epochs, model_file, batch_size=128, learning_rate=0.01):
    for x in open(os.path.join('..', 'Training', 'ecg3.txt')):
        ecg1.append(float(re.findall('([-0-9.]+)', x)[-1]))

    for x in open(os.path.join('..', 'Training', 'sig3.txt')):
        s1.append(float(re.findall('([-0-9.]+)', x)[-1]))

    # plt.plot(range(len(ecg1)), ecg1)
    # plt.plot(range(len(s1)), s1)
    # plt.show()
    x_train, y_train = sequential_sampling(ecg1, s1, interval_length, step, stack=False)
    temp = y_train.tolist()
    y_train = []
    for i in temp:
        if 1 not in i:
            y_train.append(1)
        else:
            y_train.append(0)
    y_train = np.asarray(y_train)
    x_train = np.append(x_train, -x_train, axis=0)
    y_train = np.append(y_train, y_train, axis=0)
    x_train = x_train.reshape(x_train.size // (1 * interval_length), interval_length, 1)
    # x_train = sequence.pad_sequences(x_train, maxlen=4)
    print(x_train.shape)
    optim = tf.keras.optimizers.Adam(lr=learning_rate)
    Model.compile(optimizer=optim, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    #
    Model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2)

    Model.save(model_file)

    for i in range(x_train.shape[0]):
        temp = x_train[i]
        #
        x = Model.predict(temp.reshape(1, interval_length, 1))
        if x.argmax==1:
            plt.plot(range(temp.size), temp)
            plt.show()


train_marker(50, 'Marker.h5', learning_rate=0.005)
