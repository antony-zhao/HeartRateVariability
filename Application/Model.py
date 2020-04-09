from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D, Activation, GRU
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

interval_length = 400

Model = Sequential()
Model.add(Conv1D(input_shape=(interval_length, 1), filters=32, kernel_size=12, strides=2, activation = 'relu'))
Model.add(MaxPooling1D())
Model.add(Conv1D(64, kernel_size=16, strides=2, activation = 'relu'))
Model.add(MaxPooling1D())
Model.add(Flatten())
Model.add(Dense(512))
Model.add(Activation('relu'))
Model.add(Dropout(0.4))
Model.add(Dense(512))
Model.add(Activation('relu'))
Model.add(Dropout(0.4))
Model.add(Dense(interval_length))
Model.add(Activation('sigmoid'))


def load_model(modelFile):
    Model.load_weights(modelFile)


def distance(y_true, y_labels):
    return K.mean(K.abs(K.argmax(y_true) - K.argmax(y_labels)))


def train_model(epochs, modelFile, samples=10000, batch_size=512):
    for x in open(os.path.join('..', 'Training', 'ecg4.txt')):
        ecg1.append(float(re.findall('([-0-9.]+)', x)[-1]))

    for x in open(os.path.join('..', 'Training', 'sig4.txt')):
        s1.append(float(re.findall('([-0-9.]+)', x)[-1]))

    x_train, y_train = random_sampling(ecg1, s1, samples, interval_length)
    x_train = np.append(x_train, -x_train)
    y_train = np.append(y_train, y_train)
    x_train = x_train.reshape(samples * 2,interval_length,1)
    y_train = y_train.reshape(samples * 2, interval_length)
    optim = keras.optimizers.Adam(lr=0.01, clipvalue = 5)
    Model.compile(optimizer = optim, loss = 'categorical_crossentropy', metrics = ['categorical_accuracy', distance], verbose=1)

    Model.fit(x_train, y_train, batch_size = batch_size,epochs = epochs, verbose = 2)

    Model.save(modelFile)