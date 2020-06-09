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

interval_length = 400
step = interval_length // 4

#
# Model = Sequential()
# Model.add(Conv1D(input_shape=(interval_length, 1), filters=32, kernel_size=12, strides=2, activation='relu'))
# Model.add(MaxPooling1D())
# Model.add(Conv1D(64, kernel_size=16, strides=2, activation='relu'))
# Model.add(MaxPooling1D())
# Model.add(LSTM(512))
# Model.add(Activation('relu'))
# Model.add(Dropout(0.4))
# Model.add(Dense(200))
# Model.add(Activation('relu'))
# Model.add(Dropout(0.4))
# Model.add(Dense(interval_length))
# Model.add(Activation('sigmoid'))

Model = Sequential()
Model.add(Conv1D(input_shape=(interval_length, 1), filters=32, kernel_size=12, strides=2, activation='relu'))
Model.add(MaxPooling1D())
Model.add(Conv1D(64, kernel_size=16, strides=2, activation='relu'))
Model.add(MaxPooling1D())
Model.add(Flatten())
Model.add(Dense(512)) #normally 512
Model.add(Activation('relu'))
Model.add(Dropout(0.4))
Model.add(Dense(512)) #normally 512
Model.add(Activation('relu'))
Model.add(Dropout(0.4))
Model.add(Dense(interval_length))
Model.add(Activation('sigmoid'))

def load_model(model_file):
    Model.load_weights(model_file)


def distance(y_true, y_labels):
    return K.mean(K.abs(K.argmax(y_true) - K.argmax(y_labels)))


def train_model(epochs, model_file, samples=10000, batch_size=512, learning_rate=0.01):
    for x in open(os.path.join('..', 'Training', 'ecg3.txt')):
        ecg1.append(float(re.findall('([-0-9.]+)', x)[-1]))

    for x in open(os.path.join('..', 'Training', 'sig3.txt')):
        s1.append(float(re.findall('([-0-9.]+)', x)[-1]))

    # plt.plot(range(len(ecg1)), ecg1)
    # plt.plot(range(len(s1)), s1)
    # plt.show()

    # x_train, y_train = random_sampling(ecg1, s1, samples, interval_length)
    x_train, y_train = sequential_sampling(ecg1, s1, interval_length, step, stack=False)
    x_train = np.append(x_train, -x_train, axis=0)
    y_train = np.append(y_train, y_train, axis=0)
    x_train = x_train.reshape(x_train.size // (1 * interval_length), interval_length, 1)
    # x_train = sequence.pad_sequences(x_train, maxlen=4)
    print(x_train.shape)
    y_train = y_train.reshape(y_train.size // interval_length, interval_length)
    optim = tf.keras.optimizers.Adadelta(lr=learning_rate)
    Model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['categorical_accuracy', distance])

    Model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2)

    Model.save(model_file)
