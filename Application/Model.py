from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D, \
    Activation, BatchNormalization
import tensorflow as tf
import re
import random
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import lfilter, lfilter_zi, filtfilt, savgol_filter, butter, resample
from collections import deque
from sklearn.preprocessing import MinMaxScaler
import joblib
import keras.backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import json

"""
Default configs
"""
config_file = open("config.json", "r")
config = json.load(config_file)
interval_length = config["interval_length"]
step = config["step"]
stack = config["stack"]
scale_down = config["scale_down"]
datapoints = config["datapoints"]

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


class Res1D(tf.keras.layers.Layer):
    """
    Optional Residual layer
    """

    def __init__(self, filters, kernel_size):
        super(Res1D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.res = Sequential()

    def build(self, input_shape):
        self.res.add(
            Conv1D(filters=self.filters, kernel_size=self.kernel_size, padding='same', input_shape=input_shape[1:]))
        self.res.add(BatchNormalization(axis=1))
        self.res.add(Activation('relu'))
        self.res.add(Conv1D(filters=self.filters, kernel_size=self.kernel_size, padding='same'))
        self.res.add(BatchNormalization(axis=1))
        self.res.add(Activation('relu'))
        self.res.add(Conv1D(filters=self.filters, kernel_size=self.kernel_size, padding='same'))
        self.res.add(BatchNormalization(axis=1))

    def call(self, inputs, training=None):
        x = self.res(inputs, training=training)
        x += inputs
        return tf.nn.relu(x)

    def get_config(self):
        return {"filters": self.filters, "kernel_size": self.kernel_size}


"""
ML model
"""
model = Sequential()
model.add(
    Conv1D(input_shape=(datapoints, stack), filters=stack * 2, kernel_size=datapoints // 25, strides=1, padding='same'))
model.add(BatchNormalization(axis=1))
model.add(MaxPooling1D())
model.add(Conv1D(filters=stack * 4, kernel_size=datapoints // 20, strides=1, padding='same'))
model.add(BatchNormalization(axis=1))
model.add(MaxPooling1D())
model.add(Conv1D(filters=stack * 8, kernel_size=datapoints // 20, strides=1, padding='same'))
model.add(BatchNormalization(axis=1))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(
    Dense(units=datapoints, kernel_regularizer='l2', activity_regularizer='l2', kernel_initializer='glorot_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(
    Dense(units=datapoints * 2, kernel_regularizer='l2', activity_regularizer='l2', kernel_initializer='glorot_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(interval_length, use_bias=False, kernel_initializer='glorot_normal'))
model.add(Activation('sigmoid'))

model.summary()


def distance(y_true, y_labels):
    return K.mean(K.abs(K.argmax(y_true) - K.argmax(y_labels)))


def train(model_file, epochs, batch_size, learning_rate, x_train, y_train, x_test, y_test):
    """
    Initializing the model and training
    """
    optim = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optim, loss='categorical_crossentropy',
                  metrics=['categorical_accuracy', 'top_k_categorical_accuracy', distance])
    vd = ModelCheckpoint('val_distance.h5', monitor='val_distance', mode='min', verbose=1,
                         save_best_only=True)
    vc = ModelCheckpoint('val_cat_acc.h5', monitor='val_categorical_accuracy', mode='max', verbose=1,
                         save_best_only=True)
    vk = ModelCheckpoint('val_top_k.h5', monitor='val_top_k_categorical_accuracy', mode='max', verbose=1,
                         save_best_only=True)
    vl = ModelCheckpoint('val_loss.h5', monitor='val_loss', mode='min', verbose=1,
                         save_best_only=True)
    reducelr = ReduceLROnPlateau()
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2,
                        validation_data=(x_test, y_test),
                        callbacks=[vd, vc, vk, vl, reducelr])

    plt.plot(history.history['top_k_categorical_accuracy'])
    plt.plot(history.history['val_top_k_categorical_accuracy'])
    plt.ylabel('top k accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    """
    Visualizing results
    """
    for i in range(10):
        plt.plot(x_test[i, :, -1])
        sig = model.predict(x_test[i][np.newaxis, :, :])[0]
        sig = np.sum(sig.reshape((-1, scale_down)), axis=1) / scale_down
        plt.plot(sig)
        plt.show()

    model.save(model_file)


if __name__ == '__main__':
    model_file = 'model_new.h5'

    epochs = 100
    batch_size = 64
    learning_rate = 2e-5
    x_train = np.load("x_train.npy")
    y_train = np.load("y_train.npy")
    x_test = np.load("x_test.npy")
    y_test = np.load("y_test.npy")

    """
    Optional data visualizer
    """
    # for i in range(200):
    #     temp = x_train[i, :, 0]
    #     for j in range(1, stack):
    #         temp = np.append(temp, x_train[i, :, j][datapoints//(interval_length//step):])
    #     plt.plot(temp)
    #     sig = y_train[i, :]
    #     sum = np.sum(sig)
    #     sig = np.sum(sig.reshape((-1, scale_down)), axis=1) / scale_down * sum
    #     ls = np.asarray([0] * (datapoints//(interval_length//step)) * (stack - 1))
    #     sig = np.append(ls, sig)
    #     plt.plot(sig)
    #     plt.show(block=False)
    #     plt.pause(0.5)
    #     plt.close()

    train(model_file, epochs, batch_size, learning_rate, x_train, y_train, x_test, y_test)


