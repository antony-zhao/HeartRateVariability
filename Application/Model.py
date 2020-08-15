from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D,\
                                    Activation, LeakyReLU, GRU, LSTM, BatchNormalization, AveragePooling1D, Input, \
                                    GaussianNoise, Reshape, Permute
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import tensorflow as tf
from Methods import *
import os
import keras.backend as K
from scipy.signal import filtfilt, butter
from scipy import signal
from Parameters import interval_length
from keras.callbacks import EarlyStopping, ModelCheckpoint

T = 0.1          # Sample Period
fs = 4000.0      # sample rate, Hz
cutoff = 200      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
nyq = 0.5 * fs   # Nyquist Frequency
order = 4        # sin wave can be approx represented as quadratic
n = int(T * fs)  # total number of samples
b, a = butter(N=order, Wn=cutoff/nyq, btype='low', analog=False)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

interval_length = 400
step = interval_length // 2
stack = 16

# model = Sequential()
# model.add(Conv1D(input_shape=(interval_length, stack), filters=16*stack, kernel_size=12, strides=2, padding='same'))
# model.add(BatchNormalization(axis=1))
# model.add(MaxPooling1D())
# model.add(Conv1D(filters=32*stack, kernel_size=16, strides=2, padding='same'))  # stride 2 when interval_length = 400
# model.add(BatchNormalization(axis=1))
# model.add(MaxPooling1D())
# # model.add(Flatten())
# model.add(Permute((2, 1)))
# model.add(Reshape((stack, 800)))
# model.add(GRU(units=interval_length//2, kernel_regularizer='l2', activity_regularizer='l2', return_sequences=True, kernel_initializer='glorot_normal'))
# # model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
# model.add(GRU(units=interval_length//2, kernel_regularizer='l2', activity_regularizer='l2', kernel_initializer='glorot_normal'))
# # model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
# model.add(Dense(interval_length, use_bias=False, kernel_initializer='glorot_normal'))
# model.add(Activation('sigmoid'))


model = Sequential()
model.add(Conv1D(input_shape=(interval_length, stack), filters=8*stack, kernel_size=16, strides=2, padding='same'))
model.add(BatchNormalization(axis=1))
model.add(MaxPooling1D())
model.add(Conv1D(filters=16*stack, kernel_size=16, strides=2, padding='same'))  # stride 2 when interval_length = 400
model.add(BatchNormalization(axis=1))
model.add(MaxPooling1D())
# model.add(Flatten())
model.add(Permute((2, 1)))
model.add(Reshape((stack, 400)))
model.add(GRU(units=interval_length//2, kernel_regularizer='l2', activity_regularizer='l2', return_sequences=True, kernel_initializer='glorot_normal'))
# model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(GRU(units=interval_length//2, kernel_regularizer='l2', activity_regularizer='l2', kernel_initializer='glorot_normal'))
# model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(interval_length, use_bias=False, kernel_initializer='glorot_normal'))
model.add(Activation('sigmoid'))

model.summary()


def load_model(model_file):
    model.load_weights(model_file)


def distance(y_true, y_labels):
    return K.mean(K.abs(K.argmax(y_true) - K.argmax(y_labels)))


def train_model(epochs, model_file, samples=10000, batch_size=512, learning_rate=0.001):
    ecg1, s1 = [], []
    ecg2, s2 = [], []
    for x in open(os.path.join('..', 'Training', 'ecg5.txt')):
        ecg1.append(float(re.findall('([-0-9.]+)', x)[-1]))

    for x in open(os.path.join('..', 'Training', 'sig5.txt')):
        s1.append(float(re.findall('([-0-9.]+)', x)[-1]))

    for x in open(os.path.join('..', 'Training', 'ecg6.txt')):
        ecg2.append(float(re.findall('([-0-9.]+)', x)[-1]))

    for x in open(os.path.join('..', 'Training', 'sig6.txt')):
        s2.append(float(re.findall('([-0-9.]+)', x)[-1]))

    x_train, y_train = random_sampling(ecg1, s1, samples, interval_length, step, stack)
    x_train = np.append(x_train, -x_train, axis=0)
    y_train = np.append(y_train, y_train, axis=0)
    x_test, y_test = random_sampling(ecg2, s2, samples//4, interval_length, step, stack)
    x_test = np.append(x_test, -x_test, axis=0)
    y_test = np.append(y_test, y_test, axis=0)
    # x_train, y_train = sequential_sampling(ecg1, s1, interval_length, step)
    del s1
    del ecg1
    del ecg2
    del s2

    # for i in range(100):
    #     ind = np.random.choice(x_train.shape[0])
    #     plt.plot(range(interval_length), x_train[ind, :, -1])
    #     plt.plot(range(interval_length), y_train[ind, :])
    #     plt.show(block=False)
    #     plt.pause(0.5)
    #     plt.close()

    optim = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['categorical_accuracy', 'top_k_categorical_accuracy', distance])
    vd = ModelCheckpoint('val_distance.h5', monitor='val_distance', mode='min', verbose=1,
                         save_best_only=True)
    vc = ModelCheckpoint('val_cat_acc.h5', monitor='val_categorical_accuracy', mode='max', verbose=1,
                         save_best_only=True)
    vk = ModelCheckpoint('val_top_k.h5', monitor='val_top_k_categorical_accuracy', mode='max', verbose=1,
                         save_best_only=True)
    vl = ModelCheckpoint('val_loss.h5', monitor='val_loss', mode='min', verbose=1,
                         save_best_only=True)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(x_test, y_test),
              callbacks=[vd, vc, vk, vl])

    # model.evaluate(x_test, y_test, batch_size=1, verbose=2)
    #
    # model.save(model_file)
