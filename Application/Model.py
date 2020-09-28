from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D, \
    Activation, LeakyReLU, GRU, LSTM, BatchNormalization, AveragePooling1D, Input, \
    GaussianNoise, Reshape, Permute, Add
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

T = 0.1  # Sample Period
fs = 4000.0  # sample rate, Hz
cutoff = 200  # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
nyq = 0.5 * fs  # Nyquist Frequency
order = 4  # sin wave can be approx represented as quadratic
n = int(T * fs)  # total number of samples
b, a = butter(N=order, Wn=cutoff / nyq, btype='low', analog=False)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

interval_length = 400
step = interval_length // 2
stack = 8


class Res1D(tf.keras.Model):
    def __init__(self, filters, kernel_size):
        super(Res1D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.res = Sequential()

    def build(self, input_shape):
        self.res.add(Conv1D(filters=self.filters, kernel_size=self.kernel_size, padding='same', input_shape=input_shape[1:]))
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


model = Sequential()
model.add(Conv1D(input_shape=(interval_length, stack), filters=16, kernel_size=12, strides=1, padding='same'))
model.add(BatchNormalization(axis=1))
model.add(Res1D(16, 8))
model.add(MaxPooling1D())
model.add(Conv1D(filters=32, kernel_size=12, strides=2, padding='same'))  # stride 2 when interval_length = 400
model.add(BatchNormalization(axis=1))
model.add(Res1D(32, 8))
model.add(MaxPooling1D())
model.add(Conv1D(filters=64, kernel_size=12, strides=2, padding='same'))  # stride 2 when interval_length = 400
model.add(BatchNormalization(axis=1))
model.add(Res1D(64, 8))
model.add(MaxPooling1D())
model.add(Flatten())
# model.add(Permute((2, 1)))
# model.add(Reshape((stack, 800)))
model.add(Dense(units=interval_length // 2, kernel_regularizer='l2', activity_regularizer='l2',
                kernel_initializer='glorot_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(units=interval_length // 1, kernel_regularizer='l2', activity_regularizer='l2',
                kernel_initializer='glorot_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(interval_length, use_bias=False, kernel_initializer='glorot_normal'))
model.add(Activation('sigmoid'))

# model = Sequential()
# model.add(Conv1D(input_shape=(interval_length, stack), filters=16, kernel_size=12, strides=1, padding='same'))
# model.add(BatchNormalization(axis=1))
# model.add(Res1D(16, 8))
# model.add(MaxPooling1D())
# model.add(Conv1D(filters=48, kernel_size=12, strides=2, padding='same'))  # stride 2 when interval_length = 400
# model.add(BatchNormalization(axis=1))
# model.add(Res1D(48, 8))
# model.add(MaxPooling1D())
# model.add(Conv1D(filters=96, kernel_size=12, strides=2, padding='same'))  # stride 2 when interval_length = 400
# model.add(BatchNormalization(axis=1))
# model.add(Res1D(96, 8))
# model.add(MaxPooling1D())
# model.add(Flatten())
# # model.add(Permute((2, 1)))
# # model.add(Reshape((stack, 800)))
# model.add(Dense(units=interval_length // 2, kernel_regularizer='l2', activity_regularizer='l2',
#                 kernel_initializer='glorot_normal'))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
# model.add(Dense(units=interval_length // 2, kernel_regularizer='l2', activity_regularizer='l2',
#                 kernel_initializer='glorot_normal'))
# model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
# model.add(Dense(interval_length, use_bias=False, kernel_initializer='glorot_normal'))
# model.add(Activation('sigmoid'))


# model = Sequential()
# model.add(Conv1D(input_shape=(interval_length, stack), filters=16 * stack, kernel_size=12, strides=2, padding='same'))
# model.add(BatchNormalization(axis=1))
# model.add(MaxPooling1D())
# model.add(Conv1D(filters=32 * stack, kernel_size=16, strides=2, padding='same'))  # stride 2 when interval_length = 400
# model.add(BatchNormalization(axis=1))
# model.add(MaxPooling1D())
# # model.add(Flatten())
# model.add(Permute((2, 1)))
# model.add(Reshape((stack, 800)))
# model.add(GRU(units=interval_length // 2, kernel_regularizer='l2', activity_regularizer='l2', return_sequences=True,
#               kernel_initializer='glorot_normal'))
# # model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
# model.add(GRU(units=interval_length // 2, kernel_regularizer='l2', activity_regularizer='l2',
#               kernel_initializer='glorot_normal'))
# # model.add(Activation('relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
# model.add(Dense(interval_length, use_bias=False, kernel_initializer='glorot_normal'))
# model.add(Activation('sigmoid'))

model.summary()


def load_model(model_file):
    model.load_weights(model_file)


def distance(y_true, y_labels):
    return K.mean(K.abs(K.argmax(y_true) - K.argmax(y_labels)))


def train_model(epochs, model_file, samples=10000, batch_size=512, learning_rate=0.001):
    ecg1, s1 = [], []
    ecg2, s2 = [], []
    ecg3, s3 = [], []
    for x in open(os.path.join('..', 'Training', 'ecg1.txt')):
        ecg1.append(float(re.findall('([-0-9.]+)', x)[-1]))

    for x in open(os.path.join('..', 'Training', 'sig1.txt')):
        s1.append(float(re.findall('([-0-9.]+)', x)[-1]))

    for x in open(os.path.join('..', 'Training', 'ecg.txt')):
        ecg2.append(float(re.findall('([-0-9.]+)', x)[-1]))

    for x in open(os.path.join('..', 'Training', 'sig.txt')):
        s3.append(float(re.findall('([-0-9.]+)', x)[-1]))

    for x in open(os.path.join('..', 'Training', 'ecg6.txt')):
        ecg3.append(float(re.findall('([-0-9.]+)', x)[-1]))

    for x in open(os.path.join('..', 'Training', 'sig6.txt')):
        s3.append(float(re.findall('([-0-9.]+)', x)[-1]))

    x_train, y_train = random_sampling(ecg1, s1, samples, interval_length, step, stack)
    x_train_1, y_train_1 = random_sampling(ecg2, s2, samples // 4, interval_length, step, stack)
    x_train = np.append(x_train, x_train_1, axis=0)
    y_train = np.append(y_train, y_train_1, axis=0)
    x_train = np.append(x_train, -x_train, axis=0)
    y_train = np.append(y_train, y_train, axis=0)
    x_test, y_test = random_sampling(ecg3, s3, samples // 4, interval_length, step, stack)
    x_test = np.append(x_test, -x_test, axis=0)
    y_test = np.append(y_test, y_test, axis=0)
    # x_train, y_train = sequential_sampling(ecg1, s1, interval_length, step)
    del s1
    del ecg1
    del ecg2
    del s2
    del ecg3
    del s3

    # for i in range(100):
    #     ind = np.random.choice(x_train.shape[0])
    #     plt.plot(range(interval_length), x_train[ind, :, -1])
    #     plt.plot(range(interval_length), y_train[ind, :])
    #     plt.show(block=False)
    #     plt.pause(0.5)
    #     plt.close()

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
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data=(x_test, y_test),
              callbacks=[vd, vc, vk, vl])

    # model.evaluate(x_test, y_test, batch_size=1, verbose=2)
    #
    # model.save(model_file)
