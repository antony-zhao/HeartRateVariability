import os
import scipy
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, \
    MaxPooling1D, BatchNormalization, TimeDistributed, Reshape, \
    Activation, LayerNormalization, Input, GRU, Bidirectional, Concatenate, \
    MultiHeadAttention, LSTM, Permute, GlobalAveragePooling1D, Embedding
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from config import window_size, stack, scale_down, datapoints, animal
import pandas as pd
from dataset import bandpass_filter, process_sample, highpass_filter, process_segment
from config import *


def weighted_crossentropy(weight):
    def loss(labels, logits):
        return tf.nn.weighted_cross_entropy_with_logits(labels, logits, pos_weight=weight)

    return loss


embedding_dim = 256
num_heads = 8
dropout = 0.5


def attention_layer(x):
    x_skip = x
    x_norm = LayerNormalization()(x)
    x = MultiHeadAttention(num_heads, embedding_dim // num_heads, dropout=dropout)(x_norm, x_norm)
    x = x_skip = Dense(embedding_dim)(x) + x_skip
    x_norm = LayerNormalization()(x)
    x_expand = Dense(units=embedding_dim * 4, activation='relu')(x_norm)
    x_expand = Dropout(dropout)(x_expand)
    x = Dense(units=embedding_dim)(x_expand)
    x = Dropout(dropout)(x) + x_skip
    return x


# positions = Input((stack,))
x = inputs = Input((stack * window_size, 6))
x = Conv1D(filters=16, kernel_size=31, padding='same', strides=5, activation='relu')(x)
x = BatchNormalization()(x)
x = Conv1D(filters=32, kernel_size=25, padding='same', strides=5, activation='relu')(x)
x = BatchNormalization()(x)
x = Conv1D(filters=64, kernel_size=13, padding='same', strides=5, activation='relu')(x)
x = BatchNormalization()(x)
x = Conv1D(filters=128, kernel_size=7, padding='same', strides=4, activation='relu')(x)
x = BatchNormalization()(x)
x = Conv1D(filters=embedding_dim, kernel_size=5, padding='same', strides=1)(x)
# pos = Embedding(stack, embedding_dim // 2)(positions)
# x = Concatenate()([x, pos])
for _ in range(8):
    x = attention_layer(x)
x = LayerNormalization()(x)
x = TimeDistributed(Dense(window_size // 4))(x)
out = Flatten()(x)
# out = Reshape((stack, datapoints))(x)
model = Model(inputs, out)

model.summary()


def train(model_file, epochs, batch_size, learning_rate, x_train, y_train, x_test, y_test, plot=False,
          sample_weight=None):
    """
    Trains the model on the train set and evaluates its performance on the test set, then saves the model
     to the specified file, as well as some checkpoints that save the best performance version of the model.

    Args:
        model_file (str): The prefix of the file name of the model.
        epochs (int): The number of epochs to train the model for
        batch_size (int): Size of the minibatches
        learning_rate (float): The learning rate
        x_train (numpy.ndarray): The training data for the training set
        y_train (numpy.ndarray): The output data for the training set
        x_test (numpy.ndarray): The training data for the test set
        y_test (numpy.ndarray): The output data for the test set
        plot (bool): Optional parameter to specify whether to display
            some random samples drawn and the model output on them.
            (default is False)
    """
    global model
    optim = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optim,
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[keras.metrics.BinaryAccuracy(),
                           keras.metrics.AUC(curve='PR', from_logits=True, multi_label=True),
                           keras.metrics.AUC(from_logits=True, multi_label=True)])
    va = ModelCheckpoint(model_file + '_val_auc', monitor='val_auc', mode='max', verbose=1,
                         save_best_only=True, initial_value_threshold=0.15)
    vk = ModelCheckpoint(model_file + '_val_top_k', monitor='val_top_k_categorical_accuracy', mode='max',
                         verbose=1,
                         save_best_only=True, initial_value_threshold=0.9)
    vc = ModelCheckpoint(model_file + '_val_bin', monitor='val_binary_accuracy', mode='max', verbose=1,
                         save_best_only=True, initial_value_threshold=0.2)
    vp = ModelCheckpoint(model_file + '_val_cat', monitor='val_sparse_categorical_accuracy', mode='max', verbose=1,
                         save_best_only=True, initial_value_threshold=0.2)
    # vm = ModelCheckpoint(model_file + '_val_bce', monitor='val_binary_crossentropy', mode='max', verbose=1,
    #                      save_best_only=True)
    reducelr = ReduceLROnPlateau(patience=10)
    eary_stop = EarlyStopping(monitor='val_auc', patience=30)
    history = model.fit(x_train, y_train, batch_size=None, epochs=epochs, verbose=2, sample_weight=sample_weight,
                        validation_data=(x_test, y_test), callbacks=[va, reducelr], shuffle=True)

    def plot(i):
        plt.plot(x_test[i, :, 0])
        plt.show()

    for _ in range(20):
        i = np.random.randint(x_test.shape[0])
        y_pred = model.predict(x_test[i][np.newaxis, :, :])[0]
        y_pred = np.repeat(scipy.special.expit(y_pred), 4)
        plt.plot(y_pred.flatten())
        plot(i)

    model.save(model_file)
    del model
    keras.backend.clear_session()


def train_generator(model_file, epochs, batch_size, learning_rate, train_data, val_data, steps_per_epoch, val_steps, plot=False,
          sample_weight=None):
    """
    Trains the model on the train set and evaluates its performance on the test set, then saves the model
     to the specified file, as well as some checkpoints that save the best performance version of the model.

    Args:
        model_file (str): The prefix of the file name of the model.
        epochs (int): The number of epochs to train the model for
        batch_size (int): Size of the minibatches
        learning_rate (float): The learning rate
        x_train (numpy.ndarray): The training data for the training set
        y_train (numpy.ndarray): The output data for the training set
        x_test (numpy.ndarray): The training data for the test set
        y_test (numpy.ndarray): The output data for the test set
        plot (bool): Optional parameter to specify whether to display
            some random samples drawn and the model output on them.
            (default is False)
    """
    global model
    optim = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optim,
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[keras.metrics.BinaryAccuracy(),
                           keras.metrics.AUC(curve='PR', from_logits=True, multi_label=True),
                           keras.metrics.AUC(from_logits=True, multi_label=True)])
    va = ModelCheckpoint(model_file + '_val_auc', monitor='val_auc', mode='max', verbose=1,
                         save_best_only=True, initial_value_threshold=0.15)
    vk = ModelCheckpoint(model_file + '_val_top_k', monitor='val_top_k_categorical_accuracy', mode='max',
                         verbose=1,
                         save_best_only=True, initial_value_threshold=0.9)
    vc = ModelCheckpoint(model_file + '_val_bin', monitor='val_binary_accuracy', mode='max', verbose=1,
                         save_best_only=True, initial_value_threshold=0.2)
    vp = ModelCheckpoint(model_file + '_val_cat', monitor='val_sparse_categorical_accuracy', mode='max', verbose=1,
                         save_best_only=True, initial_value_threshold=0.2)
    # vm = ModelCheckpoint(model_file + '_val_bce', monitor='val_binary_crossentropy', mode='max', verbose=1,
    #                      save_best_only=True)
    reducelr = ReduceLROnPlateau(patience=10)
    eary_stop = EarlyStopping(monitor='val_auc', patience=30)
    history = model.fit(train_data, batch_size=None, epochs=epochs, verbose=2, sample_weight=sample_weight,
                        validation_data=val_data, callbacks=[va, reducelr], shuffle=True,
                        steps_per_epoch=steps_per_epoch, validation_steps=val_steps)

    def plot(i):
        plt.plot(x_test[i, :, 0])
        plt.show()

    for _ in range(20):
        i = np.random.randint(x_test.shape[0])
        y_pred = model.predict(x_test[i][np.newaxis, :, :])[0]
        y_pred = np.repeat(scipy.special.expit(y_pred), 4)
        plt.plot(y_pred.flatten())
        plot(i)

    model.save(model_file)
    del model
    keras.backend.clear_session()


if __name__ == '__main__':
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow.keras.models import load_model

    model_file = f'{animal}_model'


    # model = load_model(model_file + '_val_auc')

    def data_generator(X, y, batch_size=128, steps_per_epoch=500):
        shuffle = True
        X = process_segment(X)
        while True:
            if shuffle:
                shuffle = False  # This loop is used to run the generator indefinitely.
                random_inds = np.random.randint(0, len(X) - stack * window_size, batch_size * steps_per_epoch * 2)
                random_inds = random_inds.reshape(steps_per_epoch, batch_size * 2)
            else:
                for inds in random_inds:
                    data = []
                    labels = []
                    for ind in inds:
                        y_i = y[ind:ind + int(stack * window_size)]
                        count = np.count_nonzero(y_i)
                        if count < 4:
                            continue
                        y_i = np.array(y_i).reshape((stack * window_size // 4, 4))
                        y_i = np.max(y_i, axis=-1)

                        x_i = X[ind:ind + int(stack * window_size)]
                        x_i = process_sample(x_i)

                        data.append(x_i)
                        labels.append(y_i)

                    data = np.stack(data)
                    labels = np.stack(labels)
                    # labels = np.concatenate((labels, 1 - np.max(labels, axis=-1)[:, :, np.newaxis]), axis=-1)
                    # labels = np.argmax(labels, axis=-1)

                    # diff = np.max(data, axis=1) - np.min(data, axis=1)
                    # data = data / diff[:, np.newaxis, :]
                    yield data, labels
                shuffle = True


    epochs = 150
    batch_size = 128
    learning_rate = 1e-4
    steps_per_epoch = 2000

    # x_train = np.load(os.path.join('..', 'Training', f'{animal}_x_train.npy'))
    # y_train = np.load(
    #     os.path.join('..', 'Training', f'{animal}_y_train.npy'))

    # x_test = np.load(os.path.join('..', 'Training', f'{animal}_x_test.npy'))
    # y_test = np.load(os.path.join('..', 'Training', f'{animal}_y_test.npy'))

    df = pd.read_csv(os.path.join('..', 'Training', f'{animal}_train.txt'), header=None, sep=' ')
    x_train = df[0].to_numpy()
    y_train = df[1].to_numpy()
    df = pd.read_csv(os.path.join('..', 'Training', f'{animal}_val.txt'), header=None, sep=' ')
    x_test = df[0].to_numpy()
    y_test = df[1].to_numpy()
    x_test = x_test[:len(x_test) // 2]
    y_test = y_test[:len(y_test) // 2]

    # train(model_file, epochs, batch_size, learning_rate, x_train, y_train, x_test, y_test, True)
    train_generator(model_file, epochs, batch_size, learning_rate, data_generator(x_train, y_train),
                    data_generator(x_test, y_test), steps_per_epoch=steps_per_epoch, val_steps=steps_per_epoch // 10)
