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
from dataset import bandpass_filter, process_ecg
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


# positions = K.variable(range(stack))
# positions = K.repeat_elements(positions, rep=64, axis=0)
# cons_input = Input(tensor=positions)
x = inputs = Input((stack * window_size, 1))
x = Conv1D(filters=8, kernel_size=31, padding='same', strides=5, activation='relu')(x)
x = BatchNormalization()(x)
x = Conv1D(filters=16, kernel_size=25, padding='same', strides=5, activation='relu')(x)
x = BatchNormalization()(x)
x = Conv1D(filters=32, kernel_size=13, padding='same', strides=5, activation='relu')(x)
x = BatchNormalization()(x)
x = Conv1D(filters=64, kernel_size=7, padding='same', strides=4, activation='relu')(x)
x = BatchNormalization()(x)
x = Conv1D(filters=embedding_dim, kernel_size=5, padding='same', strides=1)(x)
# pos = Embedding(stack, embedding_dim // 2)(cons_input)
# x = Concatenate()([x, pos])
for _ in range(8):
    x = attention_layer(x)
x = LayerNormalization()(x)
x = TimeDistributed(Dense(window_size // 4))(x)
out = Flatten()(x)
# out = Reshape((stack, datapoints))(x)
model = Model([inputs], out)

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


if __name__ == '__main__':
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow.keras.models import load_model

    model_file = f'{animal}_model'


    # model = load_model(model_file + '_val_auc')

    def data_generator(X, y, batch_size=128, steps_per_epoch=500):
        shuffle = True
        filtered_X = bandpass_filter(X, order, low_cutoff, high_cutoff, nyq)
        while True:
            if shuffle:
                shuffle = False  # This loop is used to run the generator indefinitely.
                random_inds = np.random.randint(0, len(X) - stack * window_size, batch_size * 2 * steps_per_epoch)
                random_inds = random_inds.reshape(steps_per_epoch, batch_size * 2)
            else:
                for inds in random_inds:
                    data = []
                    labels = []
                    for ind in inds:
                        y_i = y[ind:ind + int(stack * window_size)]
                        if np.count_nonzero(y_i) < 3:
                            continue
                        y_i = np.array(y_i).reshape((stack, datapoints // 4, 4))
                        y_i = np.max(y_i, axis=-1)

                        y_i = np.concatenate((y_i, 1 - np.max(y_i, axis=1).reshape(-1, 1)), axis=1)
                        y_i = np.argmax(y_i, axis=-1)

                        x_i = np.stack((X[ind:ind + int(stack * window_size)],
                                        filtered_X[ind:ind + int(stack * window_size)]), axis=1)

                        data.append(x_i)
                        labels.append(y_i)

                    data = np.stack(data)
                    labels = np.stack(labels)
                    # labels = np.concatenate((labels, 1 - np.max(labels, axis=-1)[:, :, np.newaxis]), axis=-1)
                    # labels = np.argmax(labels, axis=-1)

                    diff = np.max(data, axis=1) - np.min(data, axis=1)
                    data = data / diff[:, np.newaxis, :]
                    yield np.stack(data), np.stack(labels)
                shuffle = True


    epochs = 150
    batch_size = 128
    learning_rate = 1e-4
    x_train = np.load(os.path.join('..', 'Training', f'{animal}_x_train.npy'))
    y_train = np.load(
        os.path.join('..', 'Training', f'{animal}_y_train.npy'))  #.reshape(-1, stack * 2, datapoints // 2)

    # y_one_hot = y_train.argmax(axis=-1).flatten()
    # class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_one_hot), y=y_one_hot)
    # class_weights = dict(enumerate(class_weights))
    # sample_weights = y_train @ class_weights
    # inds = np.random.randint(x_train.shape[0], size=x_train.shape[0] * 3 // 4)
    # x_train = x_train[inds]
    # y_train = y_train[inds]
    x_test = np.load(os.path.join('..', 'Training', f'{animal}_x_test.npy'))
    y_test = np.load(os.path.join('..', 'Training', f'{animal}_y_test.npy'))  #.reshape(-1, stack * 2, datapoints // 2)

    train(model_file, epochs, batch_size, learning_rate, x_train, y_train, x_test, y_test, True)
