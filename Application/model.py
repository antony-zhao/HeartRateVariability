import os
import scipy
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, \
    MaxPooling1D, LayerNormalization, TimeDistributed, Reshape, \
    Activation, LayerNormalization, Input, GRU, Bidirectional, \
    MultiHeadAttention, LSTM, Permute, GlobalAveragePooling1D
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from config import window_size, stack, scale_down, datapoints, animal


def weighted_crossentropy(weight):
    def loss(labels, logits):
        return tf.nn.weighted_cross_entropy_with_logits(labels, logits, pos_weight=weight)

    return loss


def attention_layer(x):
    x = MultiHeadAttention(8, embedding_dim, dropout=0.5, kernel_regularizer='l2',
                           activity_regularizer='l2')(x, x) + x
    x = LayerNormalization()(x)
    x_expand = Dense(units=embedding_dim * 4, activation='relu')(x)
    x_expand = Dropout(0.5)(x_expand)
    x = Dense(units=embedding_dim, activation='relu')(x_expand) + x
    x = Dropout(0.5)(x)
    x = LayerNormalization()(x)
    return x


embedding_dim = 128
x = inputs = Input((stack * 2, datapoints))
x = LSTM(embedding_dim, dropout=0.5, return_sequences=True, kernel_regularizer='l2',
         activity_regularizer='l2')(x)
x = LayerNormalization()(x)
for _ in range(4):
    x = attention_layer(x)
x = Conv1D(filters=embedding_dim, kernel_size=3, strides=2, padding='same', kernel_regularizer='l2',
           activity_regularizer='l2')(x)
x = TimeDistributed(Dense(window_size + 1))(x)
out = Reshape((stack, datapoints + 1))(x)
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
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  #weighted_crossentropy(weight=1),  # tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[keras.metrics.AUC(from_logits=True), keras.metrics.CategoricalAccuracy(),
                           keras.metrics.TopKCategoricalAccuracy(k=4)])  # ], keras.metrics.AUC(from_logits=True)]
    va = ModelCheckpoint(model_file + '_val_auc', monitor='val_auc', mode='max', verbose=1,
                         save_best_only=True, initial_value_threshold=0.95)
    vk = ModelCheckpoint(model_file + '_val_top_k', monitor='val_top_k_categorical_accuracy', mode='max',
                         verbose=1,
                         save_best_only=True, initial_value_threshold=0.9)
    vc = ModelCheckpoint(model_file + '_val_bin', monitor='val_binary_accuracy', mode='max', verbose=1,
                         save_best_only=True, initial_value_threshold=0.2)
    vp = ModelCheckpoint(model_file + '_val_cat', monitor='val_categorical_accuracy', mode='max', verbose=1,
                         save_best_only=True, initial_value_threshold=0.2)
    # vm = ModelCheckpoint(model_file + '_val_bce', monitor='val_binary_crossentropy', mode='max', verbose=1,
    #                      save_best_only=True)
    reducelr = ReduceLROnPlateau(patience=5)
    eary_stop = EarlyStopping(patience=8)
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, sample_weight=sample_weights,
                        validation_data=(x_test, y_test), callbacks=[va, vk, vp, reducelr, eary_stop], shuffle=True)

    def plot(i):
        plt.plot(x_test[i].flatten()[:3000])
        # plt.plot(-y_test[i].flatten())
        plt.show()

    for _ in range(20):
        i = np.random.randint(x_test.shape[0])
        y_pred = model.predict(x_test[i][np.newaxis, :, :])[0]
        y_pred = scipy.special.softmax(y_pred, axis=-1)
        y_pred_masked = y_pred[:, :-1] * np.repeat(1 - y_pred[:, -1], window_size).reshape(y_pred.shape[0], window_size)
        plt.plot(y_pred.flatten())
        plt.plot(y_pred_masked.flatten())
        plot(i)

    model.save(model_file)
    del model
    keras.backend.clear_session()


if __name__ == '__main__':
    from sklearn.utils.class_weight import compute_class_weight

    model_file = f'{animal}_model'

    epochs = 150
    batch_size = 64
    learning_rate = 2e-4
    x_train = np.load(os.path.join('..', 'Training', f'{animal}_x_train.npy'))
    y_train = np.load(
        os.path.join('..', 'Training', f'{animal}_y_train.npy'))  #.reshape(-1, stack * 2, datapoints // 2)

    y_one_hot = y_train.argmax(axis=-1).flatten()
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_one_hot), y=y_one_hot)
    # class_weights = dict(enumerate(class_weights))
    sample_weights = y_train @ class_weights
    # inds = np.random.randint(x_train.shape[0], size=x_train.shape[0] * 3 // 4)
    # x_train = x_train[inds]
    # y_train = y_train[inds]
    x_test = np.load(os.path.join('..', 'Training', f'{animal}_x_test.npy'))
    y_test = np.load(os.path.join('..', 'Training', f'{animal}_y_test.npy'))  #.reshape(-1, stack * 2, datapoints // 2)

    train(model_file, epochs, batch_size, learning_rate, x_train, y_train, x_test, y_test, True, sample_weights)
