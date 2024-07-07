import os

import scipy
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D, BatchNormalization, \
    Activation, LayerNormalization, Input, GRU, Bidirectional, MultiHeadAttention, LSTM, Permute, GlobalAveragePooling1D
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from config import window_size, stack, scale_down, datapoints, animal
import tensorflow.keras.backend as tfb
from tensorflow.keras.utils import get_custom_objects

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)


def build_cnn(filters, kernel):
    return keras.Sequential([
        keras.layers.Conv1D(filters, kernel[0], padding='same', activation='relu', kernel_regularizer='l2',
                            activity_regularizer='l2'),
        keras.layers.LayerNormalization(),
        keras.layers.Conv1D(filters, kernel[1], padding='same', activation='relu', kernel_regularizer='l2',
                            activity_regularizer='l2'),
        keras.layers.LayerNormalization(),
        keras.layers.Conv1D(filters, kernel[2], padding='same', activation='relu', kernel_regularizer='l2',
                            activity_regularizer='l2'),
        keras.layers.LayerNormalization(),
    ])


inputs = Input((30, 100))
x = Conv1D(filters=100, kernel_size=1,  # data_format='channels_first',
           padding='same', kernel_regularizer='l2', activity_regularizer='l2',
           activation='relu')(inputs)
x = LayerNormalization()(x)
x = MultiHeadAttention(8, 100, dropout=0.5)(x, x) + x
x = LayerNormalization()(x)
x = Dense(units=100, activation='relu')(x) + x
x = Dropout(0.5)(x)
x = LayerNormalization()(x)
x = Flatten()(x)
out = Dense(1)(x)
model = Model(inputs, out)

model.summary()

def distance(y_true, y_labels):
    """Distance metric for training, displays the average
    absolute distance between the true and predicted peak."""
    return K.mean(K.abs(K.argmax(y_true, axis=0) - K.argmax(y_labels, axis=0)))


def magnitude(y_true, y_labels):
    """Metric for what the magnitudes of the labels are, as having smaller ones
    can make it harder for model_prediction to work"""
    x = K.mean(K.max(y_labels, axis=1) * K.max(y_true, axis=1))
    return x


def train(model_file, epochs, batch_size, learning_rate, x_train, y_train, x_test, y_test, plot=False):
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
    get_custom_objects().update({'magnitude': magnitude, 'distance': distance})
    model.compile(optimizer=optim, loss=keras.losses.BinaryCrossentropy(from_logits=True),  #from_logits=True
                  metrics=['categorical_accuracy', 'top_k_categorical_accuracy',
                           keras.metrics.BinaryAccuracy(),
                           tf.keras.metrics.AUC(from_logits=True, multi_label=True)])
    va = ModelCheckpoint(model_file + '_val_auc', monitor='val_auc', mode='max', verbose=1,
                         save_best_only=True, intial_value_threshold=0.8)
    vk = ModelCheckpoint(model_file + '_val_top_k', monitor='val_top_k_categorical_accuracy', mode='max', verbose=1,
                         save_best_only=True, intial_value_threshold=0.5)
    # vm = ModelCheckpoint(model_file + '_val_mag', monitor='val_magnitude', mode='max', verbose=1,
    #                      save_best_only=True, intial_value_threshold=0.2)
    vp = ModelCheckpoint(model_file + '_val_cat', monitor='val_categorical_accuracy', mode='max', verbose=1,
                         save_best_only=True, intial_value_threshold=0.2)
    vm = ModelCheckpoint(model_file + '_val_bce', monitor='val_binary_accuracy', mode='max', verbose=1,
                         save_best_only=True)
    reducelr = ReduceLROnPlateau(patience=5)
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2,
                        validation_data=(x_test, y_test), callbacks=[reducelr, vm], shuffle=True)

    model_top_k = keras.models.load_model(f'{animal}_model_val_top_k')
    if plot:  # Optional plotting to visualize and verify the model.
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

    model.save(model_file)
    del model
    tf.keras.backend.clear_session()


if __name__ == '__main__':
    model_file = f'{animal}_model_mask'

    epochs = 60
    batch_size = 64
    learning_rate = 1e-4
    x_train = np.load(os.path.join('..', 'Training', f'{animal}_x_train_mask.npy'))
    y_train = np.load(os.path.join('..', 'Training', f'{animal}_y_train_mask.npy'))
    x_test = np.load(os.path.join('..', 'Training', f'{animal}_x_test_mask.npy'))
    y_test = np.load(os.path.join('..', 'Training', f'{animal}_y_test_mask.npy'))

    train(model_file, epochs, batch_size, learning_rate, x_train, y_train, x_test, y_test, True)
