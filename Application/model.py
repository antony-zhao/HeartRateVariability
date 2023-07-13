import os

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D, \
    Activation, BatchNormalization, Input, GRU, Bidirectional, MultiHeadAttention
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from config import interval_length, stack, scale_down, datapoints, animal
import tensorflow.keras.backend as tfb
from tensorflow.keras.utils import get_custom_objects

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

POS_WEIGHT = 50  # multiplier for positive targets, needs to be tuned


def weighted_binary_crossentropy(target, output):
    """
    Weighted binary crossentropy between an output tensor
    and a target tensor. POS_WEIGHT is used as a multiplier
    for the positive targets.

    Combination of the following functions:
    * keras.losses.binary_crossentropy
    * keras.backend.tensorflow_backend.binary_crossentropy
    * tf.nn.weighted_cross_entropy_with_logits
    """
    # transform back to logits
    _epsilon = tfb.epsilon()
    output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
    output = tf.math.log(output / (1 - output))
    # compute weighted loss
    target = tf.cast(target, tf.float32)
    loss = tf.nn.weighted_cross_entropy_with_logits(labels=target,
                                                    logits=output,
                                                    pos_weight=POS_WEIGHT)
    return tf.reduce_mean(loss, axis=-1)


model = Sequential()  # The main model used for detecting R peaks.
model.add(
    Conv1D(input_shape=(datapoints, stack * 2), filters=stack * 4, kernel_size=datapoints // 25, strides=2,
           padding='same', activation='relu'))
model.add(BatchNormalization(axis=1))
model.add(MaxPooling1D())
model.add(Conv1D(filters=stack * 4, kernel_size=datapoints // 20, strides=2, padding='same', kernel_regularizer='l2',
                 activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(strides=2))
model.add(Conv1D(filters=stack * 8, kernel_size=datapoints // 20, strides=1, padding='same', kernel_regularizer='l2',
                 activation='relu'))
model.add(BatchNormalization(axis=1))
model.add(MaxPooling1D(strides=2))
model.add(
    GRU(units=datapoints, kernel_regularizer='l2', activity_regularizer='l2', kernel_initializer='glorot_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(interval_length))
model.add(Activation('sigmoid'))

model.summary()


def distance(y_true, y_labels):
    """Distance metric for training, displays the average
    absolute distance between the true and predicted peak."""
    return K.mean(K.abs(K.argmax(y_true, axis=0) - K.argmax(y_labels, axis=0)))


def magnitude(y_true, y_labels):
    """Metric for what the magnitudes of the labels are, as having smaller ones
    can make it harder for model_prediction to work"""
    return K.mean(K.max(y_labels * y_true, axis=0))


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
    get_custom_objects().update({"weighted_binary_crossentropy": weighted_binary_crossentropy})
    model.compile(optimizer=optim, loss=weighted_binary_crossentropy,
                  metrics=['categorical_accuracy', 'top_k_categorical_accuracy', magnitude])
    # vd = ModelCheckpoint(model_file + '_val_distance', monitor='val_distance', mode='min', verbose=1,
    #                      save_best_only=True)
    # vc = ModelCheckpoint(model_file + '_val_cat_acc', monitor='val_categorical_accuracy', mode='max', verbose=1,
    #                      save_best_only=True)
    vk = ModelCheckpoint(model_file + '_val_top_k', monitor='val_top_k_categorical_accuracy', mode='max', verbose=1,
                         save_best_only=True)
    vl = ModelCheckpoint(model_file + '_val_mag', monitor='val_magnitude', mode='max', verbose=1,
                         save_best_only=True)
    reducelr = ReduceLROnPlateau()
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2,
                        validation_data=(x_test, y_test), callbacks=[vk, reducelr])

    model_top_k = keras.models.load_model('model_val_top_k')
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

        for i in range(5):
            plt.plot(x_test[i, :, -2], label='filtered')
            sig = model.predict(x_test[i][np.newaxis, :, :])[0]
            sig = np.sum(sig.reshape((-1, scale_down)), axis=1) / scale_down
            plt.plot(sig, label='prediction')
            sig = model_top_k.predict(x_test[i][np.newaxis, :, :])[0]
            sig = np.sum(sig.reshape((-1, scale_down)), axis=1) / scale_down
            plt.plot(sig, label='top_k_prediction')
            sig = y_test[i]
            sig = np.sum(sig.reshape((-1, scale_down)), axis=1) / scale_down
            plt.plot(sig, label='truth')
            plt.legend()
            plt.show()

        # for i in range(5):
        #     plt.plot(x_train[i, :, -2], label='filtered')
        #     sig = model.predict(x_test[i][np.newaxis, :, :])[0]
        #     sig = np.sum(sig.reshape((-1, scale_down)), axis=1) / scale_down
        #     plt.plot(sig, label='prediction')
        #     sig = model_top_k.predict(x_test[i][np.newaxis, :, :])[0]
        #     sig = np.sum(sig.reshape((-1, scale_down)), axis=1) / scale_down
        #     plt.plot(sig, label='top_k_prediction')
        #     sig = y_train[i]
        #     sig = np.sum(sig.reshape((-1, scale_down)), axis=1) / scale_down
        #     plt.plot(sig, label='truth')
        #     plt.legend()
        #     plt.show()
        #
        #     plt.plot(x_train[i, :, 3], label='ecg')
        #     sig = model.predict(x_test[i][np.newaxis, :, :])[0]
        #     sig = np.sum(sig.reshape((-1, scale_down)), axis=1) / scale_down
        #     plt.plot(sig, label='prediction')
        #     sig = model_top_k.predict(x_test[i][np.newaxis, :, :])[0]
        #     sig = np.sum(sig.reshape((-1, scale_down)), axis=1) / scale_down
        #     plt.plot(sig, label='top_k_prediction')
        #     sig = y_train[i]
        #     sig = np.sum(sig.reshape((-1, scale_down)), axis=1) / scale_down
        #     plt.plot(sig, label='truth')
        #     plt.legend()
        #     plt.show()

    model.save(model_file)
    del model
    tf.keras.backend.clear_session()


if __name__ == '__main__':
    model_file = 'model'

    epochs = 10
    batch_size = 64
    learning_rate = 1e-4
    x_train = np.load(os.path.join('..', 'Training', f'{animal}_x_train.npy'))
    y_train = np.load(os.path.join('..', 'Training', f'{animal}_y_train.npy'))
    x_test = np.load(os.path.join('..', 'Training', f'{animal}_x_test.npy'))
    y_test = np.load(os.path.join('..', 'Training', f'{animal}_y_test.npy'))

    train(model_file, epochs, batch_size, learning_rate, x_train, y_train, x_test, y_test, True)

    # for i in range(5):
    #     plt.plot(x_test[i, :, -2], label='filtered')
    #     # sig = model.predict(x_test[i][np.newaxis, :, :])[0]
    #     # sig = np.sum(sig.reshape((-1, scale_down)), axis=1) / scale_down
    #     # plt.plot(sig, label='prediction')
    #     # sig = model_top_k.predict(x_test[i][np.newaxis, :, :])[0]
    #     # sig = np.sum(sig.reshape((-1, scale_down)), axis=1) / scale_down
    #     # plt.plot(sig, label='top_k_prediction')
    #     sig = y_test[i]
    #     sig = np.sum(sig.reshape((-1, scale_down)), axis=1) / scale_down
    #     plt.plot(sig, label='truth')
    #     plt.legend()
    #     plt.show()
    #
    #     plt.plot(x_test[i, :, 3], label='ecg')
    #     # sig = model.predict(x_test[i][np.newaxis, :, :])[0]
    #     # sig = np.sum(sig.reshape((-1, scale_down)), axis=1) / scale_down
    #     # plt.plot(sig, label='prediction')
    #     # sig = model_top_k.predict(x_test[i][np.newaxis, :, :])[0]
    #     # sig = np.sum(sig.reshape((-1, scale_down)), axis=1) / scale_down
    #     # plt.plot(sig, label='top_k_prediction')
    #     sig = y_test[i]
    #     sig = np.sum(sig.reshape((-1, scale_down)), axis=1) / scale_down
    #     plt.plot(sig, label='truth')
    #     plt.legend()
    #     plt.show()

    # for i in range(5):
    #     plt.plot(x_train[i, :, -2], label='filtered')
    #     # sig = model.predict(x_test[i][np.newaxis, :, :])[0]
    #     # sig = np.sum(sig.reshape((-1, scale_down)), axis=1) / scale_down
    #     # plt.plot(sig, label='prediction')
    #     # sig = model_top_k.predict(x_test[i][np.newaxis, :, :])[0]
    #     # sig = np.sum(sig.reshape((-1, scale_down)), axis=1) / scale_down
    #     # plt.plot(sig, label='top_k_prediction')
    #     sig = y_train[i]
    #     sig = np.sum(sig.reshape((-1, scale_down)), axis=1) / scale_down
    #     plt.plot(sig, label='truth')
    #     plt.legend()
    #     plt.show()
    #
    #     plt.plot(x_train[i, :, 3], label='ecg')
    #     # sig = model.predict(x_test[i][np.newaxis, :, :])[0]
    #     # sig = np.sum(sig.reshape((-1, scale_down)), axis=1) / scale_down
    #     # plt.plot(sig, label='prediction')
    #     # sig = model_top_k.predict(x_test[i][np.newaxis, :, :])[0]
    #     # sig = np.sum(sig.reshape((-1, scale_down)), axis=1) / scale_down
    #     # plt.plot(sig, label='top_k_prediction')
    #     sig = y_train[i]
    #     sig = np.sum(sig.reshape((-1, scale_down)), axis=1) / scale_down
    #     plt.plot(sig, label='truth')
    #     plt.legend()
    #     plt.show()
