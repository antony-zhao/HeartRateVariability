import os

import scipy
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, MaxPooling1D, MultiHeadAttention, \
    LayerNormalization, \
    Activation, BatchNormalization, Input, GRU, Bidirectional, Attention, LSTM, Permute, Concatenate, TimeDistributed, \
    GlobalAveragePooling1D
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

n_embed = 128
n_head = 8
dropout = 0.5
n_layer = 4


class FeedForward(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Dense(n_embed * 4,
                           activation='relu')  # Conv1D(filters=n_embed * 4, kernel_size=1, activation='relu', kernel_regularizer='l2', activity_regularizer='l2')
        self.conv2 = Dense(
            n_embed)  # Conv1D(filters=n_embed, kernel_size=1, kernel_regularizer='l2', activity_regularizer='l2')
        self.dropout = Dropout(dropout)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        if training:
            x = self.dropout(x, training=training)
        x = self.conv2(x)
        if training:
            x = self.dropout(x, training=training)
        return x


class Head(Model):
    def __init__(self):
        super().__init__()
        self.attention = MultiHeadAttention(num_heads=n_head, key_dim=n_embed, dropout=dropout, kernel_regularizer='l2',
                                            activity_regularizer='l2')
        self.dense = Dense(n_embed)
        self.dropout = Dropout(0.5)

    def call(self, inputs, training=False):
        x = self.attention(inputs, inputs, training=training)
        x = self.dense(x)
        if training:
            x = self.dropout(x, training=training)
        return x


class Block(Model):
    def __init__(self):
        super().__init__()
        self.head = Head()
        self.feed_forward = FeedForward()
        self.ln1 = LayerNormalization()
        self.ln2 = LayerNormalization()

    def call(self, inputs, training=False):
        x = inputs
        x = x + self.head.call(self.ln1(x), training)
        x = x + self.feed_forward.call(self.ln2(x), training)
        return x


def positional_encoding(length, depth):
    depth = depth / 2

    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)

    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def call(self, x):
        length = tf.shape(x)[1]
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


inputs = Input((datapoints * stack, 4))
conv1 = Conv1D(filters=stack * 4, kernel_size=31, strides=2,
               padding='same', kernel_regularizer='l2', activity_regularizer='l2',
               activation='relu')(inputs)
maxpool1 = MaxPooling1D(2)(conv1)
conv2 = Conv1D(filters=stack * 8, kernel_size=17, strides=2,
               padding='same', kernel_regularizer='l2', activity_regularizer='l2',
               activation='relu')(maxpool1)
maxpool2 = MaxPooling1D(2)(conv2)
conv3 = Conv1D(filters=n_embed, kernel_size=9, strides=2,
               padding='same', kernel_regularizer='l2', activity_regularizer='l2',
               activation='relu')(maxpool2)
maxpool3 = MaxPooling1D(2)(conv3)
gpt1 = Block().call(maxpool3)
gpt2 = Block().call(gpt1)
gpt3 = Block().call(gpt2)
# gpt4 = Block().call(gpt3)
bi_lstm1 = keras.layers.GlobalAveragePooling1D()(
    gpt3)  #Bidirectional(LSTM(n_embed, return_sequences=False, dropout=0.5))(gpt2)
dense_out = Dense(window_size)(bi_lstm1)
model = Model(inputs=inputs, outputs=dense_out)
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


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 10


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
    model.compile(optimizer=optim, loss=keras.losses.CategoricalCrossentropy(from_logits=True),  #from_logits=True
                  metrics=['categorical_accuracy', 'top_k_categorical_accuracy', magnitude,
                           # keras.metrics.BinaryAccuracy(),
                           tf.keras.metrics.AUC(from_logits=True, multi_label=True)])
    # va = ModelCheckpoint(model_file + '_val_auc', monitor='val_auc', mode='max', verbose=1,
    #                      save_best_only=True, intial_value_threshold=0.8)
    # vk = ModelCheckpoint(model_file + '_val_top_k', monitor='val_top_k_categorical_accuracy', mode='max', verbose=1,
    #                      save_best_only=True, intial_value_threshold=0.5)
    # vm = ModelCheckpoint(model_file + '_val_mag', monitor='val_magnitude', mode='max', verbose=1,
    #                      save_best_only=True, intial_value_threshold=0.2)
    vp = ModelCheckpoint(model_file + '_val_cat', monitor='val_categorical_accuracy', mode='max', verbose=1,
                         save_best_only=True, intial_value_threshold=0.2)
    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    # vm = ModelCheckpoint(model_file + '_val_bce', monitor='val_binary_crossentropy', mode='max', verbose=1,
    #                      save_best_only=True)
    reducelr = ReduceLROnPlateau(patience=5)
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2,
                        validation_data=(x_test, y_test), callbacks=[reducelr], shuffle=True)

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

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # for i in range(5):
        #     plt.plot(x_test[i, :, -stack // 2], label='filtered')
        #     sig = model.predict(x_test[i][np.newaxis, :, :])[0]
        #     sig = np.sum(sig.reshape((-1, scale_down)), axis=1) / scale_down
        #     plt.plot(scipy.special.softmax(sig), label='prediction')
        #     sig = model_top_k.predict(x_test[i][np.newaxis, :, :])[0]
        #     sig = np.sum(sig.reshape((-1, scale_down)), axis=1) / scale_down
        #     plt.plot(scipy.special.softmax(sig), label='top_k_prediction')
        #     sig = y_test[i]
        #     sig = np.sum(sig.reshape((-1, scale_down)), axis=1)
        #     sig /= -np.max(sig) * 2
        #     plt.plot(sig, label='truth')
        #     plt.legend()
        #     plt.show()

        for i in np.random.randint(len(x_train), size=10):
            plt.plot(x_train[i, :, -stack // 2], label='filtered')
            sig = model.predict(x_train[i][np.newaxis, :, :])[0]
            sig = np.sum(sig.reshape((-1, scale_down)), axis=1) / scale_down
            plt.plot(scipy.special.softmax(sig), label='prediction')
            sig = model_top_k.predict(x_train[i][np.newaxis, :, :])[0]
            sig = np.sum(sig.reshape((-1, scale_down)), axis=1) / scale_down
            plt.plot(scipy.special.softmax(sig), label='top_k_prediction')
            sig = y_train[i]
            sig = np.sum(sig.reshape((-1, scale_down)), axis=1)
            sig /= -np.max(sig) * 2
            plt.plot(sig, label='truth')
            plt.legend()
            plt.show()

    model.save(model_file)
    del model
    tf.keras.backend.clear_session()


if __name__ == '__main__':
    model_file = f'{animal}_model'

    epochs = 120
    batch_size = 64
    learning_rate = 1e-5
    x_train = np.load(os.path.join('..', 'Training', f'{animal}_x_train.npy'))
    y_train = np.load(os.path.join('..', 'Training', f'{animal}_y_train.npy'))
    # inds = np.random.randint(x_train.shape[0], size=x_train.shape[0] * 3 // 4)
    # x_train = x_train[inds]
    # y_train = y_train[inds]
    x_test = np.load(os.path.join('..', 'Training', f'{animal}_x_test.npy'))
    y_test = np.load(os.path.join('..', 'Training', f'{animal}_y_test.npy'))

    train(model_file, epochs, batch_size, learning_rate, x_train, y_train, x_test, y_test, True)
