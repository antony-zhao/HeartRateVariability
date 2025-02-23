import os
import scipy
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten, \
    MaxPooling1D, BatchNormalization, TimeDistributed, Reshape, \
    Activation, LayerNormalization, Input, GRU, Bidirectional, Concatenate, \
    MultiHeadAttention, LSTM, Permute, GlobalAveragePooling1D, Embedding, Add, Layer
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from config import window_size, stack, scale_down, datapoints, animal
import pandas as pd
from dataset import bandpass_filter, process_sample, highpass_filter, process_segment
from tensorflow.keras.models import load_model
from config import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def weighted_crossentropy(weight):
    def loss(labels, logits):
        return tf.nn.weighted_cross_entropy_with_logits(labels, logits, pos_weight=weight)

    return loss


embedding_dim = 256
num_heads = 16
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


class PositionalEmbedding(Layer):
    def __init__(self, sequence_length, embedding_dim):
        super(PositionalEmbedding, self).__init__()
        # Create a learnable embedding matrix of shape (se quence_length, embedding_dim)
        self.position_embeddings = Embedding(input_dim=sequence_length, output_dim=embedding_dim)

    def call(self, inputs):
        # `tf.range` generates the sequence of position indices for embeddings (0, 1, ..., sequence_length - 1)
        positions = tf.range(start=0, limit=tf.shape(inputs)[1], delta=1)
        pos_embeddings = self.position_embeddings(positions)  # Shape: (sequence_length, embedding_dim)
        return inputs + pos_embeddings  # Broadcast addition to add position embeddings to inputs


# positions = Input((stack,))
x = inputs = Input((stack * window_size, 12))
x = Conv1D(filters=16, kernel_size=31, padding='same', strides=5, activation='relu')(x)
x = BatchNormalization()(x)
x = Conv1D(filters=32, kernel_size=25, padding='same', strides=1, activation='relu')(x)
x = MaxPooling1D(strides=5)(x)
x = BatchNormalization()(x)
# x = Conv1D(filters=64, kernel_size=17, padding='same', strides=2, activation='relu')(x)
# x = BatchNormalization()(x)
x = Conv1D(filters=64, kernel_size=17, padding='same', strides=1, activation='relu')(x)
x = MaxPooling1D(strides=4)(x)
x = BatchNormalization()(x)
x = Conv1D(filters=embedding_dim, kernel_size=11, padding='same', strides=1, activation='relu')(x)
pos = PositionalEmbedding(stack, embedding_dim)(x)
embedding = Dropout(dropout)(Dense(embedding_dim)(x))
x = Add()([embedding, pos])
for _ in range(4):
    x = attention_layer(x)
x = LayerNormalization()(x)
x = TimeDistributed(Dense(100))(x)
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
                           keras.metrics.AUC(from_logits=True, multi_label=True),
                           ])
    va = ModelCheckpoint(model_file + '_val_auc', monitor='val_auc', mode='max', verbose=1,
                         save_best_only=True, initial_value_threshold=0.15)
    vk = ModelCheckpoint(model_file + '_val_top_k', monitor='val_top_k_categorical_accuracy', mode='max',
                         verbose=1,
                         save_best_only=True, initial_value_threshold=0.9)
    vc = ModelCheckpoint(model_file + '_val_bin', monitor='val_binary_accuracy', mode='max', verbose=1,
                         save_best_only=True, initial_value_threshold=0.9)
    vp = ModelCheckpoint(model_file + '_val_cat', monitor='val_sparse_categorical_accuracy', mode='max', verbose=1,
                         save_best_only=True, initial_value_threshold=0.9)
    # vm = ModelCheckpoint(model_file + '_val_bce', monitor='val_binary_crossentropy', mode='max', verbose=1,
    #                      save_best_only=True)
    reducelr = ReduceLROnPlateau(monitor='val_auc', patience=10)
    eary_stop = EarlyStopping(monitor='val_auc', patience=20)
    history = model.fit(x_train, y_train, batch_size=None, epochs=epochs, verbose=2, sample_weight=sample_weight,
                        validation_data=(x_test, y_test), callbacks=[va, reducelr, eary_stop], shuffle=True)

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


def train_generator(model_file, epochs, batch_size, learning_rate, train_data, val_data, steps_per_epoch, val_steps,
                    plot=False,
                    sample_weight=None, loaded_model=None):
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
    if loaded_model is not None:
        model = load_model(loaded_model)
    optim = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optim,
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=[keras.metrics.BinaryAccuracy(),
                           keras.metrics.AUC(curve='PR', from_logits=True, multi_label=True),
                           keras.metrics.AUC(from_logits=True, multi_label=True),
                           keras.metrics.Precision(thresholds=0),
                           keras.metrics.Recall(thresholds=0),
                           ])
    va = ModelCheckpoint(model_file + '_val_auc', monitor='val_auc', mode='max', verbose=1,
                         save_best_only=True, initial_value_threshold=0.4)
    vk = ModelCheckpoint(model_file + '_val_top_k', monitor='val_top_k_categorical_accuracy', mode='max',
                         verbose=1,
                         save_best_only=True, initial_value_threshold=0.9)
    vc = ModelCheckpoint(model_file + '_val_bin', monitor='val_binary_accuracy', mode='max', verbose=1,
                         save_best_only=True, initial_value_threshold=0.9)
    vp = ModelCheckpoint(model_file + '_val_precision', monitor='val_precision', mode='max', verbose=1,
                         save_best_only=True, initial_value_threshold=0.9)
    vr = ModelCheckpoint(model_file + '_val_recall', monitor='val_recall', mode='max', verbose=1,
                         save_best_only=True, initial_value_threshold=0.9)
    reducelr = ReduceLROnPlateau(patience=10)
    early_stop = EarlyStopping(monitor='val_auc', patience=30)
    history = model.fit(train_data, batch_size=None, epochs=epochs, verbose=2, sample_weight=sample_weight,
                        validation_data=val_data, callbacks=[va, reducelr, vp, vr, early_stop], shuffle=True,
                        steps_per_epoch=steps_per_epoch, validation_steps=val_steps)

    def plot(sample):
        plt.plot(sample[:, 0])
        plt.show()

    for _ in range(20):
        x_sample, y_sample = next(val_data)
        y_pred = model.predict(x_sample[0][np.newaxis, :, :])[0]
        y_pred = scipy.special.expit(y_pred)
        plt.plot(y_pred.flatten())
        plot(x_sample[0])

    model.save(model_file)
    del model
    keras.backend.clear_session()


if __name__ == '__main__':
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow.keras.models import load_model

    model_file = f'{animal}_model'


    def augment_data(X):
        normal_noise = np.random.normal(0, 0.01, X.shape)
        wavelengths = [np.random.uniform(800, 2000) for _ in range(4)]
        sinusoidal_noise = np.zeros(X.shape)
        for wavelength in wavelengths:
            sinusoidal_noise += np.sin(
                np.arange(X.size) * (2 * np.pi / wavelength) + np.random.randint(0, 1000)) * np.random.uniform(0, 0.01)
        X_aug = X + normal_noise + sinusoidal_noise
        return X_aug

    def data_generator(X, y, batch_size=128, steps_per_epoch=500, augment=True):
        shuffle = True
        if augment:
            X_augmented = augment_data(X)
            X_normal_aug = process_segment(X_augmented)
        X_normal = process_segment(X)
        while True:
            if shuffle:
                shuffle = False  # This loop is used to run the generator indefinitely.
                random_inds = np.random.randint(0, len(X) - stack * window_size, batch_size * steps_per_epoch * 2)
                random_inds = random_inds.reshape(steps_per_epoch, batch_size * 2)
                if augment:
                    X_augmented = augment_data(X)
                    X_normal_aug = process_segment(X_augmented)
                X_normal = process_segment(X)
            else:
                for inds in random_inds:
                    data = []
                    labels = []
                    for ind in inds:
                        y_i = y[ind:ind + int(stack * window_size)]
                        count = np.count_nonzero(y_i[:800])
                        if count < 2:
                            continue

                        if np.random.rand() < 0.5 or not augment:
                            x_i = X_normal[ind:ind + int(stack * window_size)] * np.random.uniform(0.5, 1.2)
                            x_i = process_sample(x_i)
                        else:
                            x_i = X_normal_aug[ind:ind + int(stack * window_size)] * np.random.uniform(0.5, 1.2)
                            x_i = process_sample(x_i)

                        data.append(x_i)
                        labels.append(y_i)

                    data = np.stack(data)
                    labels = np.stack(labels)
                    yield data, labels
                shuffle = True


    epochs = 120
    batch_size = 128
    learning_rate = 1e-3
    steps_per_epoch = 4000

    df = pd.read_csv(os.path.join('..', 'Training', f'{animal}_train.csv'), header=None)
    x_train = df[0].to_numpy()
    y_train = df[1].to_numpy()
    df = pd.read_csv(os.path.join('..', 'Training', f'{animal}_val.csv'), header=None)
    x_test = df[0].to_numpy()
    y_test = df[1].to_numpy()

    train_generator(model_file, epochs, batch_size, learning_rate,
                    data_generator(x_train, y_train, batch_size, steps_per_epoch),
                    data_generator(x_test, y_test, batch_size, steps_per_epoch, augment=False),
                    steps_per_epoch=steps_per_epoch, val_steps=steps_per_epoch // 10)
