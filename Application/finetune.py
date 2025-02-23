import os
import pandas as pd
from model import train_generator
from tensorflow.keras.models import load_model
from config import animal, stack, window_size
from dataset import process_segment, process_sample
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
    model_file = f'{animal}_model_finetune'


    def data_generator(X_list, y_list, batch_size=128, steps_per_epoch=500):
        """
        Generator to handle multiple arrays. It yields batches of data and labels for any number of input arrays.

        Args:
            X_list (list of np.ndarray): List of feature arrays.
            y_list (list of np.ndarray): List of label arrays.
            batch_size (int): Number of samples per batch.
            steps_per_epoch (int): Number of steps per epoch.
            invert (bool): Flag for optional processing.

        Yields:
            tuple: Batch of data and corresponding labels.
        """
        shuffle = True

        # Preprocess each array in the input list
        X_normal_list = [process_segment(X) for X in X_list]

        while True:
            if shuffle:
                shuffle = False  # This loop is used to run the generator indefinitely.

                random_inds_list = []
                for X in X_list:
                    # Generate random indices for each array
                    random_inds = np.random.randint(0, len(X) - stack * window_size, batch_size // len(X_list) * steps_per_epoch * 2)
                    random_inds_list.append(random_inds.reshape(steps_per_epoch, batch_size // len(X_list) * 2))
            else:
                for step in range(steps_per_epoch):
                    data = []
                    labels = []

                    for X_normal, y, random_inds in zip(X_normal_list, y_list, random_inds_list):
                        inds = random_inds[step]  # Get the indices for this step

                        for ind in inds:
                            y_i = y[ind:ind + int(stack * window_size)]
                            count = np.count_nonzero(y_i[:800])
                            if count < 2:
                                continue

                            x_i = X_normal[ind:ind + int(stack * window_size)]
                            x_i = process_sample(x_i)

                            data.append(x_i)
                            labels.append(y_i)

                    # Stack the data and labels from all arrays
                    data = np.stack(data)
                    labels = np.stack(labels)

                    yield data, labels

                shuffle = True


    epochs = 120
    batch_size = 256
    learning_rate = 1e-4
    steps_per_epoch = 4000

    train_files = ['finetune_train1.csv', 'finetune_train2.csv', 'finetune_train3.csv']
    x_train = []
    y_train = []
    for train_file in train_files:
        df = pd.read_csv(os.path.join('..', 'Training', train_file), header=None)
        x_train.append(df[0].to_numpy())
        y_train.append(df[1].to_numpy())
    # x_train = np.concatenate(x_train)
    # y_train = np.concatenate(y_train)

    val_files = ['finetune_val1.csv', 'finetune_val2.csv', 'finetune_val3.csv']
    x_test = []
    y_test = []
    for val_file in val_files:
        df = pd.read_csv(os.path.join('..', 'Training', val_file), header=None)
        x_test.append(df[0].to_numpy())
        y_test.append(df[1].to_numpy())
    # x_test = np.concatenate(x_test)
    # y_test = np.concatenate(y_test)
    model = f'{animal}_model_val_recall'

    train_generator(model_file, epochs, batch_size, learning_rate,
                    data_generator(x_test, y_test, batch_size, steps_per_epoch),
                    data_generator(x_train, y_train, batch_size, steps_per_epoch),
                    steps_per_epoch=steps_per_epoch, val_steps=steps_per_epoch // 10, loaded_model=model)
