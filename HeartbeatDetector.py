# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 15:36:38 2019

@author: Antony Zhao
"""
from functions import *
from keras.models import Sequential
from keras.layers import Conv1D, Dense, MaxPooling1D, LSTM, Flatten

model = Sequential()
model.add(Conv1D(20,(16),padding = 'valid',activation = 'relu'))
model.add(MaxPooling1D())
model.add(Conv1D(40,(8),padding = 'valid'))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(800,activation = 'relu'))
model.add(LSTM(400,activation = 'relu'))
model.add(Dense(400, activation = 'softmax'))


