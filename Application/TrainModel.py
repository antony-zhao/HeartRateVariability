from Model import *
import os

model_file = 'Model5.h5'
if os.path.isfile(model_file):
    load_model(model_file)
model_file = 'Model5.h5'

train_model(10, model_file, samples=16384, batch_size=128)