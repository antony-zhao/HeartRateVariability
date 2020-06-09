from Model import *
import os

model_file = 'Model1.h5'
if os.path.isfile(model_file):
    load_model(model_file)

train_model(20, model_file, samples=16384, batch_size=64, learning_rate=0.01)