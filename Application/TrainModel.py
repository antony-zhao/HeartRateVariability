from Model import *
import os

model_file = 'Model3h5'
if os.path.isfile(model_file):
    load_model(model_file)

train_model(200, model_file, samples=200000, batch_size=256, learning_rate=0.004)
