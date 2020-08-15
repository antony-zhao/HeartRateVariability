from Model import *
import os

model_file = 'Model3.h5'
if os.path.isfile(model_file):
    print("loaded")
    load_model(model_file)

train_model(100, model_file, samples=60000, batch_size=32, learning_rate=0.00001)
