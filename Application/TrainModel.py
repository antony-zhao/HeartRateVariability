from Model import *
import os

model_file = 'Model.h5'
if os.path.isfile(model_file):
    print("loaded")
    load_model(model_file)

train_model(30, model_file, samples=1000000, batch_size=512, learning_rate=0.004)
