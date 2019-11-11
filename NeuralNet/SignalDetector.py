from keras.layers import Conv1D, Recurrent, AveragePooling1D, Dense, Activation, Dropout, Flatten
from keras.activations import sigmoid, relu
from keras.models import Sequential


layers_dims = [256, 400, 300, 256]

model = Sequential()
model.add(Conv1D(layers_dims[0]))
model.add(AveragePooling1D())
model.add(Flatten())
model.add(Recurrent(layers_dims[1]))
model.add(Activation(relu))
model.add(Dropout(0.5))
model.add(Dense(layers_dims[2]))
model.add(Activation(relu))
model.add(Dropout(0.5))
model.add(Dense(layers_dims[3]))
model.add(Activation(sigmoid))

print("hi")
