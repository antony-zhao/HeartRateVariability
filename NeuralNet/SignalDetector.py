import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Recurrent, Dropout, Conv1D
from keras.activations import relu, sigmoid

tf.config.experimental.list_physical_devices('GPU')
tf.debugging.set_log_device_placement(True)

# Place tensors on the CPU
#with tf.device('/CPU:0'):
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

c = tf.matmul(a, b)
print(c)