import numpy as np
import os
from config import animal

# Using this to augment the rat data we have since it isn't nearly as varied compared to the mouse data.

rat_x_train = np.load(os.path.join('..', 'Training', 'rat_x_train.npy'))
rat_y_train = np.load(os.path.join('..', 'Training', 'rat_y_train.npy'))
rat_x_test = np.load(os.path.join('..', 'Training', 'rat_x_test.npy'))
rat_y_test = np.load(os.path.join('..', 'Training', 'rat_y_test.npy'))

mouse_x_train = np.load(os.path.join('..', 'Training', 'mouse_x_train.npy'))
mouse_y_train = np.load(os.path.join('..', 'Training', 'mouse_y_train.npy'))
mouse_x_test = np.load(os.path.join('..', 'Training', 'mouse_x_test.npy'))
mouse_y_test = np.load(os.path.join('..', 'Training', 'mouse_y_test.npy'))

x_train = np.concatenate([rat_x_train, mouse_x_train], axis=0)
y_train = np.concatenate([rat_y_train, mouse_y_train], axis=0)
x_test = np.concatenate([rat_x_test, mouse_x_test], axis=0)
y_test = np.concatenate([rat_y_test, mouse_y_test], axis=0)

np.save(os.path.join('..', 'Training', f'{animal}_x_train'), x_train)
np.save(os.path.join('..', 'Training', f'{animal}_y_train'), y_train)
np.save(os.path.join('..', 'Training', f'{animal}_x_test'), x_test)
np.save(os.path.join('..', 'Training', f'{animal}_y_test'), y_test)
