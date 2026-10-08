import numpy as np
import tensorflow as tf

import geoml

g = np.array([0.093, -0.47, 1e-6, 3.0])
v = tf.Variable(np.zeros(4), dtype=tf.float64)
opt = geoml.models.VGPNetwork._expert_adam()
opt.build([v])
opt.apply_gradients([(tf.constant(g), v)])
keras_step = v.numpy()
print("keras lr dtype", opt.learning_rate.dtype if hasattr(opt.learning_rate, "dtype") else type(opt.learning_rate))
print("keras step ", keras_step)
lr = 0.01
m1 = (1 - 0.9) * g
v1 = (1 - 0.999) * g ** 2
alpha = lr * np.sqrt(1 - 0.999) / (1 - 0.9)
mine = -m1 * alpha / (np.sqrt(v1) + 1e-7)
print("formula    ", mine)
print("ratio      ", keras_step / mine)
print("epsilon", opt.epsilon, "beta", opt.beta_1, opt.beta_2)
