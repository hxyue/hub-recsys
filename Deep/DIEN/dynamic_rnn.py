import tensorflow as tf

import numpy as np

n_steps = 2

n_inputs = 3

n_neurons = 5

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
