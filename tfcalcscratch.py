import tensorflow as tf
import numpy as np

x = tf.Variable(tf.random.uniform([5, 30], -1, 1))

# Split `x` into 3 tensors along dimension 1
s0, s1, s2 = tf.split(x, num_or_size_splits=3, axis=1)
tf.shape(s0).numpy()