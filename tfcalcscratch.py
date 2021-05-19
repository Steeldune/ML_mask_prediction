import numpy as np
import tensorflow as tf

x = tf.Variable(tf.random.uniform([1, 5, 5, 1], -1, 1))
y = tf.Variable(tf.random.uniform([1, 5, 5, 3], -1, 1))

def apply_weights(y_true, y_pred):
    y0, y1, y2 = tf.split(y_true, [1, 1, 1], 3)
    y_pred_weighted = tf.multiply(y_pred, y1)
    return y_pred_weighted


print('Here comes x')
tf.print(x)
tf.print(tf.shape(x))
print('Here comes the reduced sum')
tf.print(tf.reduce_sum(x))
# print('Here comes y')
# tf.print(y)
# tf.print(tf.shape(y))
# print('Here comes y unstacked')
# y1 = apply_weights(y, x)
# tf.print(y1)
# tf.print(tf.shape(y1))
# z = tf.multiply(x, y1)
# print('Here comes the multipliication')
# tf.print(z)
# tf.print(tf.shape(z))
