import numpy as np
import tensorflow as tf

x = tf.Variable(tf.random.uniform([3, 3, 2], -1, 1))
y = tf.Variable(tf.random.uniform([3, 3, 2], -1, 1))

def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    tflist = tf.unstack(y_pred, axis=2)
    return tflist


print(dice_coeff(x,y))
print('split')
print(y)