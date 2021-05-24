import cv2
import numpy as np
import tensorflow as tf

x = tf.Variable(tf.random.uniform([1, 5, 5, 1], -1, 1))
y = tf.Variable(tf.random.uniform([1, 5, 5, 3], -1, 1))

def apply_weights(y_true, y_pred):
    y0, y1, y2 = tf.split(y_true, [1, 1, 1], 3)
    y_pred_weighted = tf.multiply(y_pred, y1)
    return y_pred_weighted

def MSE_loss_weighted(y_true, y_pred):
    y_pred = tf.divide(y_pred, 255)
    y_true = tf.divide(y_true, 255)
    y_mask, y_weights, y_amplify = tf.split(y_true, [1, 1, 1], 3)
    # tf.print(y_mask)

    y_amplify_num = y_amplify.numpy()[0, :, :, 0]

    # cv2.imshow('amplify', y_amplify_num)
    # cv2.imshow('weights', y_weights.numpy()[0, :, :, 0])
    # cv2.imshow('mask', y_mask.numpy()[0, :, :, 0])
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    y_amplify = tf.add(tf.multiply(y_amplify, 9), 1)

    y_pred_weighted = tf.multiply(y_pred, y_weights)
    y_true_weighted = tf.multiply(y_mask, y_weights)

    cv2.imshow('amplify', y_amplify.numpy()[0, :, :, 0])
    cv2.imshow('pred', y_pred_weighted.numpy()[0, :, :, 0])
    cv2.imshow('true', y_true_weighted.numpy()[0, :, :, 0])
    cv2.waitKey()
    cv2.destroyAllWindows()

    relevant_pixels = tf.reduce_sum(y_weights)

    loss_img_1 = tf.square(tf.subtract(y_pred_weighted, y_true_weighted))
    loss_img_2 = tf.multiply(y_amplify, loss_img_1)

    cv2.imshow('loss1', loss_img_1.numpy()[0, :, :, 0])
    cv2.imshow('loss2', loss_img_2.numpy()[0, :, :, 0])
    cv2.waitKey()
    cv2.destroyAllWindows()

    loss = tf.divide(tf.reduce_sum(loss_img_2), relevant_pixels)

    return loss


print('Here comes x')
tf.print(tf.shape(x))

pred = cv2.imread('X:\\BEP_data\\RL012\\Manual Masks\\3_4_2_3.png', cv2.IMREAD_GRAYSCALE)
pred_1 = tf.reshape(pred, [1, 1024, 1024, 1])
mask = cv2.imread('X:\\BEP_data\\Test_set\\Test_masks\\1\\3_4_2_3.png')
mask_1 = tf.reshape(mask, [1, 1024, 1024, 3])

tf.print(MSE_loss_weighted(mask_1, pred_1))
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
