# coding: utf-8

from tensorflow.keras import layers as L
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import data_augments
import cv2
import os

_epsilon = tf.convert_to_tensor(K.epsilon(), np.float32)


def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    loss = tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


def my_loss(target, output):
    """
    A custom function defined to simply sum the pixelwise loss.
    This function doesn't compute the crossentropy loss, since that is made a
    part of the model's computational graph itself.
    Parameters
    ----------
    target : tf.tensor
        A tensor corresponding to the true labels of an image.
    output : tf.tensor
        Model output
    Returns
    -------
    tf.tensor
        A tensor holding the aggregated loss.
    """
    return - tf.reduce_sum(target * output,
                           len(output.get_shape()) - 1)


def make_weighted_loss_unet(input_shape, n_classes):
    # two inputs, one for the image and one for the weight maps
    ip = L.Input(shape=input_shape)
    # the shape of the weight maps has to be such that it can be element-wise
    # multiplied to the softmax output.
    weight_ip = L.Input(shape=(1024, 1024, 1))

    # adding the layers
    conv1 = L.Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(ip)
    conv1 = L.Dropout(0.1)(conv1)
    conv1 = L.Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    mpool1 = L.MaxPool2D()(conv1)

    conv2 = L.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mpool1)
    conv2 = L.Dropout(0.2)(conv2)
    conv2 = L.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    mpool2 = L.MaxPool2D()(conv2)

    conv3 = L.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mpool2)
    conv3 = L.Dropout(0.3)(conv3)
    conv3 = L.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    mpool3 = L.MaxPool2D()(conv3)

    conv4 = L.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mpool3)
    conv4 = L.Dropout(0.4)(conv4)
    conv4 = L.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    mpool4 = L.MaxPool2D()(conv4)

    conv5 = L.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(mpool4)
    conv5 = L.Dropout(0.5)(conv5)
    conv5 = L.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    up6 = L.Conv2DTranspose(128, 2, strides=2, kernel_initializer='he_normal', padding='same')(conv5)
    conv6 = L.Concatenate()([up6, conv4])
    conv6 = L.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = L.Dropout(0.4)(conv6)
    conv6 = L.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = L.Conv2DTranspose(64, 2, strides=2, kernel_initializer='he_normal', padding='same')(conv6)
    conv7 = L.Concatenate()([up7, conv3])
    conv7 = L.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = L.Dropout(0.3)(conv7)
    conv7 = L.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = L.Conv2DTranspose(32, 2, strides=2, kernel_initializer='he_normal', padding='same')(conv7)
    conv8 = L.Concatenate()([up8, conv2])
    conv8 = L.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = L.Dropout(0.2)(conv8)
    conv8 = L.Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = L.Conv2DTranspose(16, 2, strides=2, kernel_initializer='he_normal', padding='same')(conv8)
    conv9 = L.Concatenate()([up9, conv1])
    conv9 = L.Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = L.Dropout(0.1)(conv9)
    conv9 = L.Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    c10 = L.Conv2D(n_classes, 1, activation='sigmoid', kernel_initializer='he_normal')(conv9)

    # Add a few non trainable layers to mimic the computation of the crossentropy
    # loss, so that the actual loss function just has to peform the
    # aggregation.
    c11 = L.Lambda(lambda x: x / tf.reduce_sum(x, len(x.get_shape()) - 1, True))(c10)
    c11 = L.Lambda(lambda x: tf.clip_by_value(x, _epsilon, 1. - _epsilon))(c11)
    c11 = L.Lambda(lambda x: K.log(x))(c11)
    weighted_sm = L.multiply([c11, weight_ip])

    model = Model(inputs=[ip, weight_ip], outputs=[weighted_sm])
    return model


def generate_generator_multiple(generator, dir1, dir2, dir3, batch_size, img_height, img_width):
    genX1 = generator.flow_from_directory(dir1,
                                          target_size=(img_height, img_width),
                                          class_mode=None,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          seed=7)

    genX2 = generator.flow_from_directory(dir2,
                                          target_size=(img_height, img_width),
                                          class_mode=None,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          seed=7)

    genY1 = generator.flow_from_directory(dir3,
                                          target_size=(img_height, img_width),
                                          class_mode=None,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          seed=7)
    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        Y1i = genY1.next()
        yield [X1i, X2i], Y1i  # Yield both images and their mutual label


def train_model(ini_data_path, model_export, BATCH_SIZE=8, patience=150, IMG_CHANNELS=3):
    if IMG_CHANNELS == 3:
        using_rgb = True
    else:
        using_rgb = False

    data_gen_args = dict(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1. / 255,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect'
    )

    test_gen_args = dict(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rescale=1. / 255,
    )

    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
    mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
    weights_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

    image_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**test_gen_args)
    mask_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**test_gen_args)
    weights_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**test_gen_args)

    seed = 1

    input_generator = generate_generator_multiple(generator=image_datagen,
                                                  dir1='data/Train_set_generated/Train_data',
                                                  dir2='data/Train_set_generated/Train_weights',
                                                  dir3='data/Train_set_generated/Train_masks',
                                                  batch_size=4,
                                                  img_height=1024,
                                                  img_width=1024)

    test_generator = generate_generator_multiple(generator=image_test_datagen,
                                                 dir1='data/Test_set_generated/Test_data',
                                                 dir2='data/Test_set_generated/Test_weights',
                                                 dir3='data/Test_set_generated/Test_masks',
                                                 batch_size=4,
                                                 img_height=1024,
                                                 img_width=1024
                                                 )

    # image_generator = image_datagen.flow_from_directory(
    #     ini_data_path + 'Train_set_generated/Train_data',
    #     class_mode=None,
    #     seed=seed,
    #     target_size=(1024, 1024),
    #     color_mode=('grayscale', 'rgb')[using_rgb],
    #     batch_size=1,
    #     shuffle=True
    # )
    #
    # mask_generator = mask_datagen.flow_from_directory(
    #     'data/Train_set_generated/Train_masks',
    #     class_mode=None,
    #     seed=seed,
    #     target_size=(1024, 1024),
    #     color_mode='grayscale',
    #     batch_size=1,
    #     shuffle=True
    # )
    #
    # weights_generator = weights_datagen.flow_from_directory(
    #     'data/Train_set_generated/Train_weights',
    #     class_mode=None,
    #     seed=seed,
    #     target_size=(1024, 1024),
    #     color_mode='grayscale',
    #     batch_size=1,
    #     shuffle=True
    # )
    #
    # image_test_generator = image_test_datagen.flow_from_directory(
    #     ini_data_path + 'Test_set_generated/Test_data',
    #     class_mode=None,
    #     shuffle=False,
    #     target_size=(1024, 1024),
    #     color_mode=('grayscale', 'rgb')[using_rgb],
    #     batch_size=1
    # )
    # mask_test_generator = mask_test_datagen.flow_from_directory(
    #     ini_data_path + 'Test_set_generated/Test_masks',
    #     class_mode=None,
    #     shuffle=False,
    #     target_size=(1024, 1024),
    #     color_mode='grayscale',
    #     batch_size=1
    # )
    # weights_test_generator = weights_test_datagen.flow_from_directory(
    #     ini_data_path + 'Test_set_generated/Test_weights',
    #     class_mode=None,
    #     shuffle=False,
    #     target_size=(1024, 1024),
    #     color_mode='grayscale',
    #     batch_size=1
    # )
    #
    # train_generator = zip(zip(image_generator, weights_generator), mask_generator)
    # test_generator = zip(zip(image_test_generator, weights_test_generator), mask_test_generator)

    model = make_weighted_loss_unet((1024, 1024, 3), 1)

    model.compile(optimizer='adam', loss=[my_loss], metrics=[dice_loss])
    model.summary()

    checkpointer = tf.keras.callbacks.ModelCheckpoint('latest_model.h5', verbose=1, save_best_only=True)
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=patience),
                 tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=1)]

    results = model.fit(input_generator, validation_data=test_generator, validation_steps=1, steps_per_epoch=80,
                        epochs=50, callbacks=callbacks, batch_size=BATCH_SIZE, validation_batch_size=1)
    model.save(model_export + '.h5', include_optimizer=False)
    print('Done! Model can be found in ' + model_export)
    return True


def Use_Model(model_path, data_path, img_strs, export_path='data/Nuclei_masks/', Zlevel=1):
    model = tf.keras.models.load_model(model_path + '.h5', compile=False)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255
    )
    file_list = os.listdir(data_path + 'Validate_set/Input')
    validation_generator = test_datagen.flow_from_directory(data_path + 'Validate_set/',
                                                            target_size=(1024, 1024),
                                                            batch_size=4,
                                                            shuffle=False,
                                                            color_mode='rgb')

    output = model.predict(validation_generator)
    for i, pic in enumerate(output):
        cv2.imwrite(export_path + str(Zlevel) + '/' + '{}.png'.format(img_strs[i]), pic * 255)
    return True


if __name__ == '__main__':
    ini_data_path = 'data/'
    results = train_model('data/', 'auto_models/')
    Zlevel = 3

    img_strs = data_augments.gen_input_from_img_coords(ini_data_path, (1, 1, 4, 4), Z=Zlevel, use_predicted_data=False,
                                                       only_EM=False)
    Use_Model('Models/latest_model', ini_data_path, img_strs, Zlevel=Zlevel)
