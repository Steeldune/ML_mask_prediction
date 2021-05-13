import datetime

import tensorflow as tf
import numpy as np
import os
import shutil
import errno

# from tensorflow.keras.preprocessing import image_dataset_from_directory
import time
from glob import glob
from start_over import add_threshold

# pragma warning(disable:4996)
from PIL import Image
from tqdm import tqdm

import cv2


def dice_coeff(y_true, y_pred):
    """
    Measures the dice coefficient
    :param y_true: The true image for validation.
    :param y_pred: The image predicted by the network
    :return: Dice coefficient. The closer to one, the more the images are alike.
    """
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    """
    returns a dice coefficient which can be used as a loss parameter (aka, the closer to zero, the better)
    :param y_true: The true image for validation
    :param y_pred: The predicted image from the network
    :return: Dice loss, the closer to zero, the better.
    """
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    """
    Returns the sum of the binary cross entropy score and the dice loss score
    :param y_true: The true image for validation
    :param y_pred: The predicted image from the network
    :return: bce dice loss score
    """
    loss = tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


def cv2_imread(file_path):
    """
    Expands the cv2.imread function to be a little more forgiving with its input string.
    :param file_path: string describing an image address
    :return: the image at that address.
    """
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


def scandirs(path):
    """
    Deletes all png's and jpg's in a file hierarchy
    :param path: The address of the highest level address
    :return:
    """
    for root, dirs, files in os.walk(path):
        for currentFile in files:
            exts = ('.png', '.jpg')
            if currentFile.lower().endswith(exts):
                os.remove(os.path.join(root, currentFile))

def backup_predicts(path, dest):
    try:
        shutil.copytree(path, dest)
    except OSError as exc: # python >2.5
        if exc.errno == errno.ENOTDIR:
            shutil.copy(path, dest)
        else: raise


def Train_Model(ini_data_path, model_export, IMG_WIDTH=1024, IMG_HEIGHT=1024,
                IMG_CHANNELS=3, BATCH_SIZE=8, patience=100, normalize=False, using_weights=False):
    """
    This is the training function, which compiles a network, and uses the data given to train a model.
    :param ini_data_path:   Address where the following files have to be found:
                                Test_set
                                Train_set
    :param model_export:    Address where the model will be saved eventually
    :param IMG_WIDTH:       Expected width of images in pixels
    :param IMG_HEIGHT:      Expected height if images in pixels
    :param IMG_CHANNELS:    Expected amount of channels in image training data
    :param BATCH_SIZE:      Amount of images loaded into the model per step.
    :param patience:        Amount of epochs with no noticeable improvement that have to occur before the training terminates early
    :param normalize:       Boolean on whether the input data will be normalized along the standard deviation
    :param using_weights:   Boolean on whether the validation data will include weights (and therefore have multiple channels)
    :return:                True
    """
    if IMG_CHANNELS == 3:
        using_rgb = True
    else:
        using_rgb = False

    data_gen_args = dict(
        samplewise_std_normalization=normalize,
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
    """Training data is augmented by flipping, rotating and shifting the image. Any part of the image that has no data will be filled with reflected data."""

    test_gen_args = dict(
        samplewise_std_normalization=normalize,
        featurewise_center=False,
        featurewise_std_normalization=False,
        rescale=1. / 255,
    )
    """Testing data is not augmented"""

    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
    mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

    image_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**test_gen_args)
    mask_test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**test_gen_args)

    seed = 1

    image_generator = image_datagen.flow_from_directory(
        ini_data_path + 'Train_set/Train_data',
        class_mode=None,
        seed=seed,
        target_size=(1024, 1024),
        color_mode=('grayscale', 'rgb')[using_rgb],
        batch_size=1,
        shuffle=True
    )

    mask_generator = mask_datagen.flow_from_directory(
        ini_data_path + 'Train_set/Train_masks',
        class_mode=None,
        seed=seed,
        target_size=(1024, 1024),
        color_mode=('grayscale', 'rgb')[using_weights],
        batch_size=1,
        shuffle=True
    )

    image_test_generator = image_test_datagen.flow_from_directory(
        ini_data_path + 'Test_set/Test_data',
        class_mode=None,
        shuffle=False,
        target_size=(1024, 1024),
        color_mode=('grayscale', 'rgb')[using_rgb],
        batch_size=1
    )
    mask_test_generator = mask_test_datagen.flow_from_directory(
        ini_data_path + 'Test_set/Test_masks',
        class_mode=None,
        shuffle=False,
        target_size=(1024, 1024),
        color_mode=('grayscale', 'rgb')[using_weights],
        batch_size=1
    )

    train_generator = zip(image_generator, mask_generator)
    test_generator = zip(image_test_generator, mask_test_generator)

    input1 = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))

    s1 = tf.keras.layers.Lambda(lambda x: x / 255)(input1)

    # Contraction Layer
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s1)
    c1 = tf.keras.layers.Dropout(.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Expansion layer
    u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3], axis=3)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2], axis=3)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = tf.keras.Model(inputs=[input1], outputs=[outputs])
    model.compile(optimizer='adam', loss=[bce_dice_loss], metrics=[dice_loss])
    model.summary()

    ####################################################################################################################

    checkpointer = tf.keras.callbacks.ModelCheckpoint('{}.h5'.format('10jan1058'), verbose=1, save_best_only=True)
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=patience),
                 tf.keras.callbacks.TensorBoard(log_dir='logs', histogram_freq=1)]

    results = model.fit(train_generator, validation_data=test_generator, steps_per_epoch=35 // BATCH_SIZE,
                        validation_steps=15 // BATCH_SIZE,
                        epochs=1000, callbacks=callbacks, batch_size=BATCH_SIZE)
    model.save(model_export + '.h5', include_optimizer=False)
    print('Done! Model can be found in ' + model_export)
    return True


def Use_Model(model_path, data_path, glob_str, dataset, export_path='X:\\BEP_data\\Predict_set\\', only_EM=False,
              HO_adjust=False, normalize=False):
    """
    This function loads a model that has been made, and saves predictions from the data it has been given. Supply it
    with a folder and a filter for the images to predict on, and you're golden. Input data will be pulled
    from given datasets, and will be placed in the same file structure where output data will be exported to.
    :param model_path:  Address where the model can be found.
    :param data_path:   Address where datasets can be found. For instance, the folder containing 'RL012'.
    :param glob_str:    String on which the images within the data_path will be filtered. For instance, putting in '4_*' will only predict on images of z-level 4.
    :param dataset:     Specifies which dataset will be used. For instance: 'RL012'
    :param export_path: Address where the prediction data will be exported to, both input and output.
    :param only_EM:     Boolean on whether the input data will contain only EM or more (Hoechst data, for example)
    :param HO_adjust:   Boolean on whether the secondary (non-EM) data will be thresholded. Only use in very specific cases.
    :param normalize:   Boolean on whether to normalize input data.
    :return:            True
    """
    EM_addresses = glob(data_path + dataset + '\\EM\\Collected\\' + glob_str)
    for EM_ad in EM_addresses:
        """For each image in the glob string, the EM data is pulled, along with the secondary data, if requested. 
        This will then be written into the predict_data folder as input data. This operation is deliberate, 
        so that after the prediction has been made, the input data can also be looked at."""
        img_str = EM_ad.split('\\')[-1]
        EM_img = cv2_imread(EM_ad)
        if only_EM:
            out_img = cv2_imread(EM_ad)
        else:
            HO_img = cv2_imread(data_path + dataset + '\\Hoechst\\Collected\\' + img_str)
            if normalize:
                HO_img = cv2.normalize(HO_img, None, 255, 0, cv2.NORM_INF)
            if HO_adjust:
                HO_img = add_threshold(HO_img, 701, -50)
            out_img = np.dstack((EM_img, HO_img, np.zeros((1024, 1024), np.uint8)))
        cv2.imwrite(export_path + 'Input\\1\\' + img_str, out_img)

    model = tf.keras.models.load_model(model_path + '.h5', compile=True)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255
    )
    validation_generator = test_datagen.flow_from_directory(export_path + 'Input\\',
                                                            target_size=(1024, 1024),
                                                            batch_size=4,
                                                            shuffle=False,
                                                            color_mode=('rgb', 'grayscale')[only_EM])

    output = model.predict(validation_generator)
    for i, pic in enumerate(output):
        cv2.imwrite(export_path + 'Output\\{}'.format(EM_addresses[i].split('\\')[-1]), pic * 255)
    """Once the prediction has been made, it is written as a png in the predict_set folder."""
    return True


if __name__ == '__main__':
    new_time = time.asctime()
    model_name = 'sup_base_emho'
    today = datetime.datetime.now()

    scandirs('X:\\BEP_data\\Predict_set')
    ini_data_path = 'X:\\BEP_data\\'
    dataset = 'RL015'
    glob_str = 'blob*'
    Ho_adjust = False
    # Train_Model(ini_data_path, 'Models\\{}'.format('sup_base_emho'), IMG_CHANNELS=3, BATCH_SIZE=4, patience=70, normalize=False)
    # img_strs = data_augments.gen_input_from_img_coords(ini_data_path, (1, 1, 4, 4), Z=Zlevel, use_predicted_data=False, only_EM=False)
    #

    Use_Model('Models\\{}'.format(model_name), ini_data_path, glob_str, dataset, HO_adjust=Ho_adjust, only_EM=False, normalize=True)


    # particle_analysis.ShowResults('data/Nuclei_masks/' + str(Zlevel) + '/', ini_data_path, img_strs, Zlevel=Zlevel,
    #                               upscaleTo=0, threshold_masks=True)
    #
    """Beneath is some data processing code which takes predictions and does two things:
            It makes an EM overlay image, where the EM, the secondary data and the prediction can be seen 
            together in one image. 
            EM is a black and white background, the secondary data is green, and the prediction is red.
            
            The second thing it does is it looks for any manual masks in the data folder, and makes mask overlap images 
            with the predictions. 
            In these overlap images, blue indicates a false negative, and red indicates a false positive. 
            Black and white are true negative and true positive respectively.
        Finally, the images are stored in a backup location named after the model that was used, and the time it was invoked.
    """

    img_strs = glob(ini_data_path + '{}\\EM\\Collected\\{}'.format(dataset, glob_str))
    for img1 in img_strs:
        img = img1.split('\\')[-1]
        mask_img = cv2_imread('X:\\BEP_data\\Predict_set\\Output\\' + img) / 255
        EM_img = cv2_imread('X:\\BEP_data\\{}\\EM\\Collected\\'.format(dataset) + img)
        HO_img = cv2_imread('X:\\BEP_data\\{}\\Hoechst\\{}\\'.format(dataset, ('Collected', 'Collected_thresh')[Ho_adjust]) + img)
        masked_img = np.dstack((EM_img * (1 - (cv2.normalize(HO_img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX))) * (1-mask_img),
                                EM_img * (1- mask_img),
                                EM_img * (1 - (cv2.normalize(HO_img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)))))
        # cv2.imshow('{}_{}'.format(Zlevel, img), masked_img/255)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        cv2.imwrite('X:\\BEP_data\\Predict_set\\EM_overlay\\' + 'em_overlay_' + img, masked_img)


        if glob('X:\\BEP_data\\{}\\Manual Masks\\'.format(dataset) + img) != []:

            man_mask_img = cv2_imread('X:\\BEP_data\\{}\\Manual Masks\\'.format(dataset) + img)
            overl_img = mask_img * man_mask_img/255
            overl_img = np.multiply(overl_img, 255.0)

            mask_overlap = np.dstack((man_mask_img, overl_img.astype(np.uint8), np.multiply(mask_img, 255.0).astype(np.uint8)))
            cv2.imwrite('X:\\BEP_data\\Predict_set\\Mask_overlaps\\'+ 'overlap_' + img, mask_overlap)


    backup_predicts('X:\\BEP_data\\Predict_set', 'X:\\BEP_data\\Predict_backups\\{}'.format(model_name+'_'+today.strftime("%Y-%m-%d_%H-%M-%S")))

