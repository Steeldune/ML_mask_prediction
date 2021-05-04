# import tensorflow as tf
import os
import numpy as np
import cv2
import getEnvelope
import re
import glob



def cv2_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


# This file creates a list of files in a path.
def get_file_list(ini_path):
    file_list = os.listdir(ini_path)
    zoom_list = []
    for i in range(7):
        zoom_list.append([val for val in file_list if re.search(fr'^\d+_\d+_[{i}].png', val)])


def gen_input_from_mask(data_path, predicted=False, only_EM=False, adjust_test=False, generated='', dataset='RL012', normalize=False):
    file_list = os.listdir(data_path + ('Train_set{}\\Train_masks\\1\\'.format(generated), 'Test_set\\Test_masks\\1\\'.format(generated))[adjust_test])
    export_path = data_path + ('Train_set{}\\Train_data\\1\\'.format(generated), 'Test_set{}\\Test_data\\1\\'.format(generated))[adjust_test]

    for img_string in file_list:
        EM_path = data_path + '{}\\EM\\Collected\\{}'.format(dataset, img_string)
        if not only_EM:
            HO_path = data_path + ('{}\\Hoechst\\Collected\\{}'.format(dataset, img_string))



        EM_img = cv2_imread(EM_path)
        if not only_EM:
            HO_img = cv2_imread(HO_path)
            if normalize:
                HO_img = cv2.normalize(HO_img, None, 255, 0, cv2.NORM_INF)
            TR_img = np.dstack((EM_img, HO_img, np.zeros((1024,1024))))
        else:
            TR_img = EM_img
        cv2.imwrite(
            export_path + img_string,
            TR_img)
    return True


def gen_input_from_img_coords(data_path, bound_coords, zoom=3, Z=1, export_path=None, use_predicted_data=False, only_EM=False, generate_input=True):
    if export_path is None:
        export_path = data_path + 'Validate_set/Input/'
    EM_path = data_path + 'EM/' + str(Z) + '/'
    if not only_EM:
        HO_path = data_path + ('Hoechst/', 'Hoechst_predicted/')[use_predicted_data] + str(Z) + '/'
        IN_path = data_path + ('Insulin/', 'Insulin_predicted/')[use_predicted_data] + str(Z) + '/'

    img_strs = getEnvelope.get_data(data_path, Z, zoom, bound_coords, JustStrs=True)

    files = glob.glob(export_path + '\\*')
    for f in files:
        os.remove(f)

    for img_string in img_strs:

        EM_img = cv2_imread(EM_path + img_string + '.png')
        if not only_EM:
            HO_img = cv2_imread(HO_path + img_string + '.png')
            IN_img = cv2_imread(IN_path + img_string + '.png')
            TR_img = np.dstack((EM_img, HO_img, IN_img))
        else:
            TR_img = EM_img
        if generate_input:
            cv2.imwrite(
                export_path + img_string + '.png',
                TR_img)
    return img_strs


# def test_data_augments(ini_data_path, img_str = '1_3_3_3'):
#     data_gen_args = dict(
#         featurewise_center=False,
#         featurewise_std_normalization=False,
#         rotation_range=90,
#         width_shift_range=0.2,
#         height_shift_range=0.2,
#         rescale=1. / 255,
#         horizontal_flip=True,
#         vertical_flip=True,
#         fill_mode='reflect'
#     )
#
#     image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
#     mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
#
#     seed = 2
#
#     x_input = cv2.imread(ini_data_path + 'Train_set/Train_data/1/' + img_str + '.png')
#     y_input = cv2.imread(ini_data_path + 'Train_set/Train_masks/1/' + img_str + '.png')
#
#     x_input = image_datagen.random_transform(x_input, seed)
#     y_input = image_datagen.random_transform(y_input, seed)
#
#     cv2.imshow('Input Data', x_input)
#     cv2.imshow('Input Masks', y_input)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     return True


# def gen_dataset_from_file(data_path):
#     train_data = tf.io.read_file(data_path)
#     train_data = tf.io.decode_image(train_data)
#     return train_data

def gen_zoomed_training_data(data_path,current_zoom = 3, scale_down=True):

    zoom_factor = np.power(2, current_zoom - (current_zoom-1))
    file_list = os.listdir(data_path + 'Train_set/Train_masks/1/')
    img_list = [val for val in file_list if re.search(fr'^\d+_\d+_\d+_[{current_zoom}].png', val)]
    img_list = [x.split('_') for x in img_list]
    img_list = [[int(y.split('.')[0]) for y in x] for x in img_list]
    z_levels = [x[0] for x in img_list]
    img_strs = [str(x[1]) + '_' + str(x[2]) + '_' + str(x[3]) for x in img_list]
    for img, z in zip(img_strs, z_levels):
        mask_normal_scale = cv2_imread(data_path + 'Train_set/Train_masks/1/' + str(z) + '_' + img + '.png')
        mask_scaled = cv2.resize(mask_normal_scale, (np.shape(mask_normal_scale)[0] * zoom_factor, np.shape(mask_normal_scale)[1] * zoom_factor), interpolation=cv2.INTER_CUBIC)
        upscaled_image_EM = getEnvelope.data_upscale(data_path, current_zoom, current_zoom-1,img, [(0,0,1024, 1024)], Zlevel=z, use_mask=False, type='EM')[0]
        upscaled_image_HO = getEnvelope.data_upscale(data_path, current_zoom, current_zoom-1,img, [(0,0,1024, 1024)], Zlevel=z, use_mask=False, type='Hoechst')[0]
        upscaled_image_IN = getEnvelope.data_upscale(data_path, current_zoom, current_zoom-1,img, [(0,0,1024, 1024)], Zlevel=z, use_mask=False, type='Insulin')[0]

        upscaled_image = np.dstack((upscaled_image_EM, upscaled_image_HO, upscaled_image_IN))

        IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS = np.shape(upscaled_image)

        crop_data_1 = upscaled_image[0:IMG_WIDTH//2, 0:IMG_HEIGHT//2]
        crop_data_2 = upscaled_image[IMG_WIDTH//2:, 0:IMG_HEIGHT//2]
        crop_data_3 = upscaled_image[0:IMG_WIDTH//2, IMG_HEIGHT//2:]
        crop_data_4 = upscaled_image[IMG_WIDTH//2:, IMG_HEIGHT//2:]

        crop_mask_1 = mask_scaled[0:IMG_WIDTH//2, 0:IMG_HEIGHT//2]
        crop_mask_2 = mask_scaled[IMG_WIDTH//2:, 0:IMG_HEIGHT//2]
        crop_mask_3 = mask_scaled[0:IMG_WIDTH//2, IMG_HEIGHT//2:]
        crop_mask_4 = mask_scaled[IMG_WIDTH//2:, IMG_HEIGHT//2:]

        export_path_data = data_path + 'Train_set/Train_data/1/'
        export_path_masks = data_path + 'Train_set/Train_masks/1/'

        cv2.imwrite(export_path_data + str(z) + '_' + img + '_up1.png', crop_data_1)
        cv2.imwrite(export_path_data + str(z) + '_' + img + '_up2.png', crop_data_2)
        cv2.imwrite(export_path_data + str(z) + '_' + img + '_up3.png', crop_data_3)
        cv2.imwrite(export_path_data + str(z) + '_' + img + '_up4.png', crop_data_4)

        cv2.imwrite(export_path_masks + str(z) + '_' + img + '_up1.png', crop_mask_1)
        cv2.imwrite(export_path_masks + str(z) + '_' + img + '_up2.png', crop_mask_2)
        cv2.imwrite(export_path_masks + str(z) + '_' + img + '_up3.png', crop_mask_3)
        cv2.imwrite(export_path_masks + str(z) + '_' + img + '_up4.png', crop_mask_4)


    return True


if __name__ == '__main__':
    gen_input_from_mask('X:\\BEP_data\\', only_EM=False, adjust_test=False, generated='', dataset='RL010', normalize=True)
