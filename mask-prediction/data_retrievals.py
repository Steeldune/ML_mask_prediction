# import tensorflow as tf
import os
import numpy as np
import cv2


def cv2_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


def gen_input_from_mask(data_path, only_EM=False, adjust_test=False, dataset='RL012', normalize=False):
    """
    From a list of manual mask images, this function collects all the data, and assembles them in either
    the train_set or test_set file structures. The input data is made so that EM data occupies the blue channel and
    secondary data occupies the green channel.
    :param data_path:   file address where Train_set and Test_set and the dataset that is used can be found.
    :param only_EM:     Boolean which determines if only EM data is going to be used, or if there is a secondary data type.
    :param adjust_test: Boolean on whether the Test_set folder will be populated (True) or if the
                        Train_set folder will be populated (False)
    :param dataset:     Specifies the dataset that will be used.
    :param normalize:   Boolean on whether the secondary data will be normalized.
    :return:            True
    """
    file_list = os.listdir(data_path + ('Train_set\\Train_masks\\1\\', 'Test_set\\Test_masks\\1\\')[adjust_test])
    export_path = data_path + ('Train_set\\Train_data\\1\\', 'Test_set\\Test_data\\1\\')[adjust_test]

    for img_string in file_list:
        EM_path = data_path + '{}\\EM\\Collected\\{}'.format(dataset, img_string)
        EM_img = cv2_imread(EM_path)
        if not only_EM:
            HO_path = data_path + ('{}\\Hoechst\\Collected\\{}'.format(dataset, img_string))
            HO_img = cv2_imread(HO_path)
            if normalize:
                HO_img = cv2.normalize(HO_img, None, 255, 0, cv2.NORM_INF)
            TR_img = np.dstack((EM_img, HO_img, np.zeros((1024, 1024))))
        else:
            TR_img = EM_img
        cv2.imwrite(
            export_path + img_string,
            TR_img)
    return True


if __name__ == '__main__':
    gen_input_from_mask('X:\\BEP_data\\', only_EM=False, adjust_test=False, dataset='RL012', normalize=False)

"""I might revisit the idea of various data retrievals later, but this function seems to do."""
