import os
import numpy as np
import cv2


def cv2_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


def gen_input_from_mask(data_paths, work_with_fm=True, adjust_test=False, normalize=False):
    """
    From a list of manual mask images, this function collects all the data, and assembles them in either
    the train_set or test_set file structures. The input data is made so that EM data occupies the blue channel and
    secondary data occupies the green channel.
    :param data_path:   file address where Train_set and Test_set and the dataset that is used can be found.
    :param work_with_fm:     Boolean which determines if only EM data is going to be used, or if there is a secondary data type.
    :param adjust_test: Boolean on whether the Test_set folder will be populated (True) or if the
                        Train_set folder will be populated (False)
    :param dataset:     Specifies the dataset that will be used.
    :param normalize:   Boolean on whether the secondary data will be normalized.
    :return:            True
    """

    train_path, test_path, em_path, ho_path, mask_folder = data_paths

    file_list = os.listdir((train_path, test_path)[adjust_test] + ('\\masks\\1\\', '\\masks\\1\\')[adjust_test])
    export_path = (train_path, test_path)[adjust_test] + ('\\data\\1\\', '\\data\\1\\')[adjust_test]
    export_list = []
    for img_string in file_list:
        EM_path = em_path + '\\{}'.format(img_string)
        EM_img = cv2_imread(EM_path)
        if work_with_fm:
            HO_path = ho_path + ('\\{}'.format(img_string))
            HO_img = cv2_imread(HO_path)
            if normalize:
                HO_img = cv2.normalize(HO_img, None, 255, 0, cv2.NORM_INF)
            TR_img = np.dstack((EM_img, HO_img, np.zeros((1024, 1024))))
        else:
            TR_img = EM_img
        cv2.imwrite(export_path + img_string, TR_img)
        export_list.append(export_path + img_string)
    return export_list


if __name__ == '__main__':
    gen_input_from_mask('X:\\BEP_data\\', work_with_fm=True, adjust_test=True, normalize=True)

"""I might revisit the idea of various data retrievals later, but this function seems to do."""
