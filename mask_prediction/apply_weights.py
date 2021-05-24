import cv2
import numpy as np
from glob import glob


def blanket_weights(mask, c, IMG_SIZE=1024):
    """
    Takes a single channel mask image, adds two channels. One filled with c, one filled with zeroes.
    These are images, so values range from 0 to 255 in integer amounts.
    :param mask:    Image that will be affected
    :param c:       Uniform pixel value of the second channel
    :param IMG_SIZE:Size of the image in pixels.
    :return:        Mask image with uniform second and third channels.
    """

    new_mask_1 = np.ones((IMG_SIZE, IMG_SIZE), dtype=int) * c
    new_mask_2 = np.zeros((IMG_SIZE, IMG_SIZE), dtype=int)
    new_mask = np.dstack([mask, new_mask_1, new_mask_2])
    return new_mask


def sample_points(pts_array, ratio):
    rng = np.random.default_rng()
    list_len = len(pts_array)
    sample_size = int(ratio*list_len)
    sample_array = rng.choice(pts_array, sample_size, replace=False, axis=0)
    return sample_array


def get_cluster_center(in_list):
    if len(in_list) == 1:
        return in_list[0]
    pos_sum = np.sum(in_list, axis=0)
    return pos_sum / (np.shape(in_list)[0])


def get_radius_sample(mask_directory):
    img_list = glob(mask_directory + '*.png')
    diameters = []
    for img in img_list:
        img_read = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        contours, hierarchy = cv2.findContours(img_read, 1, 2)
        bounds = [cv2.boundingRect(cnt) for cnt in contours]
        for bound in bounds:
            if bound[3] + bound[2] >= 20:
                diameters.append((bound[3] + bound[2]) // 2)

    return np.array(diameters)


def get_annotated_control_points(image, thresh=False):
    if thresh:
        ret, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_thresh = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]

    mass_centres = np.zeros((len(contours_thresh), 2), dtype=int)

    for i in range(0, len(contours_thresh)):
        M = cv2.moments(contours_thresh[i], 0)
        mass_centres[i, 0] = (int(M['m10'] / M['m00']))
        mass_centres[i, 1] = (int(M['m01'] / M['m00']))

    if np.shape(mass_centres)[0] == 1:
        return [get_cluster_center(mass_centres)], [1, 2]
    return mass_centres


def add_circles(ctrl_points, diameter, ret_img=None, IMG_SIZE=1024):
    if ret_img is None:
        ret_img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=int)
    for pnt in ctrl_points:
        if len(pnt) == 2:
            cv2.circle(ret_img, (pnt[0], pnt[1]), diameter, 255, thickness=-1)
    return ret_img


def add_gauss(image, ctrl_points, diameter, IMG_SIZE=1024):
    x = np.linspace(0, IMG_SIZE - 1, IMG_SIZE, dtype=int)
    xv, yv = np.meshgrid(x, x)
    out_image = np.zeros((IMG_SIZE, IMG_SIZE))
    for pnt in ctrl_points:
        if len(pnt) !=2:
            continue
        xv_dist = np.abs(xv - pnt[0])
        yv_dist = np.abs(yv - pnt[1])
        dist_map = (np.power(xv_dist, 2) + np.power(yv_dist, 2))
        dist_map_thresh = dist_map < np.power(diameter // 2, 2)

        gauss_map = np.exp((dist_map * -1) / (diameter * 2)) * 255

        out_image = out_image + gauss_map * dist_map_thresh

    return out_image


def gen_mask_with_weights(mask, ini_weight, diameter, IMG_SIZE=1024, pnt_ratio=1.0, thresh=False):
    mask_img = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    if type(ini_weight) is str:
        ini_weight_img = cv2.imread(ini_weight, cv2.IMREAD_GRAYSCALE)
    else:
        ini_weight_img = ini_weight
    ctrl_pnts = get_annotated_control_points(mask_img, thresh=thresh)

    if pnt_ratio < 0.99:
        ctrl_pnts = sample_points(ctrl_pnts, pnt_ratio)

    weights_img = add_circles(ctrl_pnts, diameter)
    gauss_img = add_gauss(mask_img, ctrl_pnts, diameter)
    weights_img_2 = add_circles(ctrl_pnts, diameter//2)
    final_img = np.dstack([gauss_img, weights_img, weights_img_2])
    return final_img


def convert_partial_annotation(file_address, export_address, diameter, IMG_SIZE=1024, thresh=False, filter_zero=False):
    mask_list = glob(file_address + '\\*.png')
    for img in mask_list:
        img_name = img.split('\\')[-1]
        mask_img = gen_mask_with_weights(img, np.zeros((IMG_SIZE, IMG_SIZE), dtype=int), diameter, IMG_SIZE=IMG_SIZE, pnt_ratio=.8, thresh=thresh)
        if filter_zero:
            if np.sum(mask_img) > 50000000:
                cv2.imwrite('{}\\{}'.format(export_address, img_name), mask_img)
        else:
            cv2.imwrite('{}\\{}'.format(export_address, img_name), mask_img)
    return mask_list



if __name__ == '__main__':
    IMG_SIZE = 1024

    # mask_list = glob('X:\\BEP_data\\Train_set\\Train_masks\\1\\*.png')
    # for img in mask_list:
    #     mask = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    #     mask_expanded = blanket_weights(mask, 255)
    #     cv2.imwrite('X:\\BEP_data\\Train_set\\Train_masks\\2\\' + img.split('\\')[-1], mask_expanded)
    radius_array = get_radius_sample('X:\\BEP_data\\Train_set\\blobs\\1\\')
    mean_diam = np.mean(radius_array, dtype=int)
    print(mean_diam)

    print(convert_partial_annotation('X:\\BEP_data\\Train_set\\blobs\\2', 'X:\\BEP_data\\Partial Annotation\\RL012\\pred_hoechst', 90, thresh=True, filter_zero=True))

    # ex_img = gen_mask_with_weights('X:\\BEP_data\\Train_set\\blobs\\1\\1_3_1_3.png', np.zeros((1024, 1024), dtype=int),
    #                                mean_diam)
    # cv2.imshow('final', ex_img / 255)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
