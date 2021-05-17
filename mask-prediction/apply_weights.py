import cv2
import numpy as np
from glob import glob


def blanket_weights(mask, c, img_size=1024):
    """
    Takes a single channel mask image, adds two channels. One filled with c, one filled with zeroes.
    These are images, so values range from 0 to 255 in integer amounts.
    :param mask:    Image that will be affected
    :param c:       Uniform pixel value of the second channel
    :param img_size:Size of the image in pixels.
    :return:        Mask image with uniform second and third channels.
    """

    new_mask_1 = np.ones((img_size, img_size), dtype=int) * c
    new_mask_2 = np.zeros((img_size, img_size), dtype=int)
    new_mask = np.dstack([mask, new_mask_1, new_mask_2])
    return new_mask


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


def get_annotated_control_points(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mass_centres = np.zeros((len(contours), 2), dtype=int)

    for i in range(0, len(contours)):
        M = cv2.moments(contours[i], 0)
        mass_centres[i, 0] = (int(M['m10'] / M['m00']))
        mass_centres[i, 1] = (int(M['m01'] / M['m00']))

    if np.shape(mass_centres)[0] == 1:
        return [get_cluster_center(mass_centres)], [1, 2]
    return mass_centres


def add_circles(ctrl_points, diameter):
    ret_img = np.zeros((1024, 1024), dtype=int)
    for pnt in ctrl_points:
        cv2.circle(ret_img, pnt, diameter, 255, thickness=-1)
    return ret_img


def add_gauss(image, ctrl_points, diameter):
    x = np.linspace(0, 1023, 1024, dtype=int)
    xv, yv = np.meshgrid(x, x)
    out_image = np.zeros((1024, 1024))
    for pnt in ctrl_points:
        xv_dist = np.abs(xv - pnt[0])
        yv_dist = np.abs(yv - pnt[1])
        dist_map = (np.power(xv_dist, 2) + np.power(yv_dist, 2))
        dist_map_thresh = dist_map < np.power(diameter // 2, 2)

        gauss_map = np.exp((dist_map * -1) / (diameter * 2)) * 255

        out_image = out_image + gauss_map * dist_map_thresh

    return out_image


if __name__ == '__main__':
    # mask_list = glob('X:\\BEP_data\\Train_set\\Train_masks\\1\\*.png')
    # for img in mask_list:
    #     mask = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    #     mask_expanded = blanket_weights(mask, 255)
    #     cv2.imwrite('X:\\BEP_data\\Train_set\\Train_masks\\2\\' + img.split('\\')[-1], mask_expanded)
    radius_array = get_radius_sample('X:\\BEP_data\\Train_set\\blobs\\1\\')
    print(np.mean(radius_array, dtype=int))
    example_img = cv2.imread('X:\\BEP_data\\Train_set\\blobs\\1\\1_3_1_3.png', cv2.IMREAD_GRAYSCALE)
    ctrl_list = get_annotated_control_points(example_img)
    example_img_2 = add_circles(ctrl_list, 98)
    gauss_img = add_gauss(example_img, ctrl_list, 98)
    final_img = np.dstack([example_img_2, gauss_img, np.zeros((1024, 1024), dtype=int)])
    cv2.imshow('final', final_img / 255)
    cv2.waitKey()
    cv2.destroyAllWindows()
