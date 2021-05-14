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

def get_radius_sample(mask_directory):

    img_list = glob(mask_directory + '*.png')

    rad_avg = 4

    return rad_avg

if __name__ == '__main__':
    mask_list = glob('X:\\BEP_data\\Train_set\\Train_masks\\1\\*.png')
    for img in mask_list:
        mask = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        mask_expanded = blanket_weights(mask, 255)
        cv2.imwrite('X:\\BEP_data\\Train_set\\Train_masks\\2\\' + img.split('\\')[-1], mask_expanded)