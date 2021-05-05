import cv2
import numpy as np
import random
from skimage.morphology import disk
from glob import glob
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


def cv2_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


def erode_dilate(img_path, kernel=disk(3)):
    img = cv2_imread(img_path)
    r = random.randint(1, 3)
    if r == 1:
        img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
    elif r == 2:
        img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
    else:
        return img
    return img


def floodfill_random(img_path, iterations=1):
    img = cv2_imread(img_path)
    for i in range(iterations):
        r1 = random.randint(1, 1023)
        r2 = random.randint(1, 1023)
        _, img, _, _ = cv2.floodFill(img, None, (r1, r2), 0)
    return img

def holes_random(img_path, iterations=1):
    img = cv2_imread(img_path)
    for i in range(iterations):
        r1 = random.randint(1, 1023)
        r2 = random.randint(1, 1023)
        img = cv2.circle(img, (r1,r2), 6, 0, -1)
    return img

def elastic_transform(img_path, alpha=200, sigma=20):
    image = cv2_imread(img_path)

    # Gaussian filter some noise
    dx = gaussian_filter((np.random.rand(*image.shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*image.shape) * 2 - 1), sigma) * alpha

    # Create distortion grid
    x, y = np.meshgrid(np.arange(image.shape[1]),
                          np.arange(image.shape[0]))
    indices = (np.reshape(y+dy, (-1, 1)),
               np.reshape(x+dx, (-1, 1)))
    transformed = map_coordinates(image, indices, order=1, mode='reflect')

    return transformed.reshape(image.shape)


if __name__ == '__main__':
    img_file = 'X:\\BEP_data\\RL012\\Manual Masks\\'
    pred_file = 'X:\\BEP_data\\Predict_set\\Output\\'
    img_list = glob(img_file + '4_3_2_3*')

    # cv2.imshow('Image 1', img1)
    for img_str in img_list:
        img1 = cv2_imread(img_str)

        # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

        img2 = holes_random(img_str, iterations=40)

        img3 = img1/255 * img2/255
        img3 = np.multiply(img3, 255.0)

        img4 = np.dstack((img1, img3.astype(np.uint8), img2))

        cv2.imwrite(img_str.split('.')[0] + '_puncture_40.png', img4)

    # database = 'RL012'
    #
    # img_list = glob('X:\\BEP_data\\{}\\EM\\Collected\\[1-9]_[1-4]_[1-4]_*.png'.format(database))
    # for img in img_list:
    #     img_EM = cv2_imread(img)
    #     img_HO = cv2_imread('X:\\BEP_data\\{}\\Hoechst\\Collected\\'.format(database) + img.split('\\')[-1])
    #
    #     _, img_EM_thresh = cv2.threshold(img_EM, 2, 1, cv2.THRESH_BINARY)
    #     cv2.imwrite('X:\\BEP_data\\{}\\Hoechst_Thresh\\Collected\\'.format(database) + img.split('\\')[-1], img_HO * img_EM_thresh)