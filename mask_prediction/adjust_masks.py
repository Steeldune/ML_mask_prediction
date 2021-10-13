import cv2
import numpy as np
import random
from skimage.morphology import disk
from glob import glob
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import shutil


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
    data_path = 'X:\\BEP_data\\Data_External\\RL015\\Hoechst'
    target_path = 'X:\\BEP_data\\Data_External\\RL015\\Hoechst\\Collected\\'
    for i in range(1, 6):
        img_folder_path = data_path + '\\{}\\'.format(str(i))
        img_list = glob(img_folder_path+ '*2.png')
        for img in img_list:
            img_name = img.split('\\')[-1]
            shutil.copy(img, target_path + '{}_'.format(str(i)) + img_name)