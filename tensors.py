import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from skimage.filters import roberts, sobel, scharr, prewitt
from scipy import ndimage as nd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from tqdm import tqdm
from scipy.stats import pearsonr
import time

import particle_analysis


# Simple read function, grabs an image after sanitizing string input.
def cv2_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


# Put in the size of your 1-dimensional data. Operates on multiples of 1024^2
def get_slice_list(size):
    index_list = []
    for i in range(int(size / (1024 ** 2))):
        print(i)
        if i + 1 == 1:
            index_list.append([0, 1024 ** 2])
        else:
            index_list.append([index_list[-1][1], (i + 1) * 1024 ** 2])
        print(index_list)
    return index_list


def get_paths(ini_path, img_nr):
    EM_Str = ini_path + 'EM/1/{}.png'.format(img_nr)
    Hoechst_Str = ini_path + 'Hoechst/1/{}.png'.format(img_nr)
    Labeled_Str = ini_path + 'Nuclei_masks/1/{}.png'.format(img_nr)
    return EM_Str, Hoechst_Str, Labeled_Str


def add_features(df, img_var, img_string):
    edges = cv2.Canny(img_var, 100, 200)
    edges1 = edges.reshape(-1)
    df['Canny Edge ' + img_string] = edges1

    edge_roberts = roberts(img_var)
    edge_roberts1 = edge_roberts.reshape(-1)
    df['Roberts ' + img_string] = edge_roberts1

    sobel_img = sobel(img_var)
    sobel_img1 = sobel_img.reshape(-1)
    df['Sobel ' + img_string] = sobel_img1

    scharr_img = scharr(img_var)
    scharr_img1 = scharr_img.reshape(-1)
    df['Scharr ' + img_string] = scharr_img1

    prewitt_img = prewitt(img_var)
    prewitt_img1 = prewitt_img.reshape(-1)
    df['Prewitt ' + img_string] = scharr_img1

    gaussian_img = nd.gaussian_filter(img_var, sigma=3)
    gaussian_img1 = gaussian_img.reshape(-1)
    df['Gaussian s3 ' + img_string] = gaussian_img1

    gaussian_img2 = nd.gaussian_filter(img_var, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1)
    df['Gaussian s5 ' + img_string] = gaussian_img3

    median_img = nd.median_filter(img_var, size=3)
    median_img1 = median_img.reshape(-1)
    df['Median s3 ' + img_string] = median_img1
    return df


def apply_gabor(df, kernels, img_var, img_string):
    print('Applying Gabor to ' + img_string + ' image..')
    num = 0
    for kernel in tqdm(kernels):
        fimg = cv2.filter2D(img_var, cv2.CV_8UC3, kernel)
        gabor_label = img_string + 'Gabor' + str(num)
        filtered_img = fimg.reshape(-1)
        df[gabor_label] = filtered_img
        # print(gabor_label)
        num += 1
    return df


def create_dataset(ini_path, img_nr, EM=True, Hoechst=True, Insulin=True, augment=False, Answer=True):
    if not EM and not Hoechst and not Insulin:
        print('Nothing to train_AXIS_3 data from')
        raise exit()
    str_var = ''
    if EM:
        str_var += 'EM, '
    if Hoechst:
        str_var += 'Hoechst FM, '
    if Insulin:
        str_var += 'Insulin FM, '
    print('Creating a feature vector for ' + str_var + 'using image: ' + img_nr)
    EM_Str = ini_path + 'EM/1/{}.png'.format(img_nr)
    Hoechst_Str = ini_path + 'Hoechst/1/{}.png'.format(img_nr)
    Insulin_Str = ini_path + 'Insulin/1/{}.png'.format(img_nr)
    if Answer:
        Labeled_Str = ini_path + 'Nuclei_masks/1/{}.png'.format(img_nr)

    df = pd.DataFrame()

    kernels = []
    for theta in range(2):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for lamda in np.arange(0, np.pi, np.pi / 4):
                for gamma in (0.05, 0.5):
                    ksize = 9
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)
                    kernels.append(kernel)

    if EM:
        EM_img = cv2_imread(EM_Str)
        if augment:
            EM_img = cv2.flip(EM_img, 1)
        EM_img1 = EM_img.reshape(-1)
        df['EM Image'] = EM_img1
        df = apply_gabor(df, kernels, EM_img, 'EM')
        df = add_features(df, EM_img, 'EM')
    if Hoechst:
        ho_img = cv2_imread(Hoechst_Str)
        if augment:
            ho_img = cv2.flip(ho_img, 1)
        ho_img1 = ho_img.reshape(-1)
        df['Hoechst image'] = ho_img1
        df = apply_gabor(df, kernels, ho_img, 'Hoechst')
        df = add_features(df, ho_img, 'Hoechst')
    if Insulin:
        ins_img = cv2_imread(Insulin_Str)
        if augment:
            ins_img = cv2.flip(ins_img, 1)
        ins_img1 = ins_img.reshape(-1)
        df['Insulin image'] = ins_img1
        df = apply_gabor(df, kernels, ins_img, 'Insulin')
        df = add_features(df, ins_img, 'Insulin')
    if Answer:
        labeled_img = cv2_imread(Labeled_Str)
        if augment:
            labeled_img = cv2.flip(labeled_img, 1)
        labeled_img1 = labeled_img.reshape(-1)
        df['Labels'] = labeled_img1
    return df


##############################################################################
# TRAINING
##############################################################################

plotNotImage = False
images_train = ['2_3_3']
images_test = ['1_1_3', '4_1_3', '1_2_3', '2_2_3', '3_2_3', '4_2_3']
image_nr = '1_2_3'
initial_path = 'C:/Users/night/OneDrive/Documenten/TU algemeen/GPU_BEP_PRACTICE/data/'
ini_py_path = 'C:/Users/night/OneDrive/Documenten/TU algemeen/GPU_BEP_PRACTICE/'
EM = True
Hoechst = True
Insulin = False
Augmentation = False

train_df = pd.DataFrame()
for image in images_train:
    dfTemp = create_dataset(initial_path, image, EM, Hoechst, Insulin)
    train_df = pd.concat([train_df, dfTemp], ignore_index=True)
    if Augmentation:
        dfTemp = create_dataset(initial_path, image, EM, Hoechst, Insulin, augment=True)
        train_df = pd.concat([train_df, dfTemp], ignore_index=True)

Y = train_df['Labels'].values
X = train_df.drop(labels=['Labels'], axis=1)

model = RandomForestClassifier(n_estimators=10, random_state=20)
print('Training...')
t = time.time()
model.fit(X, Y)
print('Training completed in {}! Testing...'.format(time.time() - t))

for image in images_test:
    df_test = create_dataset(initial_path, image, EM, Hoechst, Insulin, Answer=False)
    # Y_test = df_test['Labels'].values
    # X_test = df_test.drop(labels=['Labels'], axis=1)
    prediction_test = model.predict(df_test)

    im = Image.fromarray(prediction_test.reshape(-1, 1024))
    im.save('{}_predicted.png'.format(image))
    # print("Pearson value of image {} = ".format(image), pearsonr(Y_test, prediction_test))
    #
    # img_labeled = cv2_imread(initial_path + '/Nuclei_masks/1/{}.png'.format(image))


particle_analysis.ShowResults(ini_py_path, initial_path, images_test)

