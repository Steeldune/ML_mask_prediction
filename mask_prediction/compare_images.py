import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import glob
import cv2
from skimage.morphology import disk
import numpy as np
from tqdm import tqdm


def output_IOU(file_ground, file_pred):
    file_list_ground = glob.glob(file_ground + '/*.png')
    file_list_pred = glob.glob(file_pred + '/*.png')

    imgs_ground = [x.split('\\')[-1] for x in file_list_ground]
    imgs_pred = [x.split('\\')[-1] for x in file_list_pred]
    imgs_list = list(set(imgs_ground) & set(imgs_pred))

    iou_dict = {}
    pearson_dict = {}

    for img_str in tqdm(imgs_list):
        ground_img = cv2.imread(file_ground + '\\' + img_str)
        ground_img = cv2.cvtColor(ground_img, cv2.COLOR_BGR2GRAY)
        _, ground_img = cv2.threshold(ground_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        pred_img = cv2.imread(file_pred + '\\' + img_str)
        pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2GRAY)
        _, pred_img = cv2.threshold(pred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        pred_img = cv2.morphologyEx(pred_img, cv2.MORPH_CLOSE, disk(5))

        intersect_img = cv2.bitwise_and(pred_img, ground_img)
        intersect_img = intersect_img > 127
        union_img = cv2.bitwise_or(pred_img, ground_img)
        union_img = union_img > 127

        iou_dict[img_str] = np.sum(intersect_img) / np.sum(union_img)
        pearson_dict[img_str] = stats.pearsonr(ground_img.reshape(-1), pred_img.reshape(-1))[0]

    return iou_dict, pearson_dict


if __name__ == '__main__':
    man_mask_folder = 'X:\\BEP_data\\Data_External\\RL012\\Manual_Masks'
    pred_out_folder = 'X:\\BEP_data\\Data_Internal\\Train_Masks\\Predict_backups\\pancreas_final_train_final_2021-11-29_19-09-39\\Output'
    gen_out_folder = 'X:\\BEP_data\\Data_Internal\\Gen_Masks\\Generated_backups\\pancreas_130_last_2021-11-29_17-23-25\\Output'
    qu_out_folder = 'X:\\BEP_data\\Data_Internal\\Qu_Iteration\\Predict_backups\\pancreas_it10_sig25_diam13010_2021-11-29_17-04-52\\Output'

    pred_iou_dict, _ = output_IOU(man_mask_folder, pred_out_folder)
    gen_iou_dict, _ = output_IOU(man_mask_folder, gen_out_folder)
    qu_iou_dict, _ = output_IOU(man_mask_folder, qu_out_folder)

    item_iou_list = set(pred_iou_dict.keys()) & set(gen_iou_dict.keys()) & set(qu_iou_dict.keys())
    for item in sorted(item_iou_list):
        print('Image {} has a Qu IOU {:.2f}, Gen IOU {:.2f}, pred IOU {:.2f}, difference {:.2f}'.format(item,
                                                                                                        qu_iou_dict[
                                                                                                            item],
                                                                                                        gen_iou_dict[
                                                                                                            item],
                                                                                                        pred_iou_dict[
                                                                                                            item],
                                                                                                        pred_iou_dict[
                                                                                                            item] -
                                                                                                        gen_iou_dict[
                                                                                                            item]))
